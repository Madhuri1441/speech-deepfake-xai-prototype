"""
json_builder.py

Converts pixel-space regions (from region_detector.py) into physical
units (seconds, Hz), using the metadata sidecar written by
hubert2xai.py, and assembles the final structured evidence JSON that
the LLM prompt builder consumes.

Row/column convention (must match hubert2xai.py's plotting):
  - Array shape is (freq_bins, time_frames), row 0 = lowest frequency.
  - imshow uses origin="lower", so row index maps directly to
    frequency (no vertical flip needed).
  - extent=[0, duration_sec, 0, max_freq_hz] means both axes are
    treated as linear across the full array, matching the STFT/LIME
    map layout produced upstream.
"""

import json
import os


def pixel_region_to_physical(region, map_shape, duration_sec, max_freq_hz):
    """
    region: one dict from region_detector.detect_regions() /
        attach_supporting_scores() — pixel space, may include
        "supporting_scores" (passed through unchanged if present).
    map_shape: (freq_bins, time_frames) — from the metadata sidecar
    """
    h, w = map_shape
    time_per_col = duration_sec / w
    freq_per_row = max_freq_hz / h

    time_start = region["col_start"] * time_per_col
    time_end = (region["col_end"] + 1) * time_per_col
    freq_low = region["row_start"] * freq_per_row
    freq_high = (region["row_end"] + 1) * freq_per_row

    centroid_time = region["centroid_col"] * time_per_col
    centroid_freq = region["centroid_row"] * freq_per_row

    physical = {
        "rank": region["rank"],
        "time_start": round(time_start, 3),
        "time_end": round(time_end, 3),
        "freq_low": round(freq_low, 1),
        "freq_high": round(freq_high, 1),
        "centroid": {
            "time": round(centroid_time, 3),
            "frequency": round(centroid_freq, 1),
        },
        "importance": round(region["importance"], 3),
        "mean_importance": round(region["mean_importance"], 3),
        "max_importance": round(region["max_importance"], 3),
        "area_px": region["area_px"],
    }

    if "supporting_scores" in region:
        physical["supporting_scores"] = {
            source_name: {
                "mean": round(scores["mean"], 3),
                "max": round(scores["max"], 3),
            }
            for source_name, scores in region["supporting_scores"].items()
        }

    return physical


def build_summary(all_regions):
    """
    all_regions: the FULL (unsliced) region list from
        region_detector.detect_regions(), in pixel space — not just
        the top-K kept for the "regions" field. This is what lets the
        verifier later ask "how much evidence existed", not just
        "what were the top regions".
    """
    if not all_regions:
        return {
            "num_regions": 0,
            "largest_region_area": 0,
            "total_evidence_area": 0,
            "max_importance": 0.0,
        }

    return {
        "num_regions": len(all_regions),
        "largest_region_area": max(r["area_px"] for r in all_regions),
        "total_evidence_area": sum(r["area_px"] for r in all_regions),
        "max_importance": round(max(r["importance"] for r in all_regions), 3),
    }


def build_evidence_json(meta, top_regions, all_regions, map_shape):
    """
    meta: the dict loaded from the meta/<base_name>.json sidecar
          (must contain 'filename', 'prediction', 'duration_sec',
          'max_freq_hz')
    top_regions: PIXEL-space regions to include in the "regions"
        field (typically detect_regions()[0][:top_k], after
        attach_supporting_scores()).
    all_regions: the FULL PIXEL-space region list (unsliced), used
        only to compute "summary" stats.
    map_shape: (freq_bins, time_frames)

    Returns the final JSON-serializable evidence dict:
        {
          "filename": ...,
          "prediction": ...,
          "summary": {num_regions, largest_region_area,
                       total_evidence_area, max_importance},
          "regions": [ {rank, time_start, time_end, freq_low,
                         freq_high, centroid, importance,
                         mean_importance, max_importance, area_px,
                         supporting_scores}, ... ]
        }
    """
    duration_sec = meta["duration_sec"]
    max_freq_hz = meta["max_freq_hz"]

    return {
        "filename": meta["filename"],
        "prediction": meta["prediction"],
        "summary": build_summary(all_regions),
        "regions": [
            pixel_region_to_physical(r, map_shape, duration_sec, max_freq_hz)
            for r in top_regions
        ],
    }


def save_evidence_json(evidence, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(evidence, f, indent=2)
