"""
region_detector.py

Finds contiguous "evidence" regions using the CGXA consensus map as
the PRIMARY importance signal (not the individual refined maps) —
consensus is the fused, final attribution output, so it should be
what determines ranking. The refined maps (IG/Saliency/LIME) are
attached separately, per region, as supporting evidence via
attach_supporting_scores() — they inform the explanation but don't
compete with consensus for ranking.

Pure numpy/scipy — no knowledge of time/frequency units. That
conversion happens one layer up, in json_builder.py.
"""

import numpy as np
from scipy import ndimage

# Regions smaller than this (in pixels) are treated as noise, not
# genuine evidence. Tune this per your STFT resolution.
MIN_REGION_PIXELS = 6

# importance = MEAN_WEIGHT * mean(region) + MAX_WEIGHT * max(region)
# Pure max lets a single hot pixel outrank a broad, consistently
# strong region — mean_weight biases toward regions that are
# strong throughout, not just at one peak.
MEAN_WEIGHT = 0.6
MAX_WEIGHT = 0.4


def detect_regions(
    consensus_map,
    mask,
    min_region_pixels=MIN_REGION_PIXELS,
    mean_weight=MEAN_WEIGHT,
    max_weight=MAX_WEIGHT,
):
    """
    consensus_map : 2D float array, values in [0, 1] — the CGXA fused
                     consensus map (results["consensus"]). This is the
                     PRIMARY importance signal.
    mask          : 2D bool array, the CGXA consensus threshold mask
                     (results["mask"]) — defines region boundaries.

    Returns (regions, labeled):
      regions: ALL regions passing min_region_pixels, ranked by
        importance descending (rank 1..N — NOT sliced to top-K; the
        caller decides how many to keep and can use the full list
        for summary statistics). Each region is a dict in PIXEL
        space:
            {
              "rank": int,
              "label_id": int,          # internal use only —
                                         # strip before JSON output
              "row_start", "row_end",
              "col_start", "col_end": int,
              "centroid_row", "centroid_col": float,
              "importance": float,       # 0.6*mean + 0.4*max
              "mean_importance": float,
              "max_importance": float,
              "area_px": int,
            }
      labeled: the connected-component label array (same shape as
        mask). Callers use this + label_id to pull the exact region
        mask again later, e.g. to compute supporting IG/Saliency/LIME
        scores over the identical pixels.
    """
    if consensus_map.shape != mask.shape:
        raise ValueError(
            f"consensus_map shape {consensus_map.shape} != mask shape {mask.shape}"
        )

    labeled, num_features = ndimage.label(mask)

    regions = []
    for label_id in range(1, num_features + 1):
        region_mask = labeled == label_id
        area_px = int(region_mask.sum())
        if area_px < min_region_pixels:
            continue

        ys, xs = np.where(region_mask)
        values = consensus_map[region_mask]

        mean_v = float(values.mean())
        max_v = float(values.max())
        importance = mean_weight * mean_v + max_weight * max_v

        regions.append({
            "label_id": int(label_id),
            "row_start": int(ys.min()),
            "row_end": int(ys.max()),
            "col_start": int(xs.min()),
            "col_end": int(xs.max()),
            "centroid_row": float(ys.mean()),
            "centroid_col": float(xs.mean()),
            "importance": importance,
            "mean_importance": mean_v,
            "max_importance": max_v,
            "area_px": area_px,
        })

    regions.sort(key=lambda r: r["importance"], reverse=True)
    for rank, region in enumerate(regions, start=1):
        region["rank"] = rank

    return regions, labeled


def attach_supporting_scores(regions, labeled, source_maps):
    """
    Mutates `regions` in place, adding a "supporting_scores" dict to
    each region: per-source (IG/Saliency/LIME) mean+max values over
    the EXACT same pixels as the region (re-derived from `labeled`
    and each region's "label_id"), not just its bounding box.

    source_maps: dict like
        {
          "refined_ig": <2D array>,
          "refined_saliency": <2D array>,
          "refined_lime": <2D array>,
        }
    """
    for region in regions:
        region_mask = labeled == region["label_id"]
        scores = {}
        for source_name, source_map in source_maps.items():
            if source_map.shape != labeled.shape:
                raise ValueError(
                    f"{source_name} shape {source_map.shape} != mask shape {labeled.shape}"
                )
            values = source_map[region_mask]
            scores[source_name] = {
                "mean": float(values.mean()),
                "max": float(values.max()),
            }
        region["supporting_scores"] = scores
    return regions
