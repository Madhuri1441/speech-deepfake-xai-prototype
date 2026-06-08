import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_curve

# =========================
# Dataset
# =========================
class NpyDataset(Dataset):
    def __init__(self, folder, mean=None, std=None):
        self.files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".npy")
        ]
        self.files.sort()

        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        feat = np.load(path).astype(np.float32)

        if self.mean is not None:
            feat = (feat - self.mean) / self.std

        feat = torch.tensor(feat)

        filename = os.path.basename(path)

        if "CON" in filename:
            label = 0
        elif "LA" in filename:
            label = 1
        else:
            raise ValueError(filename)

        return feat, torch.tensor(label), filename


# =========================
# Compute normalization stats
# =========================
def compute_stats(folder):
    feats = []
    for f in os.listdir(folder):
        if f.endswith(".npy"):
            feats.append(np.load(os.path.join(folder, f)))

    feats = np.stack(feats)
    mean = feats.mean(axis=0)
    std = feats.std(axis=0) + 1e-6
    return mean, std


# =========================
# Model (improved MLP)
# =========================
class MLPDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(88, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Evaluation
# =========================
def evaluate(loader, model, device, name, print_or=True):
    model.eval()

    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for x, y, fname in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_score.extend(probs[:, 1].cpu().numpy())

            if print_or:
                for i in range(len(fname)):
                    if pred[i].item() == y[i].item():
                        print(fname[i])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    acc = np.mean(y_true == y_pred)
    f1 = f1_score(y_true, y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

    print(f"\n{name} RESULTS")
    print(f"Samples: {len(y_true)}")
    print(f"ACC: {acc:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"EER: {eer:.4f}")

    return acc


# =========================
# Main
# =========================
if __name__ == "__main__":

    feature_root = "./features"

    mean, std = compute_stats(os.path.join(feature_root, "train"))

    train_loader = DataLoader(
        NpyDataset(os.path.join(feature_root, "train"), mean, std),
        batch_size=64,
        shuffle=True
    )

    valid_loader = DataLoader(
        NpyDataset(os.path.join(feature_root, "dev"), mean, std),
        batch_size=64
    )

    test_loader = DataLoader(
        NpyDataset(os.path.join(feature_root, "eval"), mean, std),
        batch_size=64
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = MLPDetector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    best_valid_acc = 0
    epochs = 30

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"\nEpoch {epoch} Loss: {total_loss:.4f}")

        valid_acc = evaluate(valid_loader, model, device, "VALID", False)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), "best_mlp_opensmile.pt")
            print("Best model saved.")

    print("\nLoading best model...")
    model.load_state_dict(torch.load("best_mlp_opensmile.pt"))

    evaluate(test_loader, model, device, "TEST")
    evaluate(train_loader, model, device, "TRAIN")
    evaluate(valid_loader, model, device, "VALID")



exit()
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_curve

# =========================
# Dataset
# =========================
class NpyDataset(Dataset):
    def __init__(self, folder):
        self.files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".npy")
        ]
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        feat = np.load(path).astype(np.float32)
        feat = torch.tensor(feat)

        filename = os.path.basename(path)

        if "CON" in filename:
            label = 0
        elif "LA" in filename:
            label = 1
        else:
            raise ValueError(filename)

        return feat, torch.tensor(label), filename


# =========================
# Model
# =========================
class MLPDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(88, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Evaluation
# =========================
def evaluate(loader, model, device, name, print_or = True):
    model.eval()

    print(f"evaluating {name}")

    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for x, y, fname in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_score.extend(probs[:, 1].cpu().numpy())
            
            if print_or:
                for i in range(len(fname)):
                    if pred[i].item() == y[i].item():
                        print(fname[i])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    acc = np.mean(y_true == y_pred)
    f1 = f1_score(y_true, y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

    print(f"\n{name} RESULTS")
    print(f"Samples: {len(y_true)}")
    print(f"ACC: {acc:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"EER: {eer:.4f}")

    return acc


# =========================
# Main
# =========================
if __name__ == "__main__":

    feature_root = "./features"

    train_loader = DataLoader(
        NpyDataset(os.path.join(feature_root, "train")),
        batch_size=64,
        shuffle=True
    )

    valid_loader = DataLoader(
        NpyDataset(os.path.join(feature_root, "dev")),
        batch_size=64
    )

    test_loader = DataLoader(
        NpyDataset(os.path.join(feature_root, "eval")),
        batch_size=64
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = MLPDetector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_valid_acc = 0
    epochs = 20

    # =========================
    # Training loop
    # =========================
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y,_ in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        
        print(f"\nEpoch {epoch} Loss: {total_loss:.4f}")

        # Evaluate on TRAIN and VALID
        
        valid_acc = evaluate(valid_loader, model, device, "VALID", False)

        # Save best model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), "best_mlp_opensmile.pt")
            print("Best model saved.")
        
    # =========================
    # Final Test
    # =========================
    print("\nLoading best model...")

    model.load_state_dict(torch.load("best_mlp_opensmile.pt"))

    evaluate(test_loader, model, device, "TEST")
    evaluate(train_loader, model, device, "TRAIN")
    evaluate(valid_loader, model, device, "VALID")
