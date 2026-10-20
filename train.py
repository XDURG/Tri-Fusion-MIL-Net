import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from model import TriFusionMIL


BRANCHES = ("axial", "coronal", "sagittal", "patches")


def load_case(path):
    with np.load(path, allow_pickle=False) as saved:
        case = {key: torch.from_numpy(np.array(saved[key], dtype=np.float32)) for key in BRANCHES}
    for key, value in case.items():
        ndim = 5 if key == "patches" else 4
        if value.ndim != ndim or value.shape[0] == 0 or value.shape[1] != 1 or not torch.isfinite(value).all():
            raise ValueError(f"Invalid {key} array in {path}")
    return case


class CaseDataset(Dataset):
    def __init__(self, csv_path, require_labels=False):
        csv_path = Path(csv_path).resolve()
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            self.rows = list(csv.DictReader(stream))
        self.root = csv_path.parent
        if not self.rows:
            raise ValueError("Empty case list")
        ids = [row["case_id"] for row in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Repeated case_id")
        if require_labels and any(row.get("label") not in ("0", "1") for row in self.rows):
            raise ValueError("Training and validation labels must be 0 or 1")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.root / row["path"]
        label = float(row["label"]) if row.get("label") else float("nan")
        return load_case(path), label, row["case_id"]


def collate_cases(items):
    batch = {}
    for key in BRANCHES:
        arrays = [item[0][key] for item in items]
        if len({tuple(x.shape[1:]) for x in arrays}) != 1:
            raise ValueError(f"Spatial shapes must match within a batch: {key}")
        count = max(len(x) for x in arrays)
        batch[key] = arrays[0].new_zeros(len(items), count, *arrays[0].shape[1:])
        batch[key + "_valid"] = torch.zeros(len(items), count, dtype=torch.bool)
        for i, value in enumerate(arrays):
            batch[key][i, :len(value)] = value
            batch[key + "_valid"][i, :len(value)] = True
    labels = torch.tensor([item[1] for item in items], dtype=torch.float32)
    return batch, labels, [item[2] for item in items]


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def get_device(name):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def combined_loss(logits, labels):
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    p = logits.sigmoid()
    pt = p * labels + (1 - p) * (1 - labels)
    alpha = 0.25 * labels + 0.75 * (1 - labels)
    focal = alpha * (1 - pt).pow(2) * bce
    return 0.6 * bce.mean() + 0.4 * focal.mean()


def validate(model, loader, device):
    model.eval()
    total = 0
    labels = []
    scores = []
    with torch.inference_mode():
        for batch, y, _ in loader:
            y = y.to(device)
            logits = model(move_batch(batch, device))["logits"]
            total += combined_loss(logits, y).item() * len(y)
            labels.extend(y.cpu().tolist())
            scores.extend(logits.sigmoid().cpu().tolist())
    return total / len(loader.dataset), roc_auc_score(labels, scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--out", default="runs")
    parser.add_argument("--init")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = get_device(args.device)
    training = CaseDataset(args.train, require_labels=True)
    validation = CaseDataset(args.val, require_labels=True)
    train_ids = {row["case_id"] for row in training.rows}
    val_ids = {row["case_id"] for row in validation.rows}
    train_paths = {(training.root / row["path"]).resolve() for row in training.rows}
    val_paths = {(validation.root / row["path"]).resolve() for row in validation.rows}
    if train_ids & val_ids or train_paths & val_paths:
        raise ValueError("Training and validation sets overlap")
    if {row["label"] for row in validation.rows} != {"0", "1"}:
        raise ValueError("Validation AUROC requires both classes")
    parallel = device.type == "cuda" and device.index is None and torch.cuda.device_count() > 1
    batch_size = args.batch_size * (torch.cuda.device_count() if parallel else 1)
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True, collate_fn=collate_cases)
    val_loader = DataLoader(validation, batch_size=batch_size, collate_fn=collate_cases)
    model = TriFusionMIL(pretrained=args.pretrained and not args.init)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=True))
    model = model.to(device)
    if parallel:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = -1
    stale = 0
    for epoch in range(args.epochs):
        model.train()
        total = 0
        for batch, y, _ in train_loader:
            batch = move_batch(batch, device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = combined_loss(model(batch)["logits"], y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(y)
        val_loss, auc = validate(model, val_loader, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch + 1} train_loss={total / len(training):.5f} val_loss={val_loss:.5f} val_auc={auc:.5f}", flush=True)
        if auc > best:
            best = auc
            stale = 0
            net = model.module if parallel else model
            torch.save({key: value.detach().cpu() for key, value in net.state_dict().items()}, out / "best.pt")
        else:
            stale += 1
        if stale >= 10:
            break
    print(out / "best.pt")


if __name__ == "__main__":
    main()
