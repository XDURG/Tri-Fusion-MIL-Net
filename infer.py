import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import TriFusionMIL
from train import CaseDataset, collate_cases, get_device, load_case, move_batch


def load_model(weights, device):
    model = TriFusionMIL()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--weights", default="weights.pt")
    parser.add_argument("--out", default="predictions.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    torch.set_num_threads(4)
    device = get_device(args.device)
    model = load_model(args.weights, device)
    source = Path(args.input)
    if source.suffix.lower() == ".npz":
        dataset = [(load_case(source), float("nan"), source.stem)]
    else:
        dataset = CaseDataset(source)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_cases)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as stream, torch.inference_mode():
        writer = csv.writer(stream)
        writer.writerow(("case_id", "carcinoma_probability", "prediction"))
        for batch, _, ids in loader:
            scores = model(move_batch(batch, device))["logits"].sigmoid().cpu().tolist()
            for case_id, score in zip(ids, scores):
                writer.writerow((case_id, score, int(score >= args.threshold)))
    print(out)


if __name__ == "__main__":
    main()
