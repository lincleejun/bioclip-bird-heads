"""Is the bird flying? One linear layer on the mean of the teacher's block-15 patch tokens.

  ./run ../bioclip-bird-heads/flight.py --out runs/flight-1      # from a bioclip-bird checkout

The final image vector is built to ignore pose (a linear layer on it calls perched gulls and egrets "flying":
it learns which species were photographed flying); block 15 of 32 still carries it. It is the block the eye and
beak head reads, so identification's one pass feeds all three.
Data: fetch_flight.py + bioclip-bird's distill/embed.py -> data/flight (fly/ and perch/, observers' own labels).
Two tests, both on photos kept out of training: one species in five (does it carry over to birds it never saw?),
and the photographers who posted both kinds (it cannot score there by recognising who took the photo).
"""
from __future__ import annotations

import argparse
import csv
import json
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

import distill
from distill.slim import square, tower

ROOT = Path(distill.__file__).resolve().parents[1]
LAYER = 15


def features(images: list[Path]) -> np.ndarray:
    tok, out = tower([LAYER]), []
    for i in range(0, len(images), 32):
        out.append(tok([square(Image.open(f)) for f in images[i:i + 32]])[LAYER].astype(np.float32).mean(1))
    return np.concatenate(out)


def fit(x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
    mean, std = x.mean(0), x.std(0) + 1e-6
    torch.manual_seed(0)
    w = torch.nn.Linear(x.shape[1], 1)
    opt = torch.optim.LBFGS(w.parameters(), max_iter=300)

    def step():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(w((x - mean) / std)[:, 0], y) + 1e-2 * w.weight.pow(2).sum()
        loss.backward()
        return loss
    opt.step(step)
    return {"mean": mean, "std": std, "weight": w.weight.detach()[0], "bias": w.bias.detach()}


def prob(head: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(((x - head["mean"]) / head["std"]) @ head["weight"] + head["bias"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = (ROOT / "data/flight/files.txt").read_text().splitlines()
    cache = ROOT / "data/flight" / f"tokens{LAYER}.npy"
    if not cache.exists():
        np.save(cache, features([ROOT / "data/flight/img" / f for f in files]))
    x = torch.from_numpy(np.load(cache))
    y = torch.tensor([f.startswith("fly/") for f in files], dtype=torch.float32)
    with open(ROOT / "data/flight-raw/manifest.csv", newline="") as f:
        rows = {Path(r["path"]).stem: r for r in csv.DictReader(f)}
    species = [rows[Path(f).stem]["scientific"] for f in files]
    who = [rows[Path(f).stem]["attribution"].split(",")[0] for f in files]
    both = {w for w, l in zip(who, y) if l} & {w for w, l in zip(who, y) if not l}
    report = {"layer": LAYER, "photos": len(files), "flying": int(y.sum())}
    for name, held in (("unseen_species", torch.tensor([zlib.crc32(s.encode()) % 5 == 0 for s in species])),
                       ("photographers_with_both_kinds", torch.tensor([w in both for w in who]))):
        p = prob(fit(x[~held], y[~held]), x[held])
        hit = (p > 0.5) == (y[held] > 0.5)
        report[name] = {"n": int(held.sum()), "correct": round(float(hit.float().mean()), 4),
                        "flying_found": round(float(hit[y[held] > 0.5].float().mean()), 4), "not_flying_kept": round(float(hit[y[held] < 0.5].float().mean()), 4)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(fit(x, y), out / "flight.safetensors")       # the shipped head is fit on everything; the scores above are from held-out fits
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
