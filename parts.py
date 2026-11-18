"""Beak and eye points from the teacher's patch tokens: a small head on the frozen (slimmed) image tower.

Run from a bioclip-bird checkout, with this repository cloned next to it as ../bioclip-bird-heads:
  ./run ../bioclip-bird-heads/parts.py sheet OUT.jpg            # ground truth drawn on 32 crops: check the geometry by eye
  ./run ../bioclip-bird-heads/parts.py probe                    # which block's tokens localise best (part of the train split)
  ./run ../bioclip-bird-heads/parts.py train --layer 15 --out runs/parts-1   # fit on CUB train, score once on CUB test

Data: CUB-200-2011 (bioclip-bird's data/cub), its own train/test split. A bird is framed exactly as gardener frames it for the
species encoder (rules.crop_with_context on the box, then the teacher's resize + centre crop to 224), so the head
reads tokens the identify pass has already computed. Points are in that 224 px frame; one that the framing cuts
off counts as not visible. Left/right are the bird's own (CUB's convention).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn

import distill
from distill.slim import DEV, GRID, SIZE, tower

CUB = Path(distill.__file__).resolve().parents[1] / "data" / "cub" / "CUB_200_2011"
PARTS = {2: "beak", 7: "left_eye", 11: "right_eye"}      # CUB part ids
PATCH = 14


def table(name: str) -> list[list[str]]:
    return [line.split() for line in (CUB / name).read_text().splitlines()]


def samples() -> list[dict]:
    """Every CUB image: path, box as corners, train flag, {part: (x, y) or None when not visible}."""
    out = {int(i): {"path": CUB / "images" / p, "pts": {}} for i, p in table("images.txt")}
    for i, x, y, w, h in table("bounding_boxes.txt"):
        out[int(i)]["box"] = (float(x), float(y), float(x) + float(w), float(y) + float(h))
    for i, t in table("train_test_split.txt"):
        out[int(i)]["train"] = t == "1"
    for i, p, x, y, v in table("parts/part_locs.txt"):
        if int(p) in PARTS:
            out[int(i)]["pts"][PARTS[int(p)]] = (float(x), float(y)) if v == "1" else None
    return [out[i] for i in sorted(out)]


def frame(s: dict, jitter: random.Random | None = None) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """(224 px image as the teacher sees it, (3, 2) points in it, (3,) visible). jitter: a detector's box, not CUB's
    (each side moved by up to 15 % of the box)."""
    from gardener.service.rules import MIN_CROP, crop_with_context
    im = Image.open(s["path"]).convert("RGB")
    box = s["box"]
    if jitter:
        w, h = box[2] - box[0], box[3] - box[1]
        box = tuple(float(np.clip(v + jitter.uniform(-0.15, 0.15) * d, 0, lim)) for v, d, lim in zip(box, (w, h, w, h), (*im.size, *im.size)))
    crop = crop_with_context(im, box)
    x0, y0, x1, y1 = box                                 # the same arithmetic, to know where the crop sits
    bw, bh = x1 - x0, y1 - y0
    x0, y0, x1, y1 = x0 - bw * 0.1, y0 - bh * 0.1, x1 + bw * 0.1, y1 + bh * 0.1
    side = max(x1 - x0, y1 - y0, float(MIN_CROP))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, x1, y0, y1 = cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2
    if x0 < 0:
        x1, x0 = x1 - x0, 0.0
    if y0 < 0:
        y1, y0 = y1 - y0, 0.0
    if x1 > im.width:
        x0, x1 = x0 - (x1 - im.width), float(im.width)
    if y1 > im.height:
        y0, y1 = y0 - (y1 - im.height), float(im.height)
    ox, oy = max(0, int(x0)), max(0, int(y0))
    assert crop.size == (min(im.width, int(x1)) - ox, min(im.height, int(y1)) - oy), "crop_with_context changed"
    k = SIZE / min(crop.size)                            # the teacher's preprocess: shortest side to 224, centre crop
    w, h = round(crop.width * k), round(crop.height * k)
    left, top = (w - SIZE) // 2, (h - SIZE) // 2
    small = crop.resize((w, h), Image.BICUBIC).crop((left, top, left + SIZE, top + SIZE))
    pts, vis = np.zeros((3, 2), np.float32), np.zeros(3, bool)
    for n, name in enumerate(PARTS.values()):
        if s["pts"][name]:
            p = ((s["pts"][name][0] - ox) * w / crop.width - left, (s["pts"][name][1] - oy) * h / crop.height - top)
            if 0 <= p[0] < SIZE and 0 <= p[1] < SIZE:
                pts[n], vis[n] = p, True
    return small, pts, vis


def draw(im: Image.Image, pts: np.ndarray, vis: np.ndarray, colours=("red", "lime", "cyan")) -> Image.Image:
    d = ImageDraw.Draw(im)
    for p, v, c in zip(pts, vis, colours):
        if v:
            d.ellipse((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), outline=c, width=2)
    return im


def extract(rows: list[dict], layers: list[int], jitter: int | None = None) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    tokens, rng = tower(layers), random.Random(jitter) if jitter is not None else None
    feats, pts, vis = {n: np.zeros((len(rows), GRID * GRID, 1280), np.float16) for n in layers}, [], []
    for i in range(0, len(rows), 32):
        batch = [frame(s, rng) for s in rows[i:i + 32]]
        for n, v in tokens([b[0] for b in batch]).items():
            feats[n][i:i + 32] = v
        pts += [b[1] for b in batch]
        vis += [b[2] for b in batch]
        if i % 960 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    return feats, np.stack(pts), np.stack(vis)


class Head(nn.Module):
    """Per patch token: is the part here, and where inside the patch; per image: is the part visible at all."""

    def __init__(self, width: int = 1280, hidden: int = 256) -> None:
        super().__init__()
        self.norm, self.fc = nn.LayerNorm(width), nn.Linear(width, hidden)
        self.out, self.vis = nn.Linear(hidden, 3 * 3), nn.Linear(hidden, 3)

    def forward(self, t: torch.Tensor):
        h = nn.functional.gelu(self.fc(self.norm(t.float())))
        o = self.out(h).view(len(t), GRID * GRID, 3, 3)
        return o[..., 0], torch.sigmoid(o[..., 1:]), self.vis(h.mean(1))     # (n, 256, 3) heat, (n, 256, 3, 2) offset, (n, 3)

    def predict(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        heat, off, vis = self(t)
        cell = heat.argmax(1)                                                 # (n, 3)
        o = off.gather(1, cell.view(-1, 1, 3, 1).expand(-1, 1, 3, 2))[:, 0]
        return (torch.stack([cell % GRID, cell // GRID], -1) + o) * PATCH, torch.sigmoid(vis)


def fit(feats: np.ndarray, pts: np.ndarray, vis: np.ndarray, epochs: int = 30, seed: int = 0) -> Head:
    torch.manual_seed(seed)
    head = Head().to(DEV)
    x, p, v = torch.from_numpy(feats).to(DEV), torch.from_numpy(pts).to(DEV), torch.from_numpy(vis).to(DEV)
    cell = (p[..., 1] // PATCH).long() * GRID + (p[..., 0] // PATCH).long()
    frac = p / PATCH - torch.floor(p / PATCH)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * (len(x) // 64))
    for _ in range(epochs):
        for idx in torch.randperm(len(x), device=DEV).split(64):
            if len(idx) < 64:
                continue
            heat, off, vl = head(x[idx])
            m = v[idx]
            o = off.gather(1, cell[idx].view(-1, 1, 3, 1).expand(-1, 1, 3, 2))[:, 0]
            loss = (nn.functional.cross_entropy(heat.transpose(1, 2)[m], cell[idx][m]) + nn.functional.l1_loss(o[m], frac[idx][m])
                    + nn.functional.binary_cross_entropy_with_logits(vl, m.float()))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return head.eval()


def score(pred: np.ndarray, conf: np.ndarray, pts: np.ndarray, vis: np.ndarray) -> dict:
    """Errors as a share of the 224 px frame's side. eye = what gardener needs: the more confident of the two
    predicted eyes against the nearest visible true eye (CUB's left/right labels are not reliable)."""
    out = {}
    err = np.linalg.norm(pred - pts, axis=-1) / SIZE
    for n, name in enumerate(PARTS.values()):
        e = err[vis[:, n], n]
        out[name] = {"n": int(len(e)), "median": round(float(np.median(e)), 4), "within_5pct": round(float((e < 0.05).mean()), 4), "within_10pct": round(float((e < 0.1).mean()), 4)}
    best = np.where(conf[:, 1] >= conf[:, 2], 1, 2)
    p = pred[np.arange(len(pred)), best]
    d = np.where(vis[:, 1:], np.linalg.norm(p[:, None] - pts[:, 1:], axis=-1) / SIZE, np.inf).min(1)
    e = d[np.isfinite(d)]
    out["eye"] = {"n": int(len(e)), "median": round(float(np.median(e)), 4), "within_5pct": round(float((e < 0.05).mean()), 4), "within_10pct": round(float((e < 0.1).mean()), 4)}
    out["visible_correct"] = {name: round(float(((conf[:, n] > 0.5) == vis[:, n]).mean()), 4) for n, name in enumerate(PARTS.values())}
    out["any_eye_visible_correct"] = round(float(((conf[:, 1:].max(1) > 0.5) == vis[:, 1:].any(1)).mean()), 4)
    return out


def run(head: Head, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        out = [head.predict(torch.from_numpy(feats[i:i + 256]).to(DEV)) for i in range(0, len(feats), 256)]
    return torch.cat([o[0] for o in out]).cpu().numpy(), torch.cat([o[1] for o in out]).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sheet", "probe", "train"])
    ap.add_argument("target", nargs="?")
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--out")
    ap.add_argument("--jitter", action="store_true", help="train: add a copy of the train split under shifted boxes")
    a = ap.parse_args()
    rows = samples()
    rng = random.Random(0)
    train, test = [s for s in rows if s["train"]], [s for s in rows if not s["train"]]
    rng.shuffle(train)
    if a.cmd == "sheet":
        sheet = Image.new("RGB", (8 * SIZE, 4 * SIZE))
        for i, s in enumerate(train[:32]):
            sheet.paste(draw(*frame(s, random.Random(i) if i >= 16 else None)), (i % 8 * SIZE, i // 8 * SIZE))
        sheet.save(a.target)
        return
    if a.cmd == "probe":
        layers = [11, 15, 19, 23, 27, 31]
        feats, pts, vis = extract(train[:2000], layers)
        for n in layers:
            head = fit(feats[n][:1500], pts[:1500], vis[:1500])
            s = score(*run(head, feats[n][1500:]), pts[1500:], vis[1500:])
            print(n, json.dumps({k: s[k] for k in ("beak", "eye", "any_eye_visible_correct")}), flush=True)
        return
    from safetensors.torch import save_file
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    feats, pts, vis = extract(train, [a.layer])
    extra = extract(train, [a.layer], jitter=1) if a.jitter else None       # a second copy under a detector-like box
    if extra:
        feats, pts, vis = {a.layer: np.concatenate([feats[a.layer], extra[0][a.layer]])}, np.concatenate([pts, extra[1]]), np.concatenate([vis, extra[2]])
    head = fit(feats[a.layer], pts, vis)
    save_file(head.state_dict(), out / "parts.safetensors")
    mean = np.stack([pts[vis[:, n], n].mean(0) for n in range(3)])
    del feats
    report = {"layer": a.layer, "train_rows": len(pts), "note": "errors are a share of the 224 px frame's side"}
    for name, jit in (("test", None), ("test_jittered_boxes", 2)):
        f, p, v = extract(test, [a.layer], jit)
        pred, conf = run(head, f[a.layer])
        report[name] = score(pred, conf, p, v)
        report[name + "_baseline_mean_position"] = score(np.broadcast_to(mean, pred.shape), np.ones_like(conf), p, v)
        if jit is None:
            sheet = Image.new("RGB", (8 * SIZE, 4 * SIZE))
            pick = random.Random(1).sample(range(len(test)), 32)
            for i, k in enumerate(pick):
                im = draw(frame(test[k])[0], pred[k], conf[k] > 0.5, ("red", "lime", "cyan"))
                sheet.paste(im, (i % 8 * SIZE, i // 8 * SIZE))
            sheet.save(out / "test-predictions.jpg")
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
