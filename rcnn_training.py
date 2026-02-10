import argparse
import csv
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image


# ----------------------------
# Utils
# ----------------------------
def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    return tuple(zip(*batch))


def resolve_image_path(images_dir: Path, file_name: str) -> Path:
    """
    Assumes all images exist in a flat folder images_dir, but COCO file_name might include subfolders.
    """
    p1 = images_dir / file_name
    if p1.exists():
        return p1
    p2 = images_dir / Path(file_name).name
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Cannot find image '{file_name}' under '{images_dir}'")


def safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def extract_slide_id(file_name: str, slide_regex: Optional[str] = None) -> str:
    """
    Default heuristic: slide id == first token of basename (stem) before first underscore.
    Optionally use a regex with a capture group, e.g.:
      --slide_regex "(.*?)/"  (if slide is folder name)
      --slide_regex r"(H\d+)" (if slide contains H123 etc)
    """
    base = Path(file_name).name
    stem = Path(base).stem

    if slide_regex:
        m = re.search(slide_regex, file_name)
        if not m:
            m = re.search(slide_regex, base)
        if not m:
            m = re.search(slide_regex, stem)
        if not m:
            raise ValueError(f"slide_regex did not match file_name='{file_name}'")
        if m.groups():
            return str(m.group(1))
        return str(m.group(0))

    # default: first underscore token
    return stem.split("_")[0]


# ----------------------------
# Dataset
# ----------------------------
class MidogppBinaryMitosisDataset(Dataset):
    def __init__(self, images_dir: str, ann_json: str, mitosis_cat_name: str, skip_missing: bool = True):
        self.images_dir = Path(images_dir)
        self.coco = COCO(ann_json)

        cats = self.coco.loadCats(self.coco.getCatIds())
        name2id = {c["name"]: c["id"] for c in cats}
        if mitosis_cat_name not in name2id:
            raise ValueError(f"mitosis_cat_name='{mitosis_cat_name}' not found. Available: {list(name2id.keys())}")
        self.mitosis_cat_id = name2id[mitosis_cat_name]

        all_ids = sorted(self.coco.getImgIds())
        if not skip_missing:
            self.img_ids = all_ids
            return

        kept, missing = [], []
        for img_id in all_ids:
            info = self.coco.loadImgs([img_id])[0]
            try:
                _ = resolve_image_path(self.images_dir, info["file_name"])
                kept.append(img_id)
            except FileNotFoundError:
                missing.append(info["file_name"])

        self.img_ids = kept
        print(f"[dataset] total={len(all_ids)} kept={len(kept)} missing={len(missing)}")
        if missing:
            print("[dataset] example missing:", missing[:10])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        info = self.coco.loadImgs([img_id])[0]
        img_path = resolve_image_path(self.images_dir, info["file_name"])

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, areas, iscrowd = [], [], [], []
        for a in anns:
            if a.get("category_id") != self.mitosis_cat_id:
                continue

            # COCO bbox is [x, y, width, height]
            x, y, bw, bh = a["bbox"]
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(w), x + bw), min(float(h), y + bh)
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(1)  # mitosis class
            areas.append((x2 - x1) * (y2 - y1))
            iscrowd.append(int(a.get("iscrowd", 0)))

        image = TF.to_tensor(img)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64),
        }
        return image, target

    def img_info(self, idx: int) -> Dict[str, Any]:
        img_id = self.img_ids[idx]
        return self.coco.loadImgs([img_id])[0]


# ----------------------------
# Model
def build_faster_rcnn_binary(weights: str = "DEFAULT"):
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    return model


#training/eval
def train_one_epoch(model, loader, optimizer, device, log_every=50) -> Dict[str, float]:
    model.train()
    loss_sums = defaultdict(float)
    n = 0

    for step, (imgs, targets) in enumerate(loader, start=1):
        imgs = [im.to(device) for im in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        n += 1
        loss_sums["loss_total"] += float(loss.item())
        for k, v in loss_dict.items():
            loss_sums[str(k)] += float(v.item())

        if log_every and step % log_every == 0:
            print(
                f"  step {step}/{len(loader)} "
                f"loss={loss.item():.4f} "
                f"cls={loss_dict.get('loss_classifier', torch.tensor(0.)).item():.4f} "
                f"box={loss_dict.get('loss_box_reg', torch.tensor(0.)).item():.4f}"
            )

    return {k: v / max(1, n) for k, v in loss_sums.items()}


@torch.no_grad()
def coco_eval_on_subset(
    model,
    ds: MidogppBinaryMitosisDataset,
    subset_indices: List[int],
    device,
    score_thresh: float = 0.0,
    max_dets_per_image: int = 300,
) -> Dict[str, float]:
    model.eval()
    results = []

    for idx in subset_indices:
        img, _ = ds[idx]
        img_id = int(ds.img_ids[idx])

        out = model([img.to(device)])[0]
        boxes = out["boxes"].detach().cpu()
        scores = out["scores"].detach().cpu()

        keep = scores >= float(score_thresh)
        boxes = boxes[keep]
        scores = scores[keep]

        if len(scores) > max_dets_per_image:
            topk = torch.topk(scores, k=max_dets_per_image).indices
            boxes = boxes[topk]
            scores = scores[topk]

        for b, s in zip(boxes.tolist(), scores.tolist()):
            x1, y1, x2, y2 = b
            results.append(
                {
                    "image_id": img_id,
                    "category_id": int(ds.mitosis_cat_id),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(s),
                }
            )

    if len(results) == 0:
        return {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR_100": 0.0}

    dt = ds.coco.loadRes(results)
    coco_eval = COCOeval(ds.coco, dt, iouType="bbox")
    coco_eval.params.imgIds = [int(ds.img_ids[i]) for i in subset_indices]
    coco_eval.params.catIds = [int(ds.mitosis_cat_id)]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    s = coco_eval.stats
    return {
        "mAP": float(s[0]),
        "AP50": float(s[1]),
        "AP75": float(s[2]),
        "AR_100": float(s[8]),
    }

def compute_dataset_stats(ds: MidogppBinaryMitosisDataset, indices: List[int]) -> Dict[str, Any]:
    n_images = len(indices)
    ann_counts = []
    areas = []
    empty_images = 0

    for i in indices:
        img_id = int(ds.img_ids[i])
        ann_ids = ds.coco.getAnnIds(imgIds=[img_id], catIds=[ds.mitosis_cat_id])
        anns = ds.coco.loadAnns(ann_ids)

        c = 0
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            c += 1
            areas.append(float(w * h))

        ann_counts.append(c)
        if c == 0:
            empty_images += 1

    return {
        "n_images": int(n_images),
        "empty_images": int(empty_images),
        "empty_frac": float(empty_images / max(1, n_images)),
        "anns_total": int(sum(ann_counts)),
        "anns_per_image_mean": float(safe_mean([float(x) for x in ann_counts])),
        "anns_per_image_median": float(sorted(ann_counts)[n_images // 2] if n_images else 0),
        "box_area_mean": float(safe_mean(areas)) if areas else 0.0,
        "box_area_median": float(sorted(areas)[len(areas) // 2]) if areas else 0.0,
    }


def save_history_and_plots(out_dir: str, history: List[Dict[str, Any]], ds_stats: Dict[str, Any]):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # JSONL + CSV
    jsonl_path = os.path.join(out_dir, "metrics.jsonl")
    with open(jsonl_path, "w") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")

    csv_path = os.path.join(out_dir, "metrics.csv")
    keys = sorted({k for row in history for k in row.keys()})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in history:
            w.writerow(row)

    # Loss curves
    epochs = [r["epoch"] for r in history]
    plt.figure()
    plt.plot(epochs, [r.get("train_loss_total", None) for r in history], label="train_loss_total")
    for k in ["train_loss_classifier", "train_loss_box_reg", "train_loss_objectness", "train_loss_rpn_box_reg"]:
        if any(r.get(k) is not None for r in history):
            plt.plot(epochs, [r.get(k, None) for r in history], label=k)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=200)
    plt.close()

    # mAP curves
    plt.figure()
    if any(r.get("test_mAP") is not None for r in history):
        plt.plot(epochs, [r.get("test_mAP", None) for r in history], label="test_mAP")
    if any(r.get("test_AP50") is not None for r in history):
        plt.plot(epochs, [r.get("test_AP50", None) for r in history], label="test_AP50")
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "map_curves.png"), dpi=200)
    plt.close()

    # Dataset stats plot
    plt.figure()
    names = ["train", "test"]
    empty = [ds_stats["train"]["empty_frac"], ds_stats["test"]["empty_frac"]]
    anns_mean = [ds_stats["train"]["anns_per_image_mean"], ds_stats["test"]["anns_per_image_mean"]]
    plt.plot([0, 1], empty, marker="o", label="empty_frac")
    plt.plot([0, 1], anns_mean, marker="o", label="anns_per_image_mean")
    plt.xticks([0, 1], names)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dataset_stats.png"), dpi=200)
    plt.close()

    with open(os.path.join(out_dir, "dataset_stats.json"), "w") as f:
        json.dump(ds_stats, f, indent=2)

    print("[ok] wrote:", csv_path)
    print("[ok] wrote:", jsonl_path)
    print("[ok] wrote plots: loss_curves.png, map_curves.png, dataset_stats.png")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="Path to COCO JSON")
    ap.add_argument("--images_dir", required=True, help="Flat images folder (all images stored here)")
    ap.add_argument("--split_csv", required=True, help="Path to split.csv created by your script (sep=';')")
    ap.add_argument("--mitosis_cat_name", required=True, help="Exact COCO category name for mitosis")

    ap.add_argument("--slide_regex", default=None, help="Optional regex (with group) to extract Slide from file_name")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--weight_decay", type=float, default=0.0005)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="rcnn_out")
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--skip_missing", action="store_true")

    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--eval_score_thresh", type=float, default=0.0)
    ap.add_argument("--max_dets_per_image", type=int, default=300)

    ap.add_argument("--weights", default="DEFAULT", help="Torchvision weights (DEFAULT or None)")
    args = ap.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Load dataset
    ds = MidogppBinaryMitosisDataset(args.images_dir, args.ann, args.mitosis_cat_name, skip_missing=args.skip_missing)
    print("dataset size:", len(ds), "| mitosis_cat_id:", ds.mitosis_cat_id)

# --- COCOeval requires iscrowd on GT annotations; some MIDOG++ exports omit it ---
    anns = ds.coco.dataset.get("annotations", [])
    need_iscrowd = any("iscrowd" not in a for a in anns)
    if need_iscrowd:
        for a in anns:
            a.setdefault("iscrowd", 0)
        ds.coco.createIndex()
        print("[fix] added missing 'iscrowd'=0 to GT annotations for COCOeval")


        # --- COCOeval requires 'area' and 'iscrowd' in GT annotations ---
    anns = ds.coco.dataset.get("annotations", [])

    need_fix = False
    for a in anns:
        if "iscrowd" not in a:
            a["iscrowd"] = 0
            need_fix = True
        if "area" not in a:
            # COCO bbox is [x, y, w, h]
            x, y, w, h = a.get("bbox", [0, 0, 0, 0])
            a["area"] = float(max(0.0, w) * max(0.0, h))
            need_fix = True

    if need_fix:
        ds.coco.createIndex()
        print("[fix] added missing 'iscrowd' and/or 'area' for COCOeval")


    # Load split.csv 
    split_df = pd.read_csv(args.split_csv, sep=";").apply(lambda x: x.astype(str).str.strip())
    if "Slide" not in split_df.columns or "Split" not in split_df.columns:
        raise ValueError("split.csv must contain columns: Slide, Split")

    train_slides = set(split_df.loc[split_df["Split"] == "train", "Slide"].astype(str).tolist())
    test_slides = set(split_df.loc[split_df["Split"] == "test", "Slide"].astype(str).tolist())
    if not train_slides or not test_slides:
        raise ValueError(f"split.csv produced empty train/test sets. train={len(train_slides)} test={len(test_slides)}")

    # Map dataset indices since slide id derived from COCO file_name
    train_idxs, test_idxs = [], []
    missing_slide = 0

    for i in range(len(ds)):
        info = ds.img_info(i)
        fn = info["file_name"]
        slide = extract_slide_id(fn, args.slide_regex)

        if slide in train_slides:
            train_idxs.append(i)
        elif slide in test_slides:
            test_idxs.append(i)
        else:
            missing_slide += 1

    print(f"[split] matched train images: {len(train_idxs)} | test images: {len(test_idxs)} | unmatched images: {missing_slide}")
    if len(train_idxs) == 0 or len(test_idxs) == 0:
        raise ValueError(
            "No images matched train or test. "
            "Your Slide extraction likely doesn't match your filename convention. "
            "Try setting --slide_regex to extract the correct slide id."
        )

    # Dataset stats
    ds_stats = {
        "train": compute_dataset_stats(ds, train_idxs),
        "test": compute_dataset_stats(ds, test_idxs),
    }
    print("[stats] train:", ds_stats["train"])
    print("[stats] test :", ds_stats["test"])

    # Loader
    train_loader = DataLoader(
        Subset(ds, train_idxs),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    weights = None if str(args.weights).lower() == "none" else args.weights
    model = build_faster_rcnn_binary(weights=weights).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    history = []
    best_map = -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_losses = train_one_epoch(model, train_loader, optimizer, device, log_every=50)
        elapsed = float(time.time() - t0)
        lr = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": epoch,
            "lr": lr,
            "time_sec": elapsed,
            "train_loss_total": train_losses.get("loss_total"),
            "train_loss_classifier": train_losses.get("loss_classifier"),
            "train_loss_box_reg": train_losses.get("loss_box_reg"),
            "train_loss_objectness": train_losses.get("loss_objectness"),
            "train_loss_rpn_box_reg": train_losses.get("loss_rpn_box_reg"),
        }

        print(
            f"epoch {epoch}/{args.epochs} "
            f"loss={row['train_loss_total']:.4f} "
            f"time={elapsed:.1f}s"
        )

        # Eval
        if args.eval_every and epoch % args.eval_every == 0:
            print("[eval] COCOeval on test split...")
            m = coco_eval_on_subset(
                model=model,
                ds=ds,
                subset_indices=test_idxs,
                device=device,
                score_thresh=args.eval_score_thresh,
                max_dets_per_image=args.max_dets_per_image,
            )
            row.update({f"test_{k}": v for k, v in m.items()})
            print(f"[eval] mAP={m['mAP']:.4f} AP50={m['AP50']:.4f} AR100={m['AR_100']:.4f}")

            if m["mAP"] > best_map:
                best_map = m["mAP"]
                best_path = os.path.join(args.out_dir, "fasterrcnn_best.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_map": best_map,
                        "mitosis_cat_name": args.mitosis_cat_name,
                        "ann": args.ann,
                        "images_dir": args.images_dir,
                        "split_csv": args.split_csv,
                        "slide_regex": args.slide_regex,
                    },
                    best_path,
                )
                print("[ok] saved best:", best_path)

        history.append(row)

        if args.save_every and epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.out_dir, f"fasterrcnn_epoch{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "mitosis_cat_name": args.mitosis_cat_name,
                    "ann": args.ann,
                    "images_dir": args.images_dir,
                    "split_csv": args.split_csv,
                    "slide_regex": args.slide_regex,
                },
                ckpt_path,
            )
            print("[ok] saved:", ckpt_path)

    save_history_and_plots(args.out_dir, history, ds_stats)


if __name__ == "__main__":
    main()