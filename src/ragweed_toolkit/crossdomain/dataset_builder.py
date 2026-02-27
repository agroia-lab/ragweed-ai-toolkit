"""
Dataset Builder for Cross-Domain Weed Detection
=================================================

Unify annotations from multiple formats (YOLO TXT, VOC XML, COCO JSON)
into a single-class YOLO dataset. Supports configurable train/val/test
splits, source-stratified splitting, and provenance tracking via manifest.

Functions
---------
process_yolo_labels
    Read and remap YOLO TXT labels to a target class.
process_voc_xml
    Parse Pascal VOC XML annotations into YOLO format.
build_international_dataset
    Build a flat YOLO dataset from multiple databases defined in YAML.
build_combined_dataset
    Merge Roboflow-split sources with a flat international pool.
validate_labels
    Spot-check random label files for format correctness.
"""

import csv
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Label Processing
# ---------------------------------------------------------------------------


def process_yolo_labels(
    label_path: Path,
    ragweed_class: int,
    filter_ragweed_only: bool,
) -> Optional[List[str]]:
    """Read a YOLO TXT label file, filter/remap to class 0.

    Args:
        label_path: Path to the YOLO ``.txt`` label file.
        ragweed_class: Source class ID for the target weed.
        filter_ragweed_only: If True, keep only lines matching *ragweed_class*
            (multi-class dataset). Otherwise all lines are expected to be the
            target class (single-class dataset).

    Returns:
        Remapped label lines (class 0) or ``None`` if no target found.
    """
    if not label_path.exists():
        return None

    lines_out: List[str] = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id != ragweed_class:
                continue
            lines_out.append(f"0 {' '.join(parts[1:])}")

    return lines_out if lines_out else None


def process_voc_xml(
    xml_path: Path,
    target_class_name: str,
) -> Optional[Tuple[List[str], int, int]]:
    """Parse a Pascal VOC XML file and extract target-class bboxes as YOLO.

    Args:
        xml_path: Path to the ``.xml`` annotation file.
        target_class_name: VOC class name to extract (e.g. ``"Ragweed"``).

    Returns:
        Tuple of ``(yolo_lines, img_width, img_height)`` or ``None``.
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return None

    root = tree.getroot()
    size_elem = root.find("size")
    if size_elem is None:
        return None

    img_w = int(size_elem.findtext("width", "0"))
    img_h = int(size_elem.findtext("height", "0"))
    if img_w == 0 or img_h == 0:
        return None

    lines_out: List[str] = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "")
        if name != target_class_name:
            continue
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = float(bndbox.findtext("xmin", "0"))
        ymin = float(bndbox.findtext("ymin", "0"))
        xmax = float(bndbox.findtext("xmax", "0"))
        ymax = float(bndbox.findtext("ymax", "0"))

        x_center = max(0.0, min(1.0, ((xmin + xmax) / 2) / img_w))
        y_center = max(0.0, min(1.0, ((ymin + ymax) / 2) / img_h))
        width = max(0.0, min(1.0, (xmax - xmin) / img_w))
        height = max(0.0, min(1.0, (ymax - ymin) / img_h))

        lines_out.append(
            f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    return (lines_out, img_w, img_h) if lines_out else None


# ---------------------------------------------------------------------------
# Per-database handlers
# ---------------------------------------------------------------------------


def _process_yolo_txt_db(
    db_name: str,
    db_cfg: Dict[str, Any],
    out_images: Path,
    out_labels: Path,
    dry_run: bool,
) -> List[Dict]:
    """Process a YOLO TXT database (separate images/ and labels/ dirs)."""
    images_dir = Path(db_cfg["images"])
    label_dir = Path(db_cfg["labels"])
    prefix = db_cfg["prefix"]
    ragweed_class = db_cfg["ragweed_class_id"]
    filter_ragweed = db_cfg.get("filter_ragweed_only", False)
    image_ext = db_cfg.get("image_ext", ".jpg")

    image_files = sorted(images_dir.glob(f"*{image_ext}"), key=lambda p: p.name)
    manifest_rows: List[Dict] = []

    for img_path in tqdm(image_files, desc=f"  {db_name}", leave=False):
        stem = img_path.stem
        label_path = label_dir / f"{stem}.txt"
        label_lines = process_yolo_labels(label_path, ragweed_class, filter_ragweed)
        if label_lines is None:
            continue

        out_name = f"{prefix}_{img_path.name}"
        out_label_name = f"{prefix}_{stem}.txt"

        if not dry_run:
            shutil.copy2(img_path, out_images / out_name)
            (out_labels / out_label_name).write_text("\n".join(label_lines) + "\n")

        manifest_rows.append({
            "filename": out_name,
            "source_db": db_name,
            "original_path": str(img_path),
            "n_annotations": len(label_lines),
        })

    return manifest_rows


def _process_voc_xml_db(
    db_name: str,
    db_cfg: Dict[str, Any],
    out_images: Path,
    out_labels: Path,
    dry_run: bool,
) -> List[Dict]:
    """Process a Pascal VOC XML database (co-located with images)."""
    prefix = db_cfg["prefix"]
    target_class = db_cfg["ragweed_class_name"]
    data_dirs = db_cfg.get("data_dirs", [])
    image_ext = db_cfg.get("image_ext", ".jpg")

    image_files: List[Path] = []
    for dir_str in data_dirs:
        d = Path(dir_str)
        if d.exists():
            image_files.extend(d.glob(f"*{image_ext}"))
    image_files.sort(key=lambda p: p.name)

    manifest_rows: List[Dict] = []
    for img_path in tqdm(image_files, desc=f"  {db_name}", leave=False):
        xml_path = img_path.with_suffix(".xml")
        if not xml_path.exists():
            continue
        result = process_voc_xml(xml_path, target_class)
        if result is None:
            continue
        label_lines, _, _ = result

        out_name = f"{prefix}_{img_path.name}"
        out_label_name = f"{prefix}_{img_path.stem}.txt"

        if not dry_run:
            shutil.copy2(img_path, out_images / out_name)
            (out_labels / out_label_name).write_text("\n".join(label_lines) + "\n")

        manifest_rows.append({
            "filename": out_name,
            "source_db": db_name,
            "original_path": str(img_path),
            "n_annotations": len(label_lines),
        })

    return manifest_rows


def _process_yolo_txt_flat(
    db_name: str,
    db_cfg: Dict[str, Any],
    out_images: Path,
    out_labels: Path,
    dry_run: bool,
) -> List[Dict]:
    """Process YOLO TXT labels co-located with images in subdirs."""
    base_dir = Path(db_cfg["base_dir"])
    prefix = db_cfg["prefix"]
    ragweed_class = db_cfg["ragweed_class_id"]
    filter_ragweed = db_cfg.get("filter_ragweed_only", False)
    crop_subdirs = db_cfg.get("crop_subdirs", [])
    image_ext = db_cfg.get("image_ext", ".JPG")
    exclude_files = set(db_cfg.get("exclude_files", []))

    image_files: List[Path] = []
    for sub in crop_subdirs:
        sd = base_dir / sub
        if sd.exists():
            image_files.extend(sd.glob(f"*{image_ext}"))
    image_files.sort(key=lambda p: p.name)

    manifest_rows: List[Dict] = []
    for img_path in tqdm(image_files, desc=f"  {db_name}", leave=False):
        stem = img_path.stem
        if f"{stem}.txt" in exclude_files:
            continue
        label_path = img_path.with_suffix(".txt")
        label_lines = process_yolo_labels(label_path, ragweed_class, filter_ragweed)
        if label_lines is None:
            continue

        out_ext = img_path.suffix.lower()
        out_name = f"{prefix}_{stem}{out_ext}"
        out_label_name = f"{prefix}_{stem}.txt"

        if not dry_run:
            shutil.copy2(img_path, out_images / out_name)
            (out_labels / out_label_name).write_text("\n".join(label_lines) + "\n")

        manifest_rows.append({
            "filename": out_name,
            "source_db": db_name,
            "original_path": str(img_path),
            "n_annotations": len(label_lines),
        })

    return manifest_rows


def _dispatch_database(
    db_name: str,
    db_cfg: Dict[str, Any],
    out_images: Path,
    out_labels: Path,
    dry_run: bool,
) -> List[Dict]:
    """Route a database config to the appropriate handler."""
    fmt = db_cfg.get("format", "yolo_txt")

    if "base_dir" in db_cfg and "crop_subdirs" in db_cfg:
        return _process_yolo_txt_flat(db_name, db_cfg, out_images, out_labels, dry_run)
    if fmt == "voc_xml":
        return _process_voc_xml_db(db_name, db_cfg, out_images, out_labels, dry_run)
    return _process_yolo_txt_db(db_name, db_cfg, out_images, out_labels, dry_run)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_international_dataset(
    config_path: str,
    output_dir: str,
    *,
    dry_run: bool = False,
    validate: bool = False,
) -> Dict[str, Any]:
    """Build a unified single-class YOLO dataset from multiple databases.

    Reads a YAML config defining international databases (each with a format,
    paths, prefix, and ragweed class ID) and copies images + remapped labels
    into a flat ``images/`` + ``labels/`` output directory.

    Args:
        config_path: YAML file with ``databases`` and ``output_dir`` keys.
        output_dir: Destination directory (overrides YAML ``output_dir``).
        dry_run: Count only, do not copy files.
        validate: Spot-check random labels after build.

    Returns:
        Build summary dict with per-database statistics.
    """
    cfg = yaml.safe_load(Path(config_path).read_text())
    out = Path(output_dir)
    out_images = out / "images"
    out_labels = out / "labels"

    if not dry_run:
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    db_summaries: Dict[str, Dict] = {}
    databases = cfg.get("databases", {})

    for db_name, db_cfg in databases.items():
        check_path = db_cfg.get("images") or db_cfg.get("base_dir") or (
            db_cfg.get("data_dirs", [None])[0]
        )
        if check_path and not Path(check_path).exists():
            db_summaries[db_name] = {"status": "skipped", "n_images": 0, "n_annotations": 0}
            continue

        rows = _dispatch_database(db_name, db_cfg, out_images, out_labels, dry_run)
        n_images = len(rows)
        n_annots = sum(r["n_annotations"] for r in rows)
        all_rows.extend(rows)

        db_summaries[db_name] = {
            "status": "ok",
            "n_images": n_images,
            "n_annotations": n_annots,
            "annots_per_image": round(n_annots / n_images, 1) if n_images else 0,
        }

    if not dry_run:
        manifest_path = out / "manifest.csv"
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["filename", "source_db", "original_path", "n_annotations"]
            )
            writer.writeheader()
            writer.writerows(all_rows)

        summary_path = out / "build_summary.json"
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": str(config_path),
            "output_dir": str(out),
            "dry_run": dry_run,
            "total_images": sum(s["n_images"] for s in db_summaries.values()),
            "total_annotations": sum(s["n_annotations"] for s in db_summaries.values()),
            "databases": db_summaries,
        }
        Path(summary_path).write_text(json.dumps(summary, indent=2))

    if validate and not dry_run:
        validate_labels(out_labels)

    return db_summaries


def build_combined_dataset(
    roboflow_sources: List[Dict[str, str]],
    international_dir: str,
    output_dir: str,
    *,
    intl_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Merge Roboflow-split sources with an international pool.

    Each Roboflow source is expected to have ``train/``, ``valid/``, ``test/``
    sub-directories with ``images/`` and ``labels/`` inside.  The international
    directory must contain ``images/``, ``labels/``, and ``manifest.csv``.

    Args:
        roboflow_sources: List of dicts with ``"path"`` and ``"prefix"`` keys.
        international_dir: Root of the flat international dataset.
        output_dir: Destination directory with ``train/valid/test`` structure.
        intl_split: Train/valid/test ratios for the international pool.
        seed: Random seed for reproducible splitting.
        dry_run: Count only, do not copy files.

    Returns:
        Summary dict with per-source and per-split counts.
    """
    out = Path(output_dir)
    intl = Path(international_dir)

    if not dry_run:
        for split in ("train", "valid", "test"):
            (out / split / "images").mkdir(parents=True, exist_ok=True)
            (out / split / "labels").mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    source_summaries: Dict[str, Dict] = {}

    # --- Roboflow sources (preserve splits) ---
    for src in roboflow_sources:
        prefix = src["prefix"]
        src_dir = Path(src["path"])
        counts = {"train": 0, "valid": 0, "test": 0}

        for split in ("train", "valid", "test"):
            src_images = src_dir / split / "images"
            src_labels = src_dir / split / "labels"
            if not src_images.exists():
                continue

            dst_images = out / split / "images"
            dst_labels = out / split / "labels"

            for img_path in sorted(src_images.glob("*.jpg")):
                stem = img_path.stem
                out_img = f"{prefix}_{img_path.name}"
                out_lbl = f"{prefix}_{stem}.txt"
                label_path = src_labels / f"{stem}.txt"
                n_annots = 0
                label_text = ""

                if label_path.exists():
                    label_text = label_path.read_text()
                    n_annots = sum(1 for ln in label_text.strip().splitlines() if ln.strip())

                if not dry_run:
                    shutil.copy2(img_path, dst_images / out_img)
                    (dst_labels / out_lbl).write_text(label_text)

                all_rows.append({
                    "filename": out_img,
                    "source": prefix,
                    "source_db": prefix,
                    "split": split,
                    "n_annotations": n_annots,
                })
                counts[split] += 1

        source_summaries[prefix] = counts

    # --- International pool (stratified split) ---
    manifest_path = intl / "manifest.csv"
    if manifest_path.exists():
        groups: Dict[str, List[Dict]] = defaultdict(list)
        with open(manifest_path, "r") as f:
            for row in csv.DictReader(f):
                groups[row["source_db"]].append(row)

        rng = random.Random(seed)
        intl_counts = {"train": 0, "valid": 0, "test": 0}

        for db_name in sorted(groups):
            rows = groups[db_name]
            rng.shuffle(rows)
            n = len(rows)
            n_train = round(n * intl_split[0])
            n_valid = round(n * intl_split[1])
            boundaries = [
                (0, n_train, "train"),
                (n_train, n_train + n_valid, "valid"),
                (n_train + n_valid, n, "test"),
            ]
            for start, end, split in boundaries:
                for row in rows[start:end]:
                    filename = row["filename"]
                    stem = Path(filename).stem
                    src_img = intl / "images" / filename
                    src_lbl = intl / "labels" / f"{stem}.txt"
                    if not src_img.exists():
                        continue

                    if not dry_run:
                        shutil.copy2(src_img, out / split / "images" / filename)
                        if src_lbl.exists():
                            shutil.copy2(src_lbl, out / split / "labels" / f"{stem}.txt")

                    all_rows.append({
                        "filename": filename,
                        "source": "International",
                        "source_db": row["source_db"],
                        "split": split,
                        "n_annotations": int(row.get("n_annotations", 0)),
                    })
                    intl_counts[split] += 1

        source_summaries["International"] = intl_counts

    if not dry_run:
        (out / "manifest.csv").write_text("")
        fieldnames = ["filename", "source", "source_db", "split", "n_annotations"]
        with open(out / "manifest.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "output_dir": str(out),
            "seed": seed,
            "intl_split_ratios": list(intl_split),
            "sources": source_summaries,
        }
        (out / "build_summary.json").write_text(json.dumps(summary, indent=2))

    return source_summaries


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_labels(labels_dir: Path, n: int = 5) -> List[str]:
    """Spot-check random label files for format correctness.

    Verifies each sampled file has 5-field lines, class 0, and
    coordinates in [0, 1].

    Args:
        labels_dir: Directory containing ``.txt`` label files.
        n: Number of files to check.

    Returns:
        List of issue strings (empty if all OK).
    """
    label_files = list(Path(labels_dir).glob("*.txt"))
    if not label_files:
        return ["No label files found"]

    sample = random.sample(label_files, min(n, len(label_files)))
    issues: List[str] = []

    for lf in sample:
        for i, line in enumerate(lf.read_text().strip().splitlines(), 1):
            parts = line.strip().split()
            if len(parts) != 5:
                issues.append(f"{lf.name} line {i}: expected 5 fields, got {len(parts)}")
                continue
            if int(parts[0]) != 0:
                issues.append(f"{lf.name} line {i}: class={parts[0]} (expected 0)")
            for j, v in enumerate(map(float, parts[1:])):
                if v < 0 or v > 1:
                    issues.append(f"{lf.name} line {i}: coord[{j}]={v:.4f} out of [0,1]")

    return issues
