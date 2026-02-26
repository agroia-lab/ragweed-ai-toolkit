"""
Example 01: Configure a YOLO Detector for Ragweed Training
===========================================================

Shows how to set up a TrainingConfig with the chapter's validated
hyperparameters (Table 3) and inspect the configuration.

This script does NOT launch actual training -- it only builds and
prints the config so you can verify parameters before committing GPU
time.  To train for real, call ``train(cfg)`` (requires a dataset YAML
and a GPU).
"""

from dataclasses import asdict

from ragweed_toolkit.detection.trainer import TrainingConfig

# ------------------------------------------------------------------ #
# 1.  Create a config with chapter defaults
# ------------------------------------------------------------------ #
cfg = TrainingConfig(
    model="yolo11l.pt",
    data="path/to/ragweed_dataset.yaml",
    device="0",
)

print("=== Default TrainingConfig (Chapter Table 3) ===")
for key, value in asdict(cfg).items():
    print(f"  {key:20s} = {value}")

# ------------------------------------------------------------------ #
# 2.  Override specific hyperparameters for experimentation
# ------------------------------------------------------------------ #
cfg_custom = TrainingConfig(
    model="yolo11l.pt",
    data="path/to/ragweed_dataset.yaml",
    epochs=100,            # longer training
    imgsz=1024,            # higher resolution
    batch=32,              # smaller batch to fit in VRAM
    device="0,1",          # dual-GPU DataParallel
    lr0=0.005,             # lower learning rate
    patience=25,           # more patience before early stop
    project="training_results/custom",
    name="experiment_v2",
)

print("\n=== Custom TrainingConfig ===")
print(f"  Model        : {cfg_custom.model}")
print(f"  Image size   : {cfg_custom.imgsz}")
print(f"  Batch size   : {cfg_custom.batch}")
print(f"  Epochs       : {cfg_custom.epochs}")
print(f"  Device       : {cfg_custom.device}")
print(f"  LR           : {cfg_custom.lr0}")
print(f"  Patience     : {cfg_custom.patience}")
print(f"  Output       : {cfg_custom.project}/{cfg_custom.name}_<timestamp>/")

# ------------------------------------------------------------------ #
# 3.  To actually train, uncomment below (requires GPU + dataset):
# ------------------------------------------------------------------ #
# from ragweed_toolkit.detection.trainer import train
# results = train(cfg_custom)
# print(f"Best weights: {results['best_weights']}")
# print(f"mAP50: {results['metrics'].get('metrics/mAP50(B)', 'N/A')}")
