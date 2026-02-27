#!/usr/bin/env python3
"""
================================================================================
YOLO Training CLI with Real-Time Visualization
================================================================================

A standalone training script optimized for dual RTX 4090 GPUs.
Supports YOLOv8, YOLOv11, and YOLO26 models.

USAGE:
------
    # Quick test (3 epochs) to verify setup
    python scripts/cli/train_yolo_cli.py --test --device 0 --batch 16

    # YOLOv11 training (50 epochs) on single GPU
    python scripts/cli/train_yolo_cli.py --model yolo11m.pt --epochs 50 --batch 16 --device 0

    # YOLO26 training (dual GPU, DataParallel mode)
    python scripts/cli/train_yolo_cli.py --model yolo26m.pt --epochs 50 --batch 24 --device "0,1"

COMPARISON TO ULTRALYTICS BASIC EXAMPLE:
----------------------------------------
    # Ultralytics example (notebook style):
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

    # This script does the same but adds:
    # - CLI argument parsing for configurability
    # - Real-time training curve visualization
    # - Organized output directories with timestamps
    # - Proper error handling and logging
    # - Training summary JSON export

OUTPUT STRUCTURE:
-----------------
    training_results/<project>/<run_name>/
    ├── weights/
    │   ├── best.pt          # Best model checkpoint
    │   └── last.pt          # Final model checkpoint
    ├── training_curves.png  # Loss and mAP plots
    ├── training_metrics.json
    ├── training_summary.json
    └── results.csv          # Per-epoch metrics

Author: Claude Code CLI
Date: 2024-01
================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import argparse
import json
import os
import queue
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import torch

matplotlib.use('Agg')  # Use non-interactive backend (no display required)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from ultralytics import YOLO

# ragweed_toolkit library imports

# TODO: Move GPUMonitor and RealTimeTrainingMonitor to ragweed_toolkit.detection
# when API supports real-time training visualization.
# The library's train() function handles basic training but this script adds:
#   - GPU monitoring (GPUMonitor)
#   - Real-time visualization (RealTimeTrainingMonitor)
#   - Custom YOLO callbacks for live metric capture
# These features are too specialized for the core library at this stage.


# ==============================================================================
# GPU MONITORING CLASS
# ==============================================================================
class GPUMonitor:
    """
    Real-time GPU monitoring using nvidia-smi.

    Tracks GPU utilization, memory usage, and temperature in a background thread.
    """

    def __init__(self, interval: float = 2.0):
        """
        Initialize GPU monitor.

        Args:
            interval: Sampling interval in seconds
        """
        self.interval = interval
        self.running = False
        self.thread = None

        # Data storage
        self.timestamps = []
        self.gpu0_util = []
        self.gpu1_util = []
        self.gpu0_mem = []
        self.gpu1_mem = []
        self.gpu0_temp = []
        self.gpu1_temp = []

        self.start_time = None
        self.lock = threading.Lock()

    def _query_nvidia_smi(self):
        """Query nvidia-smi for GPU stats."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return None

            stats = {}
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    idx = int(parts[0])
                    stats[idx] = {
                        'util': float(parts[1]),
                        'mem_used': float(parts[2]) / 1024,  # MB to GB
                        'mem_total': float(parts[3]) / 1024,
                        'temp': float(parts[4])
                    }
            return stats
        except Exception:
            return None

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.running:
            stats = self._query_nvidia_smi()
            if stats:
                elapsed = time.time() - self.start_time

                with self.lock:
                    self.timestamps.append(elapsed)

                    if 0 in stats:
                        self.gpu0_util.append(stats[0]['util'])
                        self.gpu0_mem.append(stats[0]['mem_used'])
                        self.gpu0_temp.append(stats[0]['temp'])
                    else:
                        self.gpu0_util.append(0)
                        self.gpu0_mem.append(0)
                        self.gpu0_temp.append(0)

                    if 1 in stats:
                        self.gpu1_util.append(stats[1]['util'])
                        self.gpu1_mem.append(stats[1]['mem_used'])
                        self.gpu1_temp.append(stats[1]['temp'])
                    else:
                        self.gpu1_util.append(0)
                        self.gpu1_mem.append(0)
                        self.gpu1_temp.append(0)

            time.sleep(self.interval)

    def start(self):
        """Start monitoring in background thread."""
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print("🌡️  GPU monitoring started")

    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("🌡️  GPU monitoring stopped")

    def get_data(self):
        """Get copy of current monitoring data."""
        with self.lock:
            return {
                'timestamps': list(self.timestamps),
                'gpu0_util': list(self.gpu0_util),
                'gpu1_util': list(self.gpu1_util),
                'gpu0_mem': list(self.gpu0_mem),
                'gpu1_mem': list(self.gpu1_mem),
                'gpu0_temp': list(self.gpu0_temp),
                'gpu1_temp': list(self.gpu1_temp),
            }

    def get_current(self):
        """Get current GPU stats."""
        return self._query_nvidia_smi()


# ==============================================================================
# REAL-TIME TRAINING MONITOR CLASS
# ==============================================================================
class RealTimeTrainingMonitor:
    """
    Monitors training progress and generates real-time visualization plots.

    This class collects metrics at the end of each epoch and updates
    matplotlib plots showing:
    - Training loss over time
    - Validation mAP50 and mAP50-95
    - GPU utilization and memory
    - GPU temperature
    - Loss components (box, cls, dfl)
    - Precision and Recall curves

    Plots are saved to PNG files for later review.

    Attributes:
        output_dir (Path): Directory where plots will be saved
        max_epochs (int): Total number of epochs for progress display
        epochs (list): List of completed epoch numbers
        train_loss (list): Total training loss per epoch
        val_map50 (list): Validation mAP@0.5 per epoch
        gpu_monitor (GPUMonitor): GPU monitoring instance
        ...
    """

    def __init__(self, output_dir: str, max_epochs: int, gpu_monitor: GPUMonitor = None):
        """
        Initialize the training monitor.

        Args:
            output_dir: Directory path where outputs will be saved
            max_epochs: Total epochs for training (used in title display)
            gpu_monitor: GPUMonitor instance for GPU stats
        """
        self.output_dir = Path(output_dir)
        self.max_epochs = max_epochs
        self.gpu_monitor = gpu_monitor
        self.metrics_queue = queue.Queue()

        # Storage for metrics history
        # Each list stores one value per epoch
        self.epochs = []           # [1, 2, 3, ...]
        self.train_loss = []       # Total training loss
        self.box_loss = []         # Bounding box regression loss
        self.cls_loss = []         # Classification loss
        self.dfl_loss = []         # Distribution focal loss (YOLO-specific)
        self.val_map50 = []        # mAP at IoU=0.5
        self.val_map50_95 = []     # mAP at IoU=0.5:0.95
        self.precision = []        # Detection precision
        self.recall = []           # Detection recall

        # Matplotlib figure reference
        self.fig = None
        self.axes = None
        self.running = True
        self.update_count = 0

    def setup_plots(self):
        """
        Initialize matplotlib figure with 3x2 subplot grid.

        Creates six subplots:
        - Top-left: Training Loss
        - Top-right: Validation mAP
        - Middle-left: GPU Utilization
        - Middle-right: GPU Memory
        - Bottom-left: GPU Temperature
        - Bottom-right: Loss Components
        """
        plt.ion()  # Enable interactive mode for updates
        self.fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=self.fig, hspace=0.3, wspace=0.25)

        self.axes = {
            'loss': self.fig.add_subplot(gs[0, 0]),
            'map': self.fig.add_subplot(gs[0, 1]),
            'gpu_util': self.fig.add_subplot(gs[1, 0]),
            'gpu_mem': self.fig.add_subplot(gs[1, 1]),
            'gpu_temp': self.fig.add_subplot(gs[2, 0]),
            'loss_components': self.fig.add_subplot(gs[2, 1]),
        }

        self._setup_axes()
        self.fig.suptitle('YOLO Training Progress with GPU Monitoring', fontsize=14, fontweight='bold')
        self.fig.canvas.draw()
        plt.pause(0.1)

    def _setup_axes(self):
        """Configure axes appearance."""
        # Training Loss
        self.axes['loss'].set_title('Training Loss', fontweight='bold')
        self.axes['loss'].set_xlabel('Epoch')
        self.axes['loss'].set_ylabel('Loss')
        self.axes['loss'].grid(True, alpha=0.3)

        # Validation mAP
        self.axes['map'].set_title('Validation mAP', fontweight='bold')
        self.axes['map'].set_xlabel('Epoch')
        self.axes['map'].set_ylabel('mAP')
        self.axes['map'].set_ylim(0, 1)
        self.axes['map'].grid(True, alpha=0.3)

        # GPU Utilization
        self.axes['gpu_util'].set_title('GPU Utilization', fontweight='bold')
        self.axes['gpu_util'].set_xlabel('Time (minutes)')
        self.axes['gpu_util'].set_ylabel('Utilization %')
        self.axes['gpu_util'].set_ylim(0, 100)
        self.axes['gpu_util'].grid(True, alpha=0.3)

        # GPU Memory
        self.axes['gpu_mem'].set_title('GPU Memory Usage', fontweight='bold')
        self.axes['gpu_mem'].set_xlabel('Time (minutes)')
        self.axes['gpu_mem'].set_ylabel('Memory (GB)')
        self.axes['gpu_mem'].set_ylim(0, 26)
        self.axes['gpu_mem'].axhline(y=24, color='gray', linestyle='--', alpha=0.5, label='24GB limit')
        self.axes['gpu_mem'].grid(True, alpha=0.3)

        # GPU Temperature
        self.axes['gpu_temp'].set_title('GPU Temperature', fontweight='bold')
        self.axes['gpu_temp'].set_xlabel('Time (minutes)')
        self.axes['gpu_temp'].set_ylabel('Temperature (C)')
        self.axes['gpu_temp'].set_ylim(20, 90)
        self.axes['gpu_temp'].axhline(y=83, color='r', linestyle='--', alpha=0.5, label='Thermal limit')
        self.axes['gpu_temp'].grid(True, alpha=0.3)

        # Loss Components
        self.axes['loss_components'].set_title('Loss Components', fontweight='bold')
        self.axes['loss_components'].set_xlabel('Epoch')
        self.axes['loss_components'].set_ylabel('Loss')
        self.axes['loss_components'].grid(True, alpha=0.3)

    def update_plots(self):
        """
        Redraw all plots with current metric data.

        Called after each epoch to refresh the visualization.
        Uses matplotlib's draw() and flush_events() for updates.
        """
        self.update_count += 1

        try:
            # Clear all axes
            for name, ax in self.axes.items():
                ax.clear()
            self._setup_axes()

            epochs = np.array(self.epochs) if self.epochs else np.array([])

            # ---- Plot 1: Total training loss ----
            if self.train_loss:
                self.axes['loss'].plot(epochs, self.train_loss, 'b-', linewidth=2, label='Total Loss')
                self.axes['loss'].legend(loc='upper right')

            # ---- Plot 2: Validation mAP metrics ----
            if self.val_map50:
                self.axes['map'].plot(epochs, self.val_map50, 'g-', linewidth=2, label='mAP50')
                if self.val_map50_95:
                    self.axes['map'].plot(epochs, self.val_map50_95, 'b--', linewidth=2, label='mAP50-95')
                # Mark best point
                best_idx = np.argmax(self.val_map50)
                self.axes['map'].scatter([epochs[best_idx]], [self.val_map50[best_idx]],
                                         color='gold', s=100, marker='*', zorder=5, label=f'Best: {self.val_map50[best_idx]:.3f}')
                self.axes['map'].legend(loc='lower right')

            # ---- Plot 3 & 4 & 5: GPU stats ----
            if self.gpu_monitor:
                gpu_data = self.gpu_monitor.get_data()
                if gpu_data['timestamps']:
                    times_min = [t / 60 for t in gpu_data['timestamps']]

                    # GPU Utilization
                    self.axes['gpu_util'].plot(times_min, gpu_data['gpu0_util'], 'b-',
                                               linewidth=1.5, label='GPU 0', alpha=0.8)
                    self.axes['gpu_util'].plot(times_min, gpu_data['gpu1_util'], 'r-',
                                               linewidth=1.5, label='GPU 1', alpha=0.8)
                    self.axes['gpu_util'].fill_between(times_min, gpu_data['gpu0_util'], alpha=0.2, color='b')
                    self.axes['gpu_util'].fill_between(times_min, gpu_data['gpu1_util'], alpha=0.2, color='r')
                    self.axes['gpu_util'].legend(loc='upper right')

                    # GPU Memory
                    self.axes['gpu_mem'].plot(times_min, gpu_data['gpu0_mem'], 'b-',
                                              linewidth=1.5, label='GPU 0', alpha=0.8)
                    self.axes['gpu_mem'].plot(times_min, gpu_data['gpu1_mem'], 'r-',
                                              linewidth=1.5, label='GPU 1', alpha=0.8)
                    self.axes['gpu_mem'].legend(loc='upper right')

                    # GPU Temperature
                    self.axes['gpu_temp'].plot(times_min, gpu_data['gpu0_temp'], 'b-',
                                               linewidth=1.5, label='GPU 0', alpha=0.8)
                    self.axes['gpu_temp'].plot(times_min, gpu_data['gpu1_temp'], 'r-',
                                               linewidth=1.5, label='GPU 1', alpha=0.8)
                    self.axes['gpu_temp'].legend(loc='upper right')

            # ---- Plot 6: Individual loss components ----
            if self.box_loss:
                self.axes['loss_components'].plot(epochs, self.box_loss, 'r-', linewidth=1.5, label='Box')
                self.axes['loss_components'].plot(epochs, self.cls_loss, 'g-', linewidth=1.5, label='Cls')
                self.axes['loss_components'].plot(epochs, self.dfl_loss, 'b-', linewidth=1.5, label='DFL')
                self.axes['loss_components'].legend(loc='upper right')

            # Update main title with current progress
            current_epoch = self.epochs[-1] if self.epochs else 0
            best_map = max(self.val_map50) if self.val_map50 else 0

            # Get current GPU temps if available
            gpu_temp_str = ""
            if self.gpu_monitor:
                current_stats = self.gpu_monitor.get_current()
                if current_stats:
                    temps = [current_stats.get(i, {}).get('temp', 0) for i in [0, 1] if i in current_stats]
                    if temps:
                        gpu_temp_str = f" | GPU Temps: {'/'.join(f'{t:.0f}C' for t in temps)}"

            self.fig.suptitle(
                f'YOLO Training - Epoch {current_epoch}/{self.max_epochs} | Best mAP50: {best_map:.4f}{gpu_temp_str}',
                fontsize=14, fontweight='bold'
            )

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)

            # Save plot periodically
            if self.update_count % 2 == 0:
                self._save_snapshot()

        except Exception as e:
            print(f"[!] Plot update error: {e}")

    def _save_snapshot(self):
        """Save current dashboard state to file."""
        try:
            self.fig.savefig(self.output_dir / 'training_curves_live.png', dpi=100, bbox_inches='tight')
        except Exception:
            pass

    def add_epoch_metrics(self, epoch: int, metrics: dict):
        """
        Record metrics for a completed epoch and update plots.

        Args:
            epoch: The epoch number (1-indexed)
            metrics: Dictionary containing metric values, expected keys:
                - train/loss, train/box_loss, train/cls_loss, train/dfl_loss
                - metrics/mAP50(B), metrics/mAP50-95(B)
                - metrics/precision(B), metrics/recall(B)
        """
        self.epochs.append(epoch)

        # Extract and store training losses
        self.train_loss.append(metrics.get('train/loss', 0))
        self.box_loss.append(metrics.get('train/box_loss', 0))
        self.cls_loss.append(metrics.get('train/cls_loss', 0))
        self.dfl_loss.append(metrics.get('train/dfl_loss', 0))

        # Extract and store validation metrics
        self.val_map50.append(metrics.get('metrics/mAP50(B)', 0))
        self.val_map50_95.append(metrics.get('metrics/mAP50-95(B)', 0))
        self.precision.append(metrics.get('metrics/precision(B)', 0))
        self.recall.append(metrics.get('metrics/recall(B)', 0))

        # Refresh plots and save metrics
        self.update_plots()
        self.save_metrics()

    def save_metrics(self):
        """Save all collected metrics to a JSON file for later analysis."""
        metrics_file = self.output_dir / 'training_metrics.json'
        data = {
            'epochs': self.epochs,
            'train_loss': self.train_loss,
            'box_loss': self.box_loss,
            'cls_loss': self.cls_loss,
            'dfl_loss': self.dfl_loss,
            'val_map50': self.val_map50,
            'val_map50_95': self.val_map50_95,
            'precision': self.precision,
            'recall': self.recall
        }
        with open(metrics_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_results_csv(self):
        """Load metrics from YOLO's results.csv file after training."""
        csv_path = self.output_dir / 'results.csv'
        if not csv_path.exists():
            print(f"[!] Results CSV not found: {csv_path}")
            return False

        try:
            import csv
            # Clear existing data
            self.epochs = []
            self.train_loss = []
            self.box_loss = []
            self.cls_loss = []
            self.dfl_loss = []
            self.val_map50 = []
            self.val_map50_95 = []
            self.precision = []
            self.recall = []

            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Clean column names (remove spaces)
                    row = {k.strip(): v for k, v in row.items()}

                    epoch = int(row.get('epoch', 0)) + 1  # 0-indexed to 1-indexed
                    self.epochs.append(epoch)
                    self.box_loss.append(float(row.get('train/box_loss', 0)))
                    self.cls_loss.append(float(row.get('train/cls_loss', 0)))
                    self.dfl_loss.append(float(row.get('train/dfl_loss', 0)))
                    self.train_loss.append(
                        float(row.get('train/box_loss', 0)) +
                        float(row.get('train/cls_loss', 0)) +
                        float(row.get('train/dfl_loss', 0))
                    )
                    self.val_map50.append(float(row.get('metrics/mAP50(B)', 0)))
                    self.val_map50_95.append(float(row.get('metrics/mAP50-95(B)', 0)))
                    self.precision.append(float(row.get('metrics/precision(B)', 0)))
                    self.recall.append(float(row.get('metrics/recall(B)', 0)))

            print(f"[+] Loaded {len(self.epochs)} epochs from {csv_path.name}")
            return True

        except Exception as e:
            print(f"[!] Error loading CSV: {e}")
            return False

    def save_final_plot(self):
        """Save the final training curves plot as a high-resolution PNG."""
        # If no epoch data was captured (DDP mode), load from results.csv
        if not self.epochs:
            self.load_from_results_csv()
            if self.epochs:
                self.update_plots()

        plot_file = self.output_dir / 'training_curves.png'
        self.fig.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"📊 Training curves saved to: {plot_file}")

    def close(self):
        """Clean up: save final plot and close matplotlib figure."""
        self.running = False
        if self.fig:
            self.save_final_plot()
            plt.close(self.fig)


# ==============================================================================
# TRAINING CALLBACKS
# ==============================================================================
def create_training_callbacks(monitor: RealTimeTrainingMonitor):
    """
    Create callback functions for the YOLO training loop.

    YOLO's training loop supports callbacks at various events:
    - on_train_start, on_train_end
    - on_train_epoch_start, on_train_epoch_end
    - on_val_start, on_val_end
    - etc.

    We use on_train_epoch_end to collect metrics after each epoch
    and update our real-time plots.

    Args:
        monitor: RealTimeTrainingMonitor instance to receive metrics

    Returns:
        dict: Mapping of event names to callback functions
    """

    def on_train_epoch_end(trainer):
        """
        Called by YOLO at the end of each training epoch.

        Extracts loss values and validation metrics from the trainer
        object and passes them to our monitor for visualization.

        Args:
            trainer: YOLO Trainer object with current state
        """
        epoch = trainer.epoch + 1  # Convert 0-indexed to 1-indexed
        metrics = {}

        # Extract training losses from trainer.loss_items
        # loss_items is a tensor: [box_loss, cls_loss, dfl_loss]
        if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
            loss_items = trainer.loss_items.cpu().numpy() if torch.is_tensor(trainer.loss_items) else trainer.loss_items
            if len(loss_items) >= 3:
                metrics['train/box_loss'] = float(loss_items[0])
                metrics['train/cls_loss'] = float(loss_items[1])
                metrics['train/dfl_loss'] = float(loss_items[2])
                metrics['train/loss'] = float(sum(loss_items))

        # Extract validation metrics from trainer.metrics
        # These are computed after each epoch's validation run
        if hasattr(trainer, 'metrics') and trainer.metrics:
            m = trainer.metrics
            metrics['metrics/mAP50(B)'] = getattr(m, 'map50', 0) or 0
            metrics['metrics/mAP50-95(B)'] = getattr(m, 'map', 0) or 0
            metrics['metrics/precision(B)'] = getattr(m, 'mp', 0) or 0
            metrics['metrics/recall(B)'] = getattr(m, 'mr', 0) or 0

        # Send metrics to monitor for plotting
        monitor.add_epoch_metrics(epoch, metrics)

    def on_train_end(trainer):
        """Called when training completes - save final plot."""
        monitor.save_final_plot()

    return {
        'on_train_epoch_end': on_train_epoch_end,
        'on_train_end': on_train_end
    }


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================
def train_yolo(args):
    """
    Main training function - equivalent to model.train() with extras.

    This function:
    1. Sets up output directories with timestamps
    2. Initializes the YOLO model
    3. Sets up real-time monitoring
    4. Runs training with configured parameters
    5. Saves results and summary

    Args:
        args: Parsed command-line arguments containing:
            - model: Model name (e.g., 'yolo11m.pt')
            - data: Path to dataset YAML
            - epochs, batch, imgsz: Training parameters
            - device: GPU device(s) to use
            - etc.

    Returns:
        Path: Output directory containing training results
    """

    # --------------------------------------------------------------------------
    # STEP 1: Setup paths and directories
    # --------------------------------------------------------------------------
    project_root = Path(__file__).parent.parent.parent  # Go up from scripts/cli/
    data_yaml = project_root / args.data

    # Create timestamped output directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.name}_{timestamp}" if args.name else f"run_{timestamp}"
    output_dir = project_root / args.project / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # STEP 2: Print configuration summary
    # --------------------------------------------------------------------------
    print("=" * 60)
    print("YOLO Training with Real-Time Visualization")
    print("=" * 60)
    print(f"Data: {data_yaml}")
    print(f"Output: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs} (patience: {args.patience})")
    print(f"Batch: {args.batch}")
    print(f"ImgSize: {args.imgsz}")
    print(f"Device: {args.device}")
    print(f"Cache: {args.cache}")
    print("AMP: True (FP16)")
    # Check W&B status
    try:
        import importlib.util
        if importlib.util.find_spec("wandb") is not None:
            wandb_status = "enabled" if os.path.exists(os.path.expanduser("~/.netrc")) else "not configured"
        else:
            wandb_status = "not installed"
    except Exception:
        wandb_status = "not installed"
    print(f"W&B: {wandb_status}")
    print("=" * 60)

    # --------------------------------------------------------------------------
    # STEP 3: Check GPU availability
    # --------------------------------------------------------------------------
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            print(f"🎮 GPU {i}: {torch.cuda.get_device_name(i)}")
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   Memory: {mem:.1f} GB")
    else:
        print("⚠️ No GPU detected, using CPU")

    # --------------------------------------------------------------------------
    # STEP 4: Initialize YOLO model
    # --------------------------------------------------------------------------
    # This is equivalent to: model = YOLO("yolo11m.pt")
    print(f"\n📥 Loading model: {args.model}")
    model = YOLO(args.model)

    # --------------------------------------------------------------------------
    # STEP 5: Setup GPU monitoring and real-time visualization
    # --------------------------------------------------------------------------
    gpu_monitor = GPUMonitor(interval=2.0)
    gpu_monitor.start()

    monitor = RealTimeTrainingMonitor(output_dir, args.epochs, gpu_monitor)
    monitor.setup_plots()

    # Register our callbacks with the model
    callbacks = create_training_callbacks(monitor)
    for event, callback in callbacks.items():
        model.add_callback(event, callback)

    # --------------------------------------------------------------------------
    # STEP 6: Configure training arguments
    # --------------------------------------------------------------------------
    # These are the same parameters you'd pass to model.train()
    # See: https://docs.ultralytics.com/modes/train/#arguments
    train_args = {
        'data': str(data_yaml),       # Path to dataset YAML
        'epochs': args.epochs,         # Number of training epochs
        'batch': args.batch,           # Batch size
        'imgsz': args.imgsz,           # Input image size
        'device': args.device,         # GPU device(s): 0, "0,1", "cpu"
        'workers': args.workers,       # DataLoader workers
        'cache': args.cache,           # Cache images in RAM for speed
        'patience': args.patience,     # Early stopping patience (0 = disabled)
        'project': str(output_dir.parent),
        'name': output_dir.name,
        'exist_ok': True,              # Overwrite existing run
        'pretrained': True,            # Use pretrained weights
        'optimizer': 'auto',           # Auto-select optimizer
        'verbose': True,               # Print detailed output
        'seed': 42,                    # Random seed for reproducibility
        'deterministic': True,         # Deterministic training
        'plots': True,                 # Generate training plots
        'save': True,                  # Save checkpoints
        'save_period': 10,             # Save checkpoint every N epochs
        'amp': True,                   # Automatic Mixed Precision (FP16)
    }

    # --------------------------------------------------------------------------
    # STEP 7: Add augmentation settings (optional)
    # --------------------------------------------------------------------------
    if not args.no_augment:
        train_args.update({
            'hsv_h': 0.015,        # Hue augmentation
            'hsv_s': 0.7,          # Saturation augmentation
            'hsv_v': 0.4,          # Value augmentation
            'degrees': 0.0,        # Rotation degrees
            'translate': 0.1,      # Translation fraction
            'scale': 0.5,          # Scale augmentation
            'shear': 0.0,          # Shear augmentation
            'perspective': 0.0,    # Perspective augmentation
            'flipud': 0.0,         # Vertical flip probability
            'fliplr': 0.5,         # Horizontal flip probability
            'mosaic': 1.0,         # Mosaic augmentation probability
            'mixup': 0.0,          # Mixup augmentation probability
        })

    # --------------------------------------------------------------------------
    # STEP 8: Run training
    # --------------------------------------------------------------------------
    try:
        print("\n🏋️ Starting training...")
        start_time = time.time()

        # THIS IS THE CORE TRAINING CALL
        # Equivalent to: results = model.train(data=..., epochs=..., ...)
        results = model.train(**train_args)

        elapsed = time.time() - start_time
        print(f"\n✅ Training completed in {elapsed/60:.1f} minutes")

        # Print final metrics
        if results:
            print("\n📊 Final Metrics:")
            print(f"   mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
            print(f"   mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
            print(f"   Precision: {results.results_dict.get('metrics/precision(B)', 0):.4f}")
            print(f"   Recall: {results.results_dict.get('metrics/recall(B)', 0):.4f}")

        # --------------------------------------------------------------------------
        # STEP 9: Save training summary
        # --------------------------------------------------------------------------
        summary = {
            'model': args.model,
            'epochs': args.epochs,
            'batch': args.batch,
            'imgsz': args.imgsz,
            'device': args.device,
            'training_time_minutes': elapsed / 60,
            'best_model': str(output_dir / 'weights' / 'best.pt'),
            'final_metrics': results.results_dict if results else {}
        }
        with open(output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        raise
    finally:
        # Always clean up the monitor and GPU monitor
        gpu_monitor.stop()
        monitor.close()

    return output_dir


# ==============================================================================
# CLI ARGUMENT PARSER
# ==============================================================================
def main():
    """
    Entry point for CLI execution.

    Parses command-line arguments and launches training.
    Run with --help to see all available options.
    """
    parser = argparse.ArgumentParser(
        description='YOLO Training CLI with Real-Time Visualization (YOLOv8/v11/26)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run (3 epochs)
  python train_yolo_cli.py --test --device 0

  # YOLOv11 training on single GPU
  python train_yolo_cli.py --model yolo11m.pt --epochs 50 --batch 16 --device 0

  # YOLO26 training on dual GPU
  python train_yolo_cli.py --model yolo26m.pt --epochs 50 --batch 24 --device "0,1"
        """
    )

    # Model settings
    parser.add_argument('--model', type=str, default='yolo11m.pt',
                        help='Model: yolo11[n/s/m/l/x].pt, yolo26[n/s/m/l/x].pt, yolov8[n/s/m/l/x].pt')
    parser.add_argument('--data', type=str,
                        default='configs/yaml_config/data_tomate-orobanche-v4-4488tiles.yaml',
                        help='Path to data YAML file (relative to project root)')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size (reduce if OOM). Safe: 12 for X@1024, 16 for M@1024')
    parser.add_argument('--imgsz', type=int, default=1024, help='Input image size')
    parser.add_argument('--device', type=str, default='0', help='Device: 0, 1, "0,1", or "cpu"')
    parser.add_argument('--workers', type=int, default=8, help='DataLoader workers')

    # Early stopping
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (epochs without improvement). Set 0 to disable.')
    parser.add_argument('--min-delta', type=float, default=0.001,
                        help='Minimum improvement to reset patience counter')

    # Caching options
    parser.add_argument('--cache', action='store_true', default=True,
                        help='Cache images in RAM (faster but uses more memory)')
    parser.add_argument('--no-cache', action='store_false', dest='cache',
                        help='Disable image caching')

    # Output settings
    parser.add_argument('--project', type=str, default='training_results/v4_4140tiles',
                        help='Project directory for outputs')
    parser.add_argument('--name', type=str, default='', help='Run name (timestamp added)')

    # Augmentation
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')

    # Quick test mode
    parser.add_argument('--test', action='store_true',
                        help='Quick test with 3 epochs only')

    args = parser.parse_args()

    # Override settings for test mode
    if args.test:
        args.epochs = 3
        args.name = 'test_run'
        print("🧪 Test mode: Running 3 epochs only")

    # Launch training
    output_dir = train_yolo(args)
    print(f"\n📁 Results saved to: {output_dir}")


# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    main()
