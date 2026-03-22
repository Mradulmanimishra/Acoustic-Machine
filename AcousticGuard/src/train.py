"""
AcousticGuard – Training Script
=================================
Run training with:
    python src/train.py --config configs/default.yaml

Or with overrides:
    python src/train.py --config configs/default.yaml --epochs 20 --lr 0.0005
"""

import os
import time
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

# Local imports
from model import AcousticAutoEncoder, ModelConfig
from data.loader import get_machine_dataloaders


# ── Pretty console colours ──────────────────────────────────────────────────
class C:
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"


def log(msg, colour=C.CYAN):
    print(f"{colour}{msg}{C.RESET}")


# ── Trainer ─────────────────────────────────────────────────────────────────

class Trainer:
    """
    Encapsulates the full training loop for AcousticGuard.

    The training strategy is intentionally conservative:
      - We only train on NORMAL class samples.
      - The objective is purely reconstruction quality (MSE).
      - Early stopping prevents overfitting to the training set.
      - The best checkpoint by validation accuracy is always saved.

    Why MSE and not something fancier?
    Because MSE on normalised spectrograms gives directly interpretable values.
    A loss of 0.01 means the average pixel deviation is ~10% of the [0,1] range.
    This makes it trivial to set human-understandable detection thresholds.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps'  if torch.backends.mps.is_available() else
            'cpu'
        )
        self.checkpoint_dir = Path(cfg.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        log(f"  Device   : {self.device}", C.BOLD)
        log(f"  Checkpoints → {self.checkpoint_dir}", C.BOLD)

        # Model
        model_cfg = ModelConfig(
            base_channels  = cfg.get('base_channels', 16),
            latent_channels= cfg.get('latent_channels', 64),
            dropout_rate   = cfg.get('dropout_rate', 0.1),
        )
        self.model = AcousticAutoEncoder(model_cfg).to(self.device)
        log(f"  Model    : {self.model}", C.BOLD)

        # Optimiser + LR scheduler
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg.get('learning_rate', 1e-3),
            weight_decay=cfg.get('weight_decay', 1e-5),
        )
        # Cosine annealing: LR decays smoothly, then restarts — helps avoid
        # getting stuck in sharp minima of the spectrogram loss surface.
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.get('epochs', 20),
            eta_min=1e-6,
        )

        # Early-stopping state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = cfg.get('patience', 5)

    # ── Data ────────────────────────────────────────────────────────────────

    def build_dataloaders(self):
        normal_classes = self.cfg.get('normal_classes', ['engine'])
        log(f"\n  Loading normal classes: {normal_classes}")
        self.train_loader, self.val_loader = get_machine_dataloaders(
            class_names=normal_classes,
            batch_size=self.cfg.get('batch_size', 16),
            root_path=self.cfg.get('data_root', '.'),
        )
        log(f"  Train batches: {len(self.train_loader)} | Val batches: {len(self.val_loader)}")

    # ── One epoch ───────────────────────────────────────────────────────────

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            recon = self.model(batch)
            loss  = self.criterion(recon, batch)

            loss.backward()
            # Gradient clipping keeps training stable when a noisy batch
            # produces an unusually large gradient step.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _val_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                recon = self.model(batch)
                loss  = self.criterion(recon, batch)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    @staticmethod
    def _loss_to_accuracy(loss: float) -> float:
        """
        Convert MSE loss to a human-readable 'accuracy' percentage.

        This is a rough proxy — it tells you what fraction of the [0,1]
        pixel range is correctly reconstructed on average.
        Loss of 0 → 100%.  Loss of 1 → 0%.
        """
        return max(0.0, (1.0 - np.sqrt(loss)) * 100)

    # ── Main training loop ──────────────────────────────────────────────────

    def train(self):
        epochs     = self.cfg.get('epochs', 20)
        save_name  = self.cfg.get('save_name', 'best_model.pth')
        save_path  = self.checkpoint_dir / save_name

        log(f"\n{'='*55}", C.BOLD)
        log(f"  Starting training — {epochs} epochs", C.BOLD)
        log(f"{'='*55}", C.BOLD)

        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss = self._train_epoch()
            val_loss   = self._val_epoch()
            val_acc    = self._loss_to_accuracy(val_loss)
            lr         = self.scheduler.get_last_lr()[0]
            elapsed    = time.time() - t0

            self.scheduler.step()

            # Log
            status = ""
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.cfg,
                }, save_path)
                status = f"  {C.GREEN}✓ saved best{C.RESET}"
            else:
                self.patience_counter += 1
                status = f"  patience {self.patience_counter}/{self.patience}"

            colour = C.GREEN if val_acc > 80 else C.YELLOW if val_acc > 70 else C.RED
            print(
                f"  Epoch {epoch:>3}/{epochs} │ "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"{colour}acc={val_acc:.1f}%{C.RESET}  "
                f"lr={lr:.2e}  {elapsed:.1f}s"
                f"{status}"
            )

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Early stopping
            if self.patience_counter >= self.patience:
                log(f"\n  Early stopping triggered at epoch {epoch}.", C.YELLOW)
                break

        log(f"\n  Best val loss : {self.best_val_loss:.4f}", C.GREEN)
        log(f"  Best val acc  : {self._loss_to_accuracy(self.best_val_loss):.1f}%", C.GREEN)
        log(f"  Checkpoint    : {save_path}", C.GREEN)
        return history


# ── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train AcousticGuard autoencoder.")
    p.add_argument('--config', type=str, default='configs/default.yaml')
    p.add_argument('--epochs',    type=int,   default=None)
    p.add_argument('--lr',        type=float, default=None)
    p.add_argument('--batch',     type=int,   default=None)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides have higher priority than config file
    if args.epochs: cfg['epochs']        = args.epochs
    if args.lr:     cfg['learning_rate'] = args.lr
    if args.batch:  cfg['batch_size']    = args.batch

    trainer = Trainer(cfg)
    trainer.build_dataloaders()
    trainer.train()


if __name__ == '__main__':
    main()
