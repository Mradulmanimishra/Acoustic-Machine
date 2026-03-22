"""
AcousticGuard – Visualization Utilities
=========================================
Helpers for plotting spectrograms, reconstruction comparisons,
error distributions, ROC curves, and training history.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import List, Optional


# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1520',
    'axes.facecolor':   '#080c10',
    'axes.edgecolor':   '#1a2d42',
    'axes.labelcolor':  '#8ab4d4',
    'xtick.color':      '#5a7a96',
    'ytick.color':      '#5a7a96',
    'text.color':       '#e2eaf4',
    'grid.color':       '#1a2d42',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'font.family':      'monospace',
})


def plot_reconstruction_comparison(
    original:  np.ndarray,    # (H, W)
    recon:     np.ndarray,    # (H, W)
    mse:       float,
    label:     str,
    save_path: Optional[str] = None,
):
    """
    Side-by-side plot: input spectrogram vs. reconstruction.
    Includes a difference map (|input - recon|) to highlight where
    the model is struggling — this is the fault signature.
    """
    diff = np.abs(original - recon)

    fig = plt.figure(figsize=(14, 5), facecolor='#0d1520')
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.12)

    titles  = ['Input Spectrogram', 'Reconstruction', 'Difference Map (Fault Signature)']
    images  = [original, recon, diff]
    cmaps   = ['inferno', 'inferno', 'hot']

    colour = '#22c55e' if label == 'NORMAL' else '#f59e0b' if label == 'WARNING' else '#ef4444'

    for i, (title, img, cmap) in enumerate(zip(titles, images, cmaps)):
        ax = fig.add_subplot(gs[i])
        im = ax.imshow(img, aspect='auto', origin='lower', cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10, pad=8)
        ax.set_xlabel('Time →', fontsize=8)
        ax.set_ylabel('Frequency (Mel) →' if i == 0 else '', fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    verdict_str = f'Verdict: {label}  |  MSE: {mse:.4f}  |  Acc: {(1 - mse**0.5)*100:.1f}%'
    fig.suptitle(verdict_str, fontsize=13, fontweight='bold', color=colour, y=1.02)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"  Saved → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_error_distribution(
    normal_errors:  List[float],
    anomaly_errors: List[float],
    threshold:      float = 0.035,
    save_path:      Optional[str] = None,
):
    """
    Histogram showing how well the threshold separates normal from anomaly MSE.
    Larger separation = better detection performance.
    """
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0d1520')
    ax.set_facecolor('#080c10')

    bins = np.linspace(0, max(max(normal_errors), max(anomaly_errors)) * 1.1, 50)

    ax.hist(normal_errors,  bins=bins, alpha=0.7, color='#22c55e', label='Normal',  density=True)
    ax.hist(anomaly_errors, bins=bins, alpha=0.7, color='#ef4444', label='Anomaly', density=True)

    ax.axvline(threshold, color='#f59e0b', linestyle='--', linewidth=2,
               label=f'Threshold = {threshold:.3f}')

    ax.set_xlabel('Reconstruction Error (MSE)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Error Distribution: Normal vs. Anomaly', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    else:
        plt.show()
    plt.close()


def plot_training_history(
    train_losses: List[float],
    val_losses:   List[float],
    val_accs:     List[float],
    save_path:    Optional[str] = None,
):
    """Plot train/val loss + accuracy curves across epochs."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor='#0d1520')

    for ax in (ax1, ax2):
        ax.set_facecolor('#080c10')
        ax.grid(True)

    ax1.plot(epochs, train_losses, color='#00d4ff', linewidth=2, label='Train Loss')
    ax1.plot(epochs, val_losses,   color='#f97316', linewidth=2, label='Val Loss',  linestyle='--')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training & Validation Loss', fontweight='bold')
    ax1.legend()

    ax2.plot(epochs, val_accs, color='#22c55e', linewidth=2)
    ax2.axhline(80, color='#f59e0b', linestyle='--', linewidth=1, label='80% target')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy', fontweight='bold')
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    else:
        plt.show()
    plt.close()
