"""
AcousticGuard – Model Definitions
==================================
This module contains the autoencoder architecture that powers AcousticGuard's
anomaly detection. The design philosophy is: train only on healthy machine sounds,
then treat reconstruction error at inference time as the fault signal.

Why an autoencoder? Because we typically have plenty of "normal" machine audio
but very few (or zero) labeled fault examples. An autoencoder learns the manifold
of healthy spectrograms; anything that doesn't lie on that manifold reconstructs
poorly, giving us a natural anomaly score without ever seeing a fault.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig:
    """Central configuration for the autoencoder architecture."""
    input_channels: int = 1        # Grayscale spectrogram
    n_mels: int = 64               # Mel frequency bands (height of spectrogram)
    latent_channels: int = 64      # Bottleneck depth
    base_channels: int = 16        # Channel count at first conv layer
    dropout_rate: float = 0.1      # Dropout in bottleneck (regularisation)


class ConvBlock(nn.Module):
    """
    A reusable encoder block: Conv2D → BatchNorm → ReLU → MaxPool.

    BatchNorm is added here (vs. the original code) to stabilise training,
    especially when the spectrogram pixel distribution varies by machine type.
    """

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeconvBlock(nn.Module):
    """
    A reusable decoder block: ConvTranspose2D → BatchNorm → ReLU.

    ConvTranspose2d reverses the spatial downsampling done by MaxPool2d.
    The final decoder step uses Sigmoid (not ReLU) to clamp output to [0,1],
    matching the normalised spectrogram input range.
    """

    def __init__(self, in_ch: int, out_ch: int, final: bool = False):
        super().__init__()
        activation = nn.Sigmoid() if final else nn.ReLU(inplace=True)
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
        ]
        if not final:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(activation)
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AcousticAutoEncoder(nn.Module):
    """
    Convolutional Autoencoder for acoustic anomaly detection.

    Architecture summary
    --------------------
      Encoder:  (1×64×T) → Conv[16] → Conv[32] → Conv[64] → latent
      Decoder:  latent   → Deconv[32] → Deconv[16] → Deconv[1] → (1×64×T)

    The model is intentionally small (~600 KB) so it can run on edge hardware
    like a Raspberry Pi 4 or an industrial PC without a GPU.

    Usage
    -----
      model = AcousticAutoEncoder()
      recon = model(spectrogram)         # (B, 1, H, W)
      loss  = F.mse_loss(recon, spect)   # reconstruction error = anomaly score
    """

    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        c = cfg.base_channels  # shorthand

        # --- Encoder ---
        # Each ConvBlock halves the spatial dimensions via MaxPool.
        # We deliberately keep the encoder shallow (3 layers) to avoid
        # over-compressing short audio segments.
        self.encoder = nn.Sequential(
            ConvBlock(cfg.input_channels, c),        # →  (16, 32, T/2)
            ConvBlock(c,   c*2),                     # →  (32, 16, T/4)
            ConvBlock(c*2, cfg.latent_channels),     # →  (64,  8, T/8)  ← latent
        )

        # Optional dropout in the bottleneck for regularisation
        self.bottleneck_drop = nn.Dropout2d(p=cfg.dropout_rate)

        # --- Decoder ---
        # Mirror the encoder exactly. The final block uses Sigmoid.
        self.decoder = nn.Sequential(
            DeconvBlock(cfg.latent_channels, c*2),   # →  (32, 16, T/4)
            DeconvBlock(c*2, c),                     # →  (16, 32, T/2)
            DeconvBlock(c, cfg.input_channels, final=True),  # → (1, 64, T)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent representation z for a spectrogram batch."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct a spectrogram from latent vector z."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        z = self.bottleneck_drop(z)
        recon = self.decode(z)

        # Safety: ensure reconstructed shape matches input exactly.
        # Small rounding differences can occur when input width is odd.
        if recon.shape != x.shape:
            recon = F.interpolate(recon, size=x.shape[2:], mode='bilinear', align_corners=False)

        return recon

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample reconstruction MSE — this IS the anomaly score.

        Returns a 1-D tensor of shape (B,) where higher values indicate
        greater deviation from the learned normal distribution.
        """
        self.eval()
        with torch.no_grad():
            recon = self.forward(x)
            # Mean over (C, H, W) dimensions, keep batch dimension
            score = F.mse_loss(recon, x, reduction='none').mean(dim=[1, 2, 3])
        return score

    @staticmethod
    def score_to_label(score: float, threshold: float = 0.035) -> str:
        """Convert a scalar MSE score to a human-readable condition label."""
        if score <= threshold:
            return "NORMAL"
        elif score <= threshold * 2.5:
            return "WARNING"
        else:
            return "ANOMALY"

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"AcousticAutoEncoder("
            f"params={self.parameter_count():,}, "
            f"size≈{self.parameter_count()*4/1024:.0f}KB)"
        )
