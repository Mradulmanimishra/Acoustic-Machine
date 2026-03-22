"""
AcousticGuard – Spectrogram Pre-Generation
==========================================
Run once before training to convert all ESC-50 WAV files into
Mel spectrogram PNG images. This makes data loading ~10x faster
during training because workers read PNGs instead of decoding audio.

Usage:
    python scripts/generate_spectrograms.py
    # or with custom paths:
    python scripts/generate_spectrograms.py \
        --csv_path data/esc50/meta/esc50.csv \
        --audio_dir data/esc50/audio \
        --output_dir spectrograms
"""

import os
import argparse
import torch
import torchaudio
from torchvision.utils import save_image
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x


def generate_spectrograms(
    csv_path:     str,
    audio_dir:    str,
    output_dir:   str,
    target_sr:    int   = 16_000,
    duration_sec: float = 5.0,
    n_mels:       int   = 64,
):
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    df = pd.read_csv(csv_path)
    print(f"  Found {len(df)} audio files in metadata.")

    fixed_len = int(target_sr * duration_sec)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=1024,
        hop_length=256,
        n_mels=n_mels,
        f_min=50,
        f_max=8000,
    ).to(device)

    skipped, processed = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        filename = row['filename']
        category = row['category']

        cat_dir  = output_dir / category
        cat_dir.mkdir(exist_ok=True)

        out_path = cat_dir / filename.replace('.wav', '.png')
        if out_path.exists():
            skipped += 1
            continue

        src_path = Path(audio_dir) / filename
        if not src_path.exists():
            print(f"  Warning: not found {src_path}")
            continue

        try:
            waveform, sr = torchaudio.load(str(src_path))

            # Mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample
            if sr != target_sr:
                waveform = torchaudio.functional.resample(waveform, sr, target_sr)

            # Pad / crop
            if waveform.shape[1] < fixed_len:
                waveform = torch.nn.functional.pad(waveform, (0, fixed_len - waveform.shape[1]))
            else:
                waveform = waveform[:, :fixed_len]

            waveform = waveform.to(device)

            spec = mel_transform(waveform)                # (1, n_mels, T)
            spec = torch.log(spec + 1e-9)                 # Log scale

            s_min, s_max = spec.min(), spec.max()
            if (s_max - s_min) > 1e-9:
                spec = (spec - s_min) / (s_max - s_min)

            save_image(spec, str(out_path))
            processed += 1

        except Exception as e:
            print(f"  Error on {filename}: {e}")

    print(f"\n  Done. Processed={processed}, Skipped={skipped} (already existed).")
    print(f"  Output: {output_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv_path',   default='meta/esc50.csv')
    p.add_argument('--audio_dir',  default='audio')
    p.add_argument('--output_dir', default='spectrograms')
    p.add_argument('--target_sr',  type=int,   default=16000)
    p.add_argument('--n_mels',     type=int,   default=64)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate_spectrograms(
        csv_path=args.csv_path,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        target_sr=args.target_sr,
        n_mels=args.n_mels,
    )
