"""
AcousticGuard – Inference CLI
================================
Run on a single audio file:
    python src/inference.py --file path/to/sound.wav

Batch evaluation (compare normal vs. anomaly folder):
    python src/inference.py --normal_dir data/normal/ --anomaly_dir data/anomaly/

List supported machine classes:
    python src/inference.py --list_classes
"""

import os
import argparse
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path


# ─── ANSI colours (gracefully disabled on Windows cmd) ─────────────────────
try:
    import os; os.get_terminal_size()  # will throw if not a TTY
    GREEN  = "\033[92m"; YELLOW = "\033[93m"; RED    = "\033[91m"
    CYAN   = "\033[96m"; BOLD   = "\033[1m";  RESET  = "\033[0m"
    BOX    = {"tl":"╔","tr":"╗","bl":"╚","br":"╝","h":"═","v":"║"}
except Exception:
    GREEN = YELLOW = RED = CYAN = BOLD = RESET = ""
    BOX   = {"tl":"+","tr":"+","bl":"+","br":"+","h":"-","v":"|"}


# ─── Preprocessing ──────────────────────────────────────────────────────────

def preprocess_audio(
    file_path:    str,
    target_sr:    int = 16_000,
    duration_sec: float = 5.0,
    n_mels:       int = 64,
    device:       torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Load and standardise a WAV/MP3/OGG file into a normalised Mel spectrogram.

    The output tensor has shape (1, 1, n_mels, T) — exactly the format the
    autoencoder expects: (batch=1, channels=1, height=n_mels, width=time).

    Design choices:
      - Resampling to 16 kHz is a deliberate trade-off. We lose frequencies
        above 8 kHz, but we gain a much smaller model and faster inference.
        Most machine fault signatures live well below 4 kHz anyway.
      - Log-scaling the Mel energies (log(x + ε)) compresses the dynamic range,
        making faint-but-characteristic frequency bands visible to the CNN.
      - Per-sample normalisation to [0,1] ensures the model's Sigmoid output
        is always compared against the same input scale, regardless of how
        loud the original recording was.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # ── Load ──────────────────────────────────────────────────────────────
    waveform, sr = torchaudio.load(str(path))

    # Collapse to mono by averaging channels
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # ── Resample ──────────────────────────────────────────────────────────
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    # ── Pad / Crop to fixed duration ──────────────────────────────────────
    fixed_len = int(target_sr * duration_sec)
    if waveform.shape[1] < fixed_len:
        pad = fixed_len - waveform.shape[1]
        waveform = F.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, :fixed_len]

    # ── Mel Spectrogram ───────────────────────────────────────────────────
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=1024,
        hop_length=256,
        n_mels=n_mels,
        f_min=50,    # Ignore sub-50Hz rumble (often just AC hum)
        f_max=8000,
    ).to(device)

    waveform = waveform.to(device)
    spec = mel_transform(waveform)            # (1, n_mels, T)

    # ── Log scale ─────────────────────────────────────────────────────────
    spec = torch.log(spec + 1e-9)

    # ── Normalise per-sample to [0, 1] ────────────────────────────────────
    s_min, s_max = spec.min(), spec.max()
    if (s_max - s_min) > 1e-9:
        spec = (spec - s_min) / (s_max - s_min)
    else:
        spec = spec * 0  # silent file → all zeros

    # Add batch dimension: (1, n_mels, T) → (1, 1, n_mels, T)
    return spec.unsqueeze(0)


# ─── Prediction ─────────────────────────────────────────────────────────────

def predict(
    model,
    file_path: str,
    threshold: float = 0.035,
    device:    torch.device = torch.device('cpu'),
) -> dict:
    """
    Run the full diagnosis pipeline on one audio file.

    Returns a dict with keys: file, mse, accuracy, label, confidence.
    """
    spec = preprocess_audio(file_path, device=device)

    model.eval()
    with torch.no_grad():
        recon = model(spec)
        mse   = F.mse_loss(recon, spec).item()

    acc   = max(0.0, (1.0 - np.sqrt(mse)) * 100)
    label = model.score_to_label(mse, threshold)

    # Confidence: distance from threshold, capped at 99%
    dist_normal  = max(0, threshold - mse) / threshold
    dist_anomaly = max(0, mse - threshold) / threshold
    confidence   = min(0.99, max(dist_normal, dist_anomaly) * 2.5 + 0.65)

    return {
        'file':       Path(file_path).name,
        'mse':        mse,
        'accuracy':   acc,
        'label':      label,
        'confidence': confidence,
    }


# ─── Output formatting ───────────────────────────────────────────────────────

def print_result(result: dict, threshold: float = 0.035):
    label = result['label']
    mse   = result['mse']
    acc   = result['accuracy']
    conf  = result['confidence']

    colour = GREEN if label == "NORMAL" else YELLOW if label == "WARNING" else RED
    icon   = "✅" if label == "NORMAL" else "⚠️ " if label == "WARNING" else "🔴"

    bar_len = 30
    filled  = int(min(1.0, mse / (threshold * 3)) * bar_len)
    bar_col = GREEN if mse <= threshold else YELLOW if mse <= threshold * 2.5 else RED
    bar     = bar_col + "█" * filled + RESET + "░" * (bar_len - filled)

    print()
    print(f"{BOX['tl']}" + BOX['h'] * 52 + f"{BOX['tr']}")
    print(f"{BOX['v']}  {BOLD}ACOUSTICGUARD DIAGNOSTIC RESULT{RESET}              {BOX['v']}")
    print(f"{BOX['v']}  File      : {result['file']:<38}{BOX['v']}")
    print(f"{BOX['v']}  MSE Score : {mse:.4f}  [{bar}]{BOX['v']}")
    print(f"{BOX['v']}  Recon Acc : {acc:.1f}%                                {BOX['v']}")
    print(f"{BOX['v']}  Confidence: {conf*100:.0f}%                                 {BOX['v']}")
    print(f"{BOX['v']}                                                    {BOX['v']}")
    print(f"{BOX['v']}  {icon}  {colour}{BOLD}{label:<48}{RESET}{BOX['v']}")
    print(f"{BOX['bl']}" + BOX['h'] * 52 + f"{BOX['br']}")
    print()


def print_batch_summary(results: list[dict], threshold: float = 0.035):
    """Print a table summarising batch evaluation results."""
    normal  = [r for r in results if r['label'] == 'NORMAL']
    warning = [r for r in results if r['label'] == 'WARNING']
    anomaly = [r for r in results if r['label'] == 'ANOMALY']

    print(f"\n{'─'*55}")
    print(f"  {BOLD}BATCH SUMMARY{RESET}   ({len(results)} files total)")
    print(f"{'─'*55}")
    print(f"  {GREEN}NORMAL  :{RESET} {len(normal):>4} files  ({len(normal)/len(results)*100:.0f}%)")
    print(f"  {YELLOW}WARNING :{RESET} {len(warning):>4} files  ({len(warning)/len(results)*100:.0f}%)")
    print(f"  {RED}ANOMALY :{RESET} {len(anomaly):>4} files  ({len(anomaly)/len(results)*100:.0f}%)")
    print(f"{'─'*55}")
    mses = [r['mse'] for r in results]
    print(f"  Avg MSE : {np.mean(mses):.4f}   Max: {max(mses):.4f}   Min: {min(mses):.4f}")
    print()


# ─── Entry point ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="AcousticGuard — Inference CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python src/inference.py --file machine_sound.wav

  # Batch: compare a folder of normals vs. a folder of anomalies
  python src/inference.py --normal_dir data/normal/ --anomaly_dir data/fault/

  # Custom threshold
  python src/inference.py --file sound.wav --threshold 0.045
        """,
    )
    p.add_argument('--file',          type=str,   help='Path to a .wav file')
    p.add_argument('--normal_dir',    type=str,   help='Folder of normal audio files')
    p.add_argument('--anomaly_dir',   type=str,   help='Folder of anomaly audio files')
    p.add_argument('--model_path',    type=str,   default='checkpoints/best_model.pth')
    p.add_argument('--threshold',     type=float, default=0.035)
    p.add_argument('--list_classes',  action='store_true', help='List supported ESC-50 machine classes')
    return p.parse_args()


MACHINE_CLASSES = [
    'engine', 'chainsaw', 'vacuum_cleaner', 'washing_machine',
    'keyboard_typing', 'mouse_click', 'clock_tick', 'water_drops',
    'drilling', 'hand_saw',
]


def main():
    args = parse_args()

    if args.list_classes:
        print(f"\n  Supported machine classes ({len(MACHINE_CLASSES)}):")
        for c in MACHINE_CLASSES:
            print(f"    • {c}")
        print()
        return

    # ── Load model ──────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Import here to avoid circular dependency at module level
    from model import AcousticAutoEncoder
    model = AcousticAutoEncoder().to(device)

    if not Path(args.model_path).exists():
        print(f"{RED}  ✗ Model checkpoint not found: {args.model_path}{RESET}")
        print(f"    Run `python src/train.py` first to train the model.\n")
        return

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    print(f"{GREEN}  ✓ Loaded model from {args.model_path}{RESET}")

    # ── Single file ─────────────────────────────────────────────────────
    if args.file:
        result = predict(model, args.file, args.threshold, device)
        print_result(result, args.threshold)

    # ── Batch evaluation ─────────────────────────────────────────────────
    elif args.normal_dir or args.anomaly_dir:
        all_results = []
        audio_ext = {'.wav', '.mp3', '.ogg', '.flac'}

        for folder, expected_label in [
            (args.normal_dir,  'NORMAL'),
            (args.anomaly_dir, 'ANOMALY'),
        ]:
            if not folder:
                continue
            files = [f for f in Path(folder).rglob('*') if f.suffix.lower() in audio_ext]
            print(f"\n  Evaluating {len(files)} files from {folder} …")
            for fpath in files:
                try:
                    r = predict(model, str(fpath), args.threshold, device)
                    r['expected'] = expected_label
                    all_results.append(r)
                    status = GREEN+"✓"+RESET if r['label']==expected_label else RED+"✗"+RESET
                    print(f"    {status}  {r['file']:<45}  {r['label']:<8}  MSE={r['mse']:.4f}")
                except Exception as e:
                    print(f"    {RED}✗ Error on {fpath.name}: {e}{RESET}")

        if all_results:
            print_batch_summary(all_results, args.threshold)

            # Accuracy vs expected labels
            correct = sum(1 for r in all_results if r.get('expected') == r['label'])
            print(f"  Detection accuracy vs ground truth: {correct}/{len(all_results)} = {correct/len(all_results)*100:.1f}%\n")

    else:
        print("  No input specified. Use --file, --normal_dir/--anomaly_dir, or --list_classes.")
        print("  Try: python src/inference.py --help\n")


if __name__ == '__main__':
    main()
