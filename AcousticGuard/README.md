<div align="center">

# 🔊 AcousticGuard
### Industrial Machine Fault Detection via Non-Contact Acoustic AI

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Demo](https://img.shields.io/badge/Live_Demo-Try_Now-F97316?style=for-the-badge)](demo/index.html)
[![Stars](https://img.shields.io/github/stars/yourusername/AcousticGuard?style=for-the-badge&color=yellow)](.)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-blueviolet?style=for-the-badge)](CONTRIBUTING.md)

<br/>

> **Listen to your machines before they break.**
>
> AcousticGuard is a production-ready, non-contact acoustic anomaly detection system for industrial machinery. No physical sensors to mount. No downtime to install. Just microphones and intelligence.

<br/>

<img src="assets/demo_preview.gif" alt="AcousticGuard Demo" width="700"/>

<br/>

**[🚀 Live Demo](demo/index.html) · [📄 Patent Document](docs/patent.pdf) · [📓 Notebook](notebooks/exploration.ipynb) · [🐳 Docker](#docker-deployment)**

</div>

---

## ✨ Why AcousticGuard?

Most industrial predictive maintenance systems require mounting vibration sensors directly on machines — an expensive, time-consuming process that causes downtime and is impossible on hot or inaccessible equipment. AcousticGuard flips this model entirely.

| | Traditional PdM | **AcousticGuard** |
|---|---|---|
| **Sensor Installation** | Hours–Days of downtime | ✅ Zero — just place a mic nearby |
| **Cost per Machine** | $500–$5,000 hardware | ✅ <$20 commodity microphone |
| **Fault Detection Time** | After thermal/visual signs appear | ✅ Early acoustic pattern shifts |
| **Scalability** | One sensor per machine point | ✅ One mic monitors multiple units |
| **Noise Robustness** | Highly susceptible | ✅ Preprocessing pipeline filters background |
| **Deployment** | Requires specialist technician | ✅ Anyone can set up |

---

## 🎯 How It Works — The Core Idea

The central insight is elegant: **healthy machines are boring**. Their acoustic signatures are repetitive and predictable. Faults introduce novelty — irregular spikes, frequency shifts, new harmonic content.

AcousticGuard exploits this by training a **Convolutional Autoencoder exclusively on normal machine sounds**. The autoencoder learns to compress and reconstruct healthy Mel spectrograms with very low error. When you feed it a faulty sound, it *fails to reconstruct it well* — that reconstruction error IS the fault signal.

```
Audio File → Mel Spectrogram → [Encoder → Latent Vector → Decoder] → Reconstructed Spectrogram
                                                                              ↓
                                                          Reconstruction Error (MSE)
                                                                              ↓
                                              Low Error = NORMAL ✅  |  High Error = ANOMALY ⚠️
```

No labeled fault data required. No need to know what type of fault. Just learn "normal" — and anything sufficiently different is a warning.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ACOUSTICGUARD PIPELINE                       │
├──────────────┬──────────────┬───────────────┬───────────────────────┤
│   CAPTURE    │  PREPROCESS  │   ANALYZE     │       OUTPUT          │
│              │              │               │                       │
│  Microphone  │  Resample    │  Encoder:     │  🟢 NORMAL            │
│  (1–4 mics)  │  → 16 kHz    │  Conv2D x3    │  🟡 WARNING           │
│              │  Normalize   │  → Latent [z] │  🔴 ANOMALY           │
│  .wav file   │  Pad/Crop    │               │                       │
│  Live stream │  Mel→Log→    │  Decoder:     │  Confidence Score     │
│              │  Normalize   │  ConvTrans x3 │  Trend History        │
│              │  → [0,1]     │  → Recon.     │  Alert / Webhook      │
└──────────────┴──────────────┴───────────────┴───────────────────────┘
```

### Model Architecture

The autoencoder uses a symmetric encoder-decoder design optimized for spectrogram images:

```python
Encoder: Conv2D(1→16) → Conv2D(16→32) → Conv2D(32→64) → Latent [64 × 8 × 39]
Decoder: ConvTrans(64→32) → ConvTrans(32→16) → ConvTrans(16→1) → Sigmoid
Model Size: ~600 KB  |  Inference: 20–45 ms/frame  |  Edge-deployable ✅
```

---

## 📊 Performance

Results measured on ESC-50 benchmark dataset under varying noise conditions:

| Metric | Value |
|---|---|
| Detection Accuracy (lab) | **78–82%** |
| Noise-Robust Accuracy | **~80%** |
| Acoustic Sampling Rate | **8–16 kHz** |
| Inference Latency (CPU) | **20–45 ms/frame** |
| Model Size | **~600 KB** |
| Deployment Time | **< 1 hour** (vs. days for vibration systems) |

*See `notebooks/exploration.ipynb` for full benchmarks and comparison plots.*

---

## 🚀 Quick Start

### Option 1: Try the Live Demo (No Installation)

Simply open `demo/index.html` in your browser. Upload any `.wav` machine sound file and see the diagnostic result instantly — no Python, no GPU, no setup.

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/AcousticGuard.git
cd AcousticGuard

# Install dependencies
pip install -r requirements.txt

# Download ESC-50 dataset (or use your own audio)
make download-data

# Generate spectrograms from audio files
python scripts/generate_spectrograms.py

# Train the model (CPU works fine; GPU accelerates ~10x)
python src/train.py --config configs/default.yaml

# Run inference on a new audio file
python src/inference.py --file path/to/machine_sound.wav

# Or launch the full dashboard
python scripts/dashboard.py
```

### Option 3: Docker (Production)

```bash
docker build -t acousticguard -f docker/Dockerfile .
docker run -p 8080:8080 -v /your/audio:/data acousticguard
# Visit http://localhost:8080
```

---

## 📁 Project Structure

```
AcousticGuard/
├── src/
│   ├── model.py            # Autoencoder + variants (CAE, VAE)
│   ├── train.py            # Full training loop with checkpointing
│   ├── inference.py        # CLI tool for single-file + batch inference
│   ├── data/
│   │   ├── loader.py       # ESC-50 + custom dataset loaders
│   │   └── preprocess.py   # Audio → Mel spectrogram pipeline
│   └── utils/
│       └── visualization.py # Spectrogram plots, ROC curves, dashboards
├── demo/
│   └── index.html          # 🌐 Zero-install browser demo (Web Audio API)
├── configs/
│   └── default.yaml        # All hyperparameters in one place
├── scripts/
│   ├── generate_spectrograms.py
│   └── dashboard.py        # Real-time monitoring dashboard
├── docker/
│   └── Dockerfile          # Production container
├── notebooks/
│   └── exploration.ipynb   # Full walkthrough with visualizations
├── assets/                 # Images, GIFs for README
├── requirements.txt
├── setup.py
└── Makefile                # Common commands
```

---

## 🔌 Supported Machine Types

AcousticGuard has been tested and works well with motors, compressors, fans, pumps, gear systems, conveyor belts, vacuum systems, and washing machines. Any rotating or cyclically-operating machinery with a consistent acoustic signature qualifies — if the machine sounds "the same" when healthy, AcousticGuard can learn it.

---

## 📡 Deployment Modes

**Edge (Local)** — Run on a Raspberry Pi or industrial PC right next to the machine. Sub-50ms latency, no internet required. Ideal for air-gapped factories.

**Server/Cloud** — Batch-process audio files on a central server, aggregate results across a fleet of machines, and send alerts via webhook or email.

**Hybrid** — Edge devices capture and preprocess audio; cloud handles model retraining and fleet-level dashboards.

---

## 🛣️ Roadmap

- [x] Core autoencoder with Mel spectrogram features
- [x] ESC-50 benchmark integration
- [x] CLI inference tool
- [x] Live browser demo
- [ ] Real-time microphone streaming mode
- [ ] REST API endpoint (FastAPI)
- [ ] Multi-machine fleet dashboard (React)
- [ ] Fault type classification (bearing/gear/rotor)
- [ ] ONNX export for embedded deployment
- [ ] HuggingFace model hub upload
- [ ] Streamlit web app

---

## 📚 Research Background

This project implements concepts described in our patent application:
**"System and Method for Acoustic-Based Predictive Maintenance"** (see `docs/patent.pdf`)

Related prior art and academic context:
- Kim et al. (2021) — Deep CNN for acoustic fault detection, 96–98% on clean data
- Jo et al. (2023) — Autoencoder-based anomaly detection, 88–92%
- US Patent 10991381 B2 — ML predictive maintenance via auditory detection
- US Patent 6772633 B2 — Acoustics-based diagnostics for mechanical systems

The innovation of AcousticGuard is the **end-to-end system design** — combining noise-resilient preprocessing, hybrid temporal-spectral analysis, and adaptive learning into a single deployable unit, versus research models that require clean controlled environments.

---

## 🤝 Contributing

Contributions are warmly welcome! If you have factory audio data, new machine types, or ideas for improving the model architecture, please open an issue or a pull request. See `CONTRIBUTING.md` for guidelines.

---

## 📄 License

MIT License — use it, build on it, deploy it. See `LICENSE` for details.

---

<div align="center">

**Built with ❤️ for factories that can't afford to stop.**

*If AcousticGuard is useful to you, please ⭐ the repo — it helps others find it.*

</div>
