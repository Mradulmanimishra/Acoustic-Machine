# Contributing to AcousticGuard

Thank you for considering a contribution! Here's how to get started.

## Ways to Contribute

- **New machine types** — share ESC-50 classes or custom audio datasets
- **Model improvements** — VAE, attention, lightweight variants
- **Documentation** — clearer explanations, tutorials, notebooks
- **Bug fixes** — open an issue describing the problem first

## Development Setup

```bash
git clone https://github.com/yourusername/AcousticGuard.git
cd AcousticGuard
pip install -r requirements.txt
pip install pytest black flake8  # dev tools
```

## Pull Request Guidelines

1. Fork the repo and create a feature branch: `git checkout -b feature/my-improvement`
2. Keep commits small and focused
3. Run `black src/` before committing (we use Black formatting)
4. Add or update tests for any new code
5. Open a PR with a clear description of what changed and why

## Reporting Issues

Please include: Python version, OS, error message, and a minimal repro script.
