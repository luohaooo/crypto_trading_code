# Repository Guidelines

## Project Structure & Module Organization
- `visualization/` serves the Flask-based charting UI (`app.py`, Jinja templates, static assets).
- `figure_model/` contains PyTorch training code, datasets, and checkpoints for OHLC figure models.
- `neural-strategy/` houses the factor-driven backtesting engine (`backtest/`, `strategies/`, `utils/`).
- `strategy/backtest/` keeps legacy factor scripts and reports; reuse utilities here before adding new ones.
- `generate_data/` provides notebooks and smoke-test scripts for building cached datasets.
- `utils/data_loader.py` centralizes IO helpers; import from here instead of duplicating loaders.
- External OHLCV data lives in `/home/craz/crypto/crypto-data/future_data_2/`; keep paths configurable via env vars when contributing.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` — create an isolated dev environment.
- `pip install -r requirements.txt` plus the appropriate PyTorch wheel for your platform before running training jobs.
- `cd visualization && python app.py` — launch the chart explorer at `http://localhost:5000`.
- `cd figure_model && python train_ohlc_model.py` — start a full training cycle (reads cached data and writes to `model_checkpoint/`).
- `cd neural-strategy && python example.py` — execute the reference market-neutral backtest.
- `cd strategy/backtest && python summarize_backtest.py` — regenerate legacy factor summaries if you touch factor code.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, and `black`-compatible line lengths (~100 chars); run `python -m black <file>` before pushing.
- Prefer explicit type hints and docstrings for public functions; follow snake_case for modules/functions and CamelCase for classes.
- Keep project-root imports deterministic: use `from utils.data_loader import ...` instead of editing `sys.path` in new files.
- Store configuration in `neural-strategy/utils/config.py` or dedicated dataclasses; avoid hard-coding absolute paths in new modules.

## Testing Guidelines
- Lightweight smoke tests live beside the pipelines; run `python generate_data/test_month_cache_loader.py` before touching cache utilities.
- For model or strategy changes, add scenario notebooks or scripts under the relevant module and document expected metrics in `README.md`.
- Backtests should report Sharpe, drawdown, and spread outputs before/after your change; attach diffs or summary tables in the PR.
- No automated CI exists yet—include manual verification steps and datasets used so reviewers can replay your run.

## Commit & Pull Request Guidelines
- Follow the existing history: short, imperative commit titles (`Add leverage support …` style) without trailing punctuation.
- Scope commits narrowly (one module or concern) and reference related docs or notebooks in the body if applicable.
- Pull requests must mention affected modules, expected numerical deltas, required datasets, and any new scripts or configs.
- Include screenshots for UI changes and sample console output for backtests or training runs so reviewers can confirm behaviour quickly.

## Security & Configuration Tips
- Never commit API keys or exchange secrets; load credentials from environment variables or `.env` files excluded via `.gitignore`.
- Verify file permissions before adding new cache directories; large data artifacts belong outside the repo under the shared `/home/craz/crypto/crypto-data/` hierarchy.
- When adding notifications (e.g., DingTalk), route through helper functions in `neural-strategy/utils` to keep tokens centralized.
