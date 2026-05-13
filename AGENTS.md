# Repository Guidelines

## Project Structure & Module Organization
- `binance_implement/neural_strategy` is the current live/testnet automation entry point; `automated_trading.py` orchestrates data loading, neural factor calculation, execution, protective orders, and logs under `logs/`.
- `binance_implement/whole_strategy`, `only_in`, and `find_in_out` contain production variants and strategy experiments that reuse the executor, data processor, DingTalk, and account-stat helpers.
- Research and offline evaluation live in `figure_model` for OHLC image model training, `neural-strategy` for factor/backtest pipelines, `utils` for shared CSV and pickle loaders, and `tests` for script-style cache/API checks.
- Update the cache constants in `utils/data_loader.py` when relocating datasets; defaults point to `/home/craz/crypto/crypto-data`.

## Build, Test, and Development Commands
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python binance_implement/neural_strategy/automated_trading.py
python neural-strategy/ohlc_backtest.py
python tests/test_month_cache_loader.py
```
- Keep one terminal per long-running strategy so realtime logs remain readable.
- Retrain OHLC models only when needed via `python figure_model/train_ohlc_model.py`; it can regenerate large checkpoints.
- Tests expect populated local CSV/pickle caches and, for API checks, valid environment variables.

## Coding Style & Naming Conventions
- Stick to PEP 8, 4-space indentation, snake_case for functions, and CapWords for classes; add type hints to new public helpers when touching shared modules.
- Use `trading_utils.logger.setup_logger` in trading code instead of bare prints, and keep established log tags such as `[SYSTEM]`, `[ERROR]`, `[OK]`, `[PROTECT]`, and `[SCHEDULE]`.
- Centralize strategy knobs in `config.py`; do not embed secrets, API credentials, or host-specific paths inside strategy classes.

## Testing Guidelines
- Cache loader checks are executable scripts; run them as `python path/to/script.py` and inspect stdout.
- Before running anything under `tests/`, confirm `utils/data_loader.py` points to available data directories.
- Prioritize coverage around cache loading, factor calculation, execution safeguards, protective orders, and account-stat scripts. Document skips when they require unavailable exchange APIs.

## Commit & Pull Request Guidelines
- Use concise imperative subjects matching existing history, for example `Fix margin type setting...` or `Add IC and RankIC...`.
- Keep each commit scoped to one logical change and avoid mixing code edits with generated logs, reports, `.pyc`, or notebook output churn.
- In PRs, list functional changes, data/env prerequisites, and commands rerun so reviewers can reproduce outcomes.

## Security & Configuration Tips
- Inject Binance credentials through environment variables; placeholders in config files are intended to fail validation.
- Sanitize or rotate logs before sharing because execution logs can include balances, position sizes, and API responses.
- Keep data paths and host-specific values in config modules or data loader constants so strategy logic stays portable.
