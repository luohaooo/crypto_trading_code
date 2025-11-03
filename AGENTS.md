# Repository Guidelines

## Project Structure & Module Organization
- `binance_implement/neural_strategy` is the live automation entry point; `automated_trading.py` orchestrates factor prep, execution, and writes rotating logs inside `logs/`.
- Research code lives in `figure_model` (training scripts, checkpoints), `neural-strategy` (factor and backtest pipeline), `visualization/` (Flask dashboard), `utils/` (shared loaders), and `tests/` (cache smoke scripts).
- Update the cache constants in `utils/data_loader.py:18` when relocating datasets; the defaults point to `/home/craz/crypto/crypto-data`.

## Build, Test, and Development Commands
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python binance_implement/neural_strategy/automated_trading.py
python tests/test_month_cache_loader.py
python neural-strategy/ohlc_backtest.py
```
- Keep one terminal per long-running module to inspect realtime logs; ignore `start.sh` and launch scripts from their respective directories.
- Retrain the OHLC baseline only as needed via `python figure_model/train_ohlc_model.py`, since it regenerates large checkpoints.

## Coding Style & Naming Conventions
- Stick to PEP 8, 4-space indentation, snake_case for functions, and CapWords for classes; add type hints to new public helpers to match existing signatures.
- Use `trading_utils.logger.setup_logger` instead of bare prints and prefix messages with the established tags (`[SYSTEM]`, `[ERROR]`, `[OK]`) for consistent monitoring.
- Centralize configuration in `config.py` files; avoid embedding secrets or file paths directly in strategy classes.

## Testing Guidelines
- Tests expect populated pickle caches and CSVs; confirm the directories in `utils/data_loader.py:18` before running anything under `tests/`.
- Current checks are executable scripts—run with `python path/to/test.py` and review stdout; gate heavy network calls behind flags or fixtures when adding pytest cases.
- Prioritize coverage for cache loaders, factor calculators, and execution safeguards, and document skipped scenarios when they rely on unavailable market APIs.

## Commit & Pull Request Guidelines
- Mirror the imperative, concise subjects seen in `git log` (e.g., `Fix margin type setting...`, `Add IC and RankIC...`) and keep each commit scoped to one logical change.
- In PRs, link issues, list functional changes, and attach evidence of the commands you reran (logs, screenshots, or metrics) so reviewers can trace outcomes.
- Call out data prerequisites and environment updates in the PR description, especially when touching trade execution or cache locations.

## Security & Configuration Tips
- Inject Binance credentials via environment variables; `binance_implement/neural_strategy/config.py:54` falls back to placeholders that intentionally fail validation.
- Sanitize or rotate log files before sharing them—execution logs can contain position sizes and API responses.
- Keep host and path tweaks inside config modules so secrets stay isolated from strategy logic.
