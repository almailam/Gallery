# Gallery

**Event-driven image annotation and retrieval** — a small system where services communicate over Redis pub-sub, persist image metadata and annotations to local JSON stores, and provide keyword-based search.

## Features

- **Broker** — Redis pub-sub for events between services
- **Services** — upload intake, document metadata persistence, annotation + search indexing, interactive CLI
- **Storage** — JSON-backed local data files under `gallery_data/`
- **Tests** — `pytest` test suite under `tests/`

## Prerequisites

- Python 3.10+ recommended
- Redis installed and available as `redis-server`

## Quick start

1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `python main.py`
3. In the CLI, try commands like:
   - `upload samples/bananas.jpg`
   - `search bananas`
   - `quit`

Notes:

- If Redis is not already running, the app will try to start `redis-server` automatically (if available in your environment).
- Metadata and index records are written to `gallery_data/documents.json` and `gallery_data/vector_db.json`.

## Usage examples

Start the app:

```bash
python main.py
```

Example 1: upload one image, then search:

```text
upload samples/bananas.jpg
search bananas
```

Example 2: upload multiple images and run a few queries:

```text
upload samples/bananas.jpg
upload samples/apples.jpg
search fruit
search bananas
quit
```

## Development

```bash
pip install -r requirements-dev.txt
pytest
```

## GitHub automation

- The workflow at `.github/workflows/copilot-auto-review.yml` runs on pushes to `main`.
- It discovers PRs associated with pushed commits and requests a Copilot reviewer for open, non-draft PRs.

