# Gallery

**Event-driven image annotation and retrieval** — a small system where services communicate over Redis pub-sub, persist image metadata and annotations to MongoDB, and provide keyword-based search.

## Features

- **Broker** — Redis pub-sub for events between services
- **Services** — upload intake, document metadata persistence, annotation + search indexing, interactive CLI
- **Storage** — MongoDB collections for image metadata and annotation/search records
- **Tests** — `pytest` test suite under `tests/`

## Prerequisites

- Python 3.10+ recommended
- Redis installed and available as `redis-server`
- MongoDB running locally, or a MongoDB connection string

## Quick start

1. Install dependencies: `pip install -r requirements.txt`
2. Start MongoDB, or set `MONGO_URI` to your MongoDB connection string.
3. Run the app: `python main.py`
4. In the CLI, try commands like:
   - `upload samples/bananas.jpg`
   - `search bananas`
   - `quit`

Notes:

- If Redis is not already running, the app will try to start `redis-server` automatically (if available in your environment).
- Mongo defaults to `MONGO_URI=mongodb://localhost:27017` and `MONGO_DB=gallery`.
- Metadata is stored in the `documents` collection, and annotation/search records are stored in the `vectors` collection.

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

## Event generator (simple simulation)

Use the standalone generator to trigger specific event types and inspect both sent and observed messages.

1. Start the app in one terminal: `python main.py`
2. In a second terminal, run for example:

```bash
python event_generator.py --events upload,search,list --count 2 --query fruit
```

Useful options:

- `--events upload,search,list` choose which events to trigger and order.
- `--count 3` repeat that event sequence N times.
- `--delay 0.2` pause between each trigger.
- `--upload-paths samples/bananas.jpg samples/apple.jpeg` rotate upload paths.
- `--observe-seconds 2` keep listening briefly for resulting messages.

## Development

```bash
pip install -r requirements-dev.txt
pytest
```

## GitHub automation

- The workflow at `.github/workflows/copilot-auto-review.yml` runs on pushes to `main`.
- It discovers PRs associated with pushed commits and requests a Copilot reviewer for open, non-draft PRs.

