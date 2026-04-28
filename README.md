# Gallery

**Event-driven image annotation and retrieval** — a small system where services communicate over Redis pub-sub, persist image metadata and annotations to MongoDB, and provide keyword-based search.

## Repository Layout

- `main.py` application entrypoint; wires broker + services
- `gallery/broker/` Redis pub/sub broker and channel definitions
- `gallery/services/` image intake, metadata storage, annotation/search, interactive CLI
- `gallery/messages.py` typed event payload contracts
- MongoDB collections store image metadata and annotation/search records
- `tests/` unit and integration tests

## Requirements

- Python 3.10+ recommended
- Redis installed and available as `redis-server`
- MongoDB running locally, or a MongoDB connection string

## Run Locally

1. Install dependencies: `pip install -r requirements.txt`
2. Start MongoDB, or set `MONGO_URI` to your MongoDB connection string. If MongoDB is unavailable, the app falls back to local JSON storage in `gallery_data/`.
3. Run the app: `python main.py`

Notes:

- If Redis is not already running, the app will try to start `redis-server` automatically (if available in your environment).
- Interactive commands and command results stay in the main terminal. Redis/debug messages stream in a separate macOS Terminal window and are also written to `gallery_data/debug.log`.
- Set `GALLERY_OPEN_DEBUG_TERMINAL=0` to skip opening the second terminal, or `GALLERY_DEBUG_LOG=/path/to/debug.log` to choose a different log file.
- Mongo defaults to `MONGO_URI=mongodb://localhost:27017` and `MONGO_DB=gallery`. Set `MONGO_TIMEOUT_MS` to adjust how long startup waits before falling back to JSON storage.
- Metadata is stored in the `documents` collection, and annotation/search records are stored in the `vectors` collection. In JSON fallback mode, those are `gallery_data/documents.json` and `gallery_data/vector_db.json`.

```bash
pip install -r requirements.txt
python main.py
```

## CLI Commands

When the prompt opens, the available commands are:

- `upload <path>`: queue an image for processing
- `search <query>`: search indexed images by terms
- `list`: show known images and annotations
- `quit` / `exit` / `q`: leave the CLI

Example:

```text
upload samples/bananas.jpg
upload samples/apple.jpeg
search fruit
list
quit
```

## MongoDB Storage

- `documents` collection: image metadata
- `vectors` collection: annotations and search text

## Event Generator

Run synthetic event traffic from another terminal while the app is running:

```bash
python event_generator.py --events upload,search,list --count 2 --query fruit
```

## Development

```bash
pip install -r requirements-dev.txt
pytest
```

