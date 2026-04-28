# Gallery

Event-driven image annotation and retrieval using Redis pub/sub and local JSON storage.

## Repository Layout

- `main.py` application entrypoint; wires broker + services
- `gallery/broker/` Redis pub/sub broker and channel definitions
- `gallery/services/` image intake, metadata storage, annotation/search, interactive CLI
- `gallery/messages.py` typed event payload contracts
- `gallery_data/` persisted local data stores
- `tests/` unit and integration tests

## Requirements

- Python 3.10+
- Redis available locally as `redis-server`

## Run Locally

```bash
pip install -r requirements.txt
python main.py
```

The app tries to start `redis-server` automatically when Redis is not already running.

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

## Data Files

- `gallery_data/documents.json` image metadata
- `gallery_data/vector_db.json` annotations and search text

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

