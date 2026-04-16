# Gallery

**Event-driven image annotation and retrieval** — a small system where services talk over pub-sub (Redis), store annotations in a document DB, and search with a vector index. Built for EC530 (see slides below).

**Intro slides:** `/Users/almailam/Downloads/Event-Driven Image Annotation and Retrieval System.pdf`

## What’s here

- **Broker** — Redis pub-sub for events between services  
- **Services** — image handling, document storage, vector search, interactive CLI  
- **Tests** — `pytest` under `tests/`

## Run

1. Install: `pip install -r requirements.txt`  
2. Run: `python main.py`  
3. In the CLI: `upload <path>` or `search <query>`, then `quit`.

Notes:
- If Redis is not already running, the program will try to start `redis-server` automatically (if it’s installed).

## Dev / tests

```bash
pip install -r requirements-dev.txt
pytest
```
