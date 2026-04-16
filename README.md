# Gallery

**Event-driven image annotation and retrieval** — a small system where services talk over pub-sub (Redis), store annotations in a document DB, and search with a vector index.

## Architecture

- **Broker** — Redis pub-sub for events between services  
- **Services** — image handling, document storage, vector search, interactive CLI  
- **Tests** — `pytest` under `tests/`

## Run

1. Start **Redis** locally.  
2. Install: `pip install -r requirements.txt`  
3. Run: `python main.py`  
4. In the CLI: `upload <path>` or `search <query>`, then `quit`.

## Dev / tests

```bash
pip install -r requirements-dev.txt
pytest
```
