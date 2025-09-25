# RAG Starter Repo

A tidy version of your uploaded project with renamed modules, a `.env`, and a reproducible run script.

## Folder structure
```
.
├─ src/
│  ├─ app.py
│  ├─ llm_assistant.py
│  ├─ embedding_models.py
│  ├─ vector_store.py
│  └─ __init__.py
├─ configs/
│  └─ config.yaml
├─ models/                # put local model folders here if you have them
├─ .env.example
├─ requirements.txt
├─ config.yaml            # duplicate for compatibility
└─ README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set ROUTER_API_KEY, optionally MISTRAL_API_KEY
```

## Run

```bash
streamlit run src/app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Notes
- Put local SentenceTransformer models under `./models/<name>` if you want offline use, e.g. `./models/all-MiniLM-L6-v2`.
- The reranker in `vector_store.py` expects `./ms-marco-MiniLM-L6-v2` by default. Either place that folder under `./models` and update the path, or change it to a hub name like `cross-encoder/ms-marco-MiniLM-L6-v2`.
- Config is loaded from `APP_CONFIG_PATH` (defaults to `./configs/config.yaml`).
