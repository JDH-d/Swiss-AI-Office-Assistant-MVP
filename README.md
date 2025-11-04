# Swiss AI Office Assistant (MVP)

A lightweight Streamlit web app that helps employees find answers to internal HR/IT policy questions. It performs local retrieval over `.txt` documents (LangChain + Chroma) and uses the OpenAI API to formulate concise, professional answers. If nothing relevant is found, it politely redirects the user to contact HR.

## Features
- Streamlit chat interface with a Swiss corporate tone
- Local `.txt` documents indexed on first run and persisted to disk
- Embeddings via `text-embedding-3-small` with Chroma vector store
- Configurable OpenAI chat model (see Environment), sequential fallbacks included
- Graceful fallback: if relevance is low, provide HR contact

## Project Structure
```
.
├─ main.py            # Streamlit UI and chat flow
├─ retriver.py        # Document loading/splitting and Chroma vector store
├─ docs/              # Internal policy documents (local knowledge base)
├─ .chroma_store/     # Generated on first run (vector index cache)
├─ requirements.txt   # Pinned dependencies
├─ .env.example       # Example environment file
└─ README.md          # This file
```

## Prerequisites
- Python 3.10+
- An OpenAI API key with access to a chat model (e.g., `gpt-4o-mini`)

## Quickstart
1) Create and activate a virtual environment

   Windows (PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   macOS/Linux (bash):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) Configure environment
```bash
cp .env.example .env   # on Windows: copy .env.example .env
```
Edit `.env` and set your API key.

4) Run the app
```bash
streamlit run main.py
```
Open the URL shown in the terminal (usually http://localhost:8501).

## Environment
The app reads configuration from environment variables (via `.env`).

| Variable        | Required | Default         | Notes |
|-----------------|----------|-----------------|-------|
| `OPENAI_API_KEY`| Yes      | —               | OpenAI API key used for embeddings and chat. |
| `OPENAI_MODEL`  | No       | `gpt-5-nano`    | Optional override for the primary chat model. The code also tries `gpt-5-mini` and `gpt-4o-mini` as fallbacks. If your account does not have access to the `gpt-5-*` placeholders, set this explicitly to a model you have, e.g. `gpt-4o-mini`. |

Notes on models: the defaults include `gpt-5-nano`/`gpt-5-mini` as placeholders and then `gpt-4o-mini`. For reliable operation, set `OPENAI_MODEL` to a model available to your account.

## How It Works
- On startup, the app loads `.txt` files from `docs/`, splits them into ~500‑character chunks, and builds a Chroma vector store persisted to `.chroma_store/`.
- When you ask a question, the app retrieves the top 3 most relevant chunks and sends them with your question to the OpenAI chat model.
- If the highest relevance score is below a threshold, the app responds with a polite HR contact message.

## Example Questions
- How many vacation days do I have?
- What should I do if I'm sick?
- Who manages IT issues?

## Troubleshooting
- Missing API key: set `OPENAI_API_KEY` in `.env`.
- Model access: if calls fail because the model is unavailable, set `OPENAI_MODEL` to a model your account can use (e.g., `gpt-4o-mini`).
- Index not updating after you edit docs: delete the `.chroma_store/` folder and restart the app to rebuild the index.

## Privacy & Data Handling
- Documents are stored locally in `docs/`; the Chroma index is cached in `.chroma_store/`.
- When answering, the app sends the user’s question and retrieved excerpts to OpenAI. Do not include sensitive content unless your data handling policies allow it.
- For fully offline operation, consider replacing embeddings and LLM calls with local models (not included in this MVP).

## Roadmap (Ideas)
- Robust language detection and user‑selectable language
- Better retrieval (e.g., MMR, re‑ranking) and tunable thresholds
- PDF/DOCX ingestion and file upload via UI
- Streaming responses and richer error handling
- Automatic index refresh when `docs/` changes

## Contributing
Issues and pull requests are welcome. Please discuss significant changes via an issue first.

