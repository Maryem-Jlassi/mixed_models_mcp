# CV Analyzer (FastAPI + React)

Full‑stack app to analyze CVs, search job postings, and compare CVs with jobs.


## Overview

- Backend (`backend/`): FastAPI API. Main app: `backend/main.py`. Orchestrator: `backend/orchestrator.py` with `UnifiedLLMHost`.
  - Uses Groq LLM (requires `GROQ_API_KEY`).
  - Starts MCP clients on demand:
    - CV OCR: `backend/cv_ocr_server.py`
    - Job Search: `backend/job_mcp_server.py`
  - Stores uploads in `backend/uploads/` and chats in `backend/conversations/` (JSON).
- Frontend (`frontend/`): React app (react-scripts). Main page: `src/pages/ChatPage.jsx` (renders `components/Main/Main.jsx` + `Sidebar`).
  - Dev server on port 3001. Proxy to API `http://localhost:8000`.


## Key Workflows

- CV analysis: upload PDF/DOCX/TXT → extract text → analysis & suggestions.
- Job search: give company name(s) or job URL(s) → fetch → convert to markdown → extract jobs.
- Comparison: use processed CV + company/URL to generate matches.

All coordinated by `UnifiedLLMHost.process_unified_request()` and streamed back as NDJSON.


## Prerequisites

- Windows, macOS, or Linux
- Python 3.10+
- Node.js 18+
- Groq API key


## Backend – Setup & Run

1) Create `backend/.env` (supported via `pydantic-settings` in `config.py`):
```
GROQ_API_KEY=your_groq_api_key
# Optional overrides
# LOG_LEVEL=INFO
# GROQ_MODEL_NAME=llama3-70b-8192
# PORT=8000
```

2) Install deps (example minimal set):
```
python -m venv venv
venv\Scripts\activate  # Windows
pip install fastapi "uvicorn[standard]" httpx pydantic pydantic-settings python-multipart psutil
```

3) Start API:
```
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Notes:
- Without a valid `GROQ_API_KEY`, unified host init will fail and `/workflow` returns 503.
- Upload dir is `uploads/` (configurable in `config.py`).


## Frontend – Setup & Run

From `frontend/`:
```
npm install
npm start
```
- Runs at http://localhost:3001
- Proxies API to http://localhost:8000 (see `package.json` → `proxy`).


## API (selected)

- `POST /upload-cv` (multipart)
  - file: Upload CV (PDF/DOCX/TXT). Returns `session_id`, stored in `conversations/`.
- `POST /workflow` (multipart or form)
  - fields: `workflow_type`, optional `cv_file`, `session_id`, `message`/`user_query`, `company_name`, `job_url`.
  - Streams NDJSON: `{type: status|intent|response|job|result|result_summary|error, ...}`.
- `GET /session/{session_id}` / `DELETE /session/{session_id}`
- `GET /sessions` (recent chats summary)
- `GET /chat/{session_id}` (full history)
- `GET /health`
- `GET /` (root info)

See `backend/main.py` for exact models and streaming details.


## Typical Flows (UI)

- Start chat without CV → send message; upload later if needed.
- Upload CV → receive analysis.
- Enter company name(s) or paste job page URL → receive jobs.
- Use comparison to match CV to extracted jobs.


## Configuration

Edit `backend/config.py` (`Settings`) or override via `backend/.env`:
- CORS origins, upload dir, max size, Groq model, timeouts, cleanup delay, etc.


## Testing

- Sample CV in `tests/data/cv1.pdf`.
- You can curl workflow quickly:
```
curl -N -X POST http://localhost:8000/workflow \
  -F workflow_type=cv_analysis \
  -F message="Analyze my CV" \
  -F session_id="$(powershell -NoP -C [guid]::NewGuid().ToString())"
```


## Notes & Troubleshooting

- 503 on `/workflow`: Unified host not initialized (likely missing/invalid `GROQ_API_KEY`).
- Large files: limited by `MAX_UPLOAD_SIZE` (see `config.py`).
- Streams are NDJSON; keep the connection open and parse per line.
- Persistent chats live in `backend/conversations/`.
