# MCP Agent for Job Offer Extraction and CV Matching

A full-stack application for CV analysis, job matching, and intelligent job search using LLMs.

## Overview

- **Backend** (`backend/`): FastAPI-based REST API with multiple LLM backends
  - Supports both Ollama and Groq LLM backends
  - Handles CV processing, job extraction, and matching workflows
  - MongoDB for session and chat history storage
  - Main entry point: `backend/app.py`

- **Frontend** (`frontend/`): Modern React application
  - Built with Create React App
  - Responsive UI with chat interface
  - Real-time streaming responses
  - Main components in `src/components/` and pages in `src/pages/`

## Key Features

- **CV Analysis**: Upload and analyze CVs in various formats
- **Job Extraction**: Extract job postings from URLs
- **Smart Matching**: Intelligent CV-job matching
- **Multi-Model Support**: Switch between Ollama and Groq LLM backends
- **Session Management**: Save and resume conversations
- **Streaming Responses**: Real-time interaction with LLMs

## Prerequisites

- Python 3.10+
- Node.js 18+
- MongoDB
- Ollama (for local LLM) or Groq API key (for cloud LLM)

## Backend Setup

1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate  # Windows

pip install -r backend/requirements.txt
```

2. Configure environment variables in `backend/.env`:
```
# Required for Groq
GROQ_API_KEY=your_groq_api_key

# MongoDB settings
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=jobmatcher

# Optional overrides
# LOG_LEVEL=INFO
# GROQ_MODEL=llama-3.1-8b-instant
# OLLAMA_MODEL=llama3
# PORT=8001
```

3. Start the backend server:
```bash
uvicorn backend.app:app --reload --port 8001
```

## Frontend Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm start
```

The frontend will be available at `http://localhost:3000`

## API Endpoints

- `GET /` - API root with available endpoints
- `POST /workflow` - Main workflow endpoint (defaults to Ollama)
- `POST /groq/workflow` - Groq LLM workflow endpoint
- `GET /simple/health` - Health check for Ollama backend
- `GET /groq/health` - Health check for Groq backend
- `GET /api/sessions` - List chat sessions
- `GET /api/chat/{session_id}` - Get chat history for a session

## Workflow Parameters

- `message`: User query/instruction
- `job_url`: URL to extract job postings from
- `cv_file`: Uploaded CV file (PDF/DOCX/TXT)
- `model`: LLM to use ('simple' for Ollama, 'groq' for Groq)
- `session_id`: Optional session ID for chat history

## Environment Variables

### Backend
- `GROQ_API_KEY`: API key for Groq service
- `MONGODB_URL`: MongoDB connection string
- `DATABASE_NAME`: MongoDB database name
- `LOG_LEVEL`: Logging level (default: INFO)
- `PORT`: Port to run the server on (default: 8001)

## Architecture

The application follows a modular architecture:

- `app.py`: Main FastAPI application and endpoints
- `groq.py`: Groq LLM integration and workflow logic
- `simple.py`: Ollama LLM integration
- `auth.py`: Authentication and user management
- `db.py`: Database connection and utilities
- `cv_client.py`: CV processing client
- `job_client.py`: Job extraction client

Frontend components are organized by feature in the `src/components/` directory, with page layouts in `src/pages/`.
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
