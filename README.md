# BoilerCheck

A RAG (Retrieval-Augmented Generation) app for answering questions about Purdue University policy. Ask a question in plain English and get a grounded answer with links back to the exact policy sections it came from.

## How it works

```
User question
     │
     ▼
[Next.js frontend]
     │  POST /ask  { query }
     ▼
[FastAPI backend]
     │
     ├─ 1. Embed query          HuggingFace all-MiniLM-L6-v2 (384-dim)
     │
    ├─ 2. Vector search        Pinecone → top candidate text/image chunks
     │
     ├─ 3. Rerank               Cross-encoder ms-marco-MiniLM-L-6-v2 → top 4
     │
     ├─ 4. Generate answer      Gemini 2.0 Flash with source-grounded prompt
     │
     └─ 5. Return { answer, documents[] } → rendered in UI with source cards
```

Policy records are read from Firestore, converted into text and image-description chunks, and indexed into Pinecone with `ingest.py`. The live query path never touches Firestore directly — everything comes from the vector index.

## Project structure

```
BoilerCheck/
├── backend/
│   ├── main.py          FastAPI server — exposes POST /ask
│   ├── rag.py           Full RAG pipeline (embed → retrieve → rerank → generate)
│   ├── ingest.py        One-time script to embed & upsert data into Pinecone
│   ├── requirements.txt Python dependencies
│   └── .env             API keys (not committed)
├── data/
│   └── rag_mock_data.json  Legacy mock data (no longer used by default ingest path)
├── src/app/
│   ├── page.js          Main UI — search input, answer panel, source cards
│   ├── layout.js        Root layout
│   └── globals.css      Global styles
└── public/              Static assets
```

## Prerequisites

- Python 3.11+
- Node.js 18+
- A [Pinecone](https://app.pinecone.io) account (free Starter tier works)
- A [Gemini API key](https://aistudio.google.com/app/apikey) (free tier works)
- A Firebase Admin SDK service-account JSON key for your Firestore project

## Setup

### 1. Clone and install frontend dependencies

```powershell
cd BoilerCheck
npm install
```

### 2. Set up the Python backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Configure environment variables

Edit `backend/.env` with your real keys:

```env
GEMINI_API_KEY=your_gemini_api_key_here

PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=your_pinecone_index_name_here

# Optional retrieval tuning
IMAGE_SCORE_THRESHOLD=0.35
IMAGE_TOP_K=4
RAG_CANDIDATE_K=16

# Optional (defaults to policies_with_images)
POLICIES_COLLECTION=policies_with_images

# Optional if key file is not stored in backend/
FIREBASE_SERVICE_ACCOUNT_PATH=C:\\path\\to\\firebase-adminsdk.json
```

Create your Pinecone index with these settings:
- **Dimensions:** `384`
- **Metric:** `cosine`
- **Type:** Serverless (AWS us-east-1)

### 4. Ingest policy data into Pinecone

`backend/ingest.py` reads records from Firestore collection `policies_with_images`
(or `POLICIES_COLLECTION` if set), then uploads text + image chunks.

Download a Firebase Admin SDK JSON key (Firebase Console → Project settings →
Service accounts → Generate new private key), then either:

1. Place it in `backend/` with a filename containing `firebase-adminsdk`, or
2. Set `FIREBASE_SERVICE_ACCOUNT_PATH` in your shell.

Run this once to embed and upload all chunks:

```powershell
# from backend/ with .venv active
$env:FIREBASE_SERVICE_ACCOUNT_PATH="$PWD\your-firebase-adminsdk.json"
python ingest.py
```

This only needs to be re-run when Firestore policy records change.

### 5. Run the app

Open two terminals:

```powershell
# Terminal 1 — backend (from backend/ with .venv active)
python main.py

# Terminal 2 — frontend (from project root)
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Firestore Record Schema (ingest)

Each Firestore document in `policies_with_images` should follow this shape:

```json
{
  "document_id": "unique_id",
  "title": "Optional document title",
  "domain": "purdue.edu",
  "url": "Optional canonical page URL",
  "effective_date": "YYYY-MM-DD",
  "has_structure": true,
  "images": [
    {
      "description": "Text used for image retrieval",
      "source_url": "https://...",
      "filename": "...",
      "format": "svg",
      "image_type": "...",
      "md5": "...",
      "width": 0,
      "height": 0
    }
  ],
  "sections": [
    {
      "section_title": "Section Name",
      "text": "The policy text that gets embedded and retrieved."
    }
  ]
}
```

Image descriptions and section text are indexed as separate entries. At query
time, images are only returned if their rerank similarity score meets
`IMAGE_SCORE_THRESHOLD`.
