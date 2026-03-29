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
     ├─ 2. Vector search        Pinecone → top 8 candidate chunks
     │
     ├─ 3. Rerank               Cross-encoder ms-marco-MiniLM-L-6-v2 → top 4
     │
     ├─ 4. Generate answer      Gemini 2.0 Flash with source-grounded prompt
     │
     └─ 5. Return { answer, documents[] } → rendered in UI with source cards
```

Policy documents are pre-chunked at the subsection level and indexed into Pinecone with `ingest.py`. The live query path never touches the raw JSON — everything comes from the vector index.

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
│   └── rag_mock_data.json  Source policy documents (Purdue housing & dining)
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
```

Create your Pinecone index with these settings:
- **Dimensions:** `384`
- **Metric:** `cosine`
- **Type:** Serverless (AWS us-east-1)

### 4. Ingest policy data into Pinecone

Run this once to embed and upload all policy chunks:

```powershell
# from backend/ with .venv active
python ingest.py
```

This only needs to be re-run if `data/rag_mock_data.json` changes.

### 5. Run the app

Open two terminals:

```powershell
# Terminal 1 — backend (from backend/ with .venv active)
python main.py

# Terminal 2 — frontend (from project root)
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Adding more policy data

Add new documents to `data/rag_mock_data.json` following the existing schema, then re-run `python ingest.py`. The structure is:

```json
{
  "document_id": "unique_id",
  "title": "Document Title",
  "domain": "housing",
  "url": "https://...",
  "effective_date": "YYYY-MM-DD",
  "sections": [
    {
      "section_title": "Section Name",
      "subsections": [
        {
          "section_title": "Subsection Name",
          "text": "The policy text that gets embedded and retrieved."
        }
      ]
    }
  ]
}
```
