from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from rag import query as rag_query

app = FastAPI(title="BoilerCheck API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


class AskRequest(BaseModel):
    query: str


@app.post("/ask")
def ask(body: AskRequest):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    return rag_query(body.query)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
