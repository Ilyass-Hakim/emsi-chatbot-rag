"""Web API of EMSI ChatBot.

Endpoints:
    POST /chat      a question (and an optional photo) -> an answer with its sources
    GET  /health    liveness check
    GET  /images/*  images extracted from the documents
"""

from contextlib import asynccontextmanager

import ollama
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from retriever import clip_text_model, search, text_model

SYSTEM_PROMPT = """Tu es EMSI ChatBot, l'assistant officiel de l'EMSI \
(École Marocaine des Sciences de l'Ingénieur).
Réponds uniquement à partir des extraits de la brochure fournis dans le CONTEXTE.
Indique la page de chaque information, par exemple : (page 5).
Si la réponse ne se trouve pas dans le contexte, dis simplement que tu ne disposes pas \
de cette information.
Réponds dans la langue de la question, de façon claire et concise."""


class Source(BaseModel):
    page: int
    type: str
    image_url: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load the embedding models at startup so the first question is not slow
    text_model()
    clip_text_model()
    yield


app = FastAPI(title="EMSI ChatBot API", version="1.0.0", lifespan=lifespan)
app.add_middleware(  # allow the Flutter web app (another origin) to call the API
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=config.IMAGES_DIR), name="images")


def build_context(hits: list[dict]) -> str:
    """Format the retrieved records as numbered excerpts for the prompt."""
    return "\n\n".join(
        f"[{i}] (page {hit['page']}, {hit['type']}) {hit['content']}"
        for i, hit in enumerate(hits, start=1)
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(question: str = Form(...), photo: UploadFile | None = File(None)) -> ChatResponse:
    # 1. Retrieve the most relevant excerpts
    photo_bytes = photo.file.read() if photo else None
    hits = search(question, k=config.TOP_K, photo=photo_bytes)

    # 2. Build the prompt: context + question, plus the best images for the VLM to see
    message = {
        "role": "user",
        "content": f"CONTEXTE :\n{build_context(hits)}\n\nQUESTION : {question}",
    }
    images = [str(config.IMAGES_DIR / hit["image"]) for hit in hits if hit["type"] == "image"]
    images = images[:config.MAX_CONTEXT_IMAGES]
    if photo_bytes:
        images.append(photo_bytes)
    if images:
        message["images"] = images

    # 3. Generate the answer
    response = ollama.chat(
        model=config.VLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, message],
        think=False,
        options={"num_ctx": config.CONTEXT_WINDOW},
    )

    sources = [
        Source(
            page=hit["page"],
            type=hit["type"],
            image_url=f"/images/{hit['image']}" if hit["type"] == "image" else None,
        )
        for hit in hits
    ]
    return ChatResponse(answer=response.message.content.strip(), sources=sources)
