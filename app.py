"""OpenAI-compatible caching proxy.

Point any OpenAI client at http://localhost:8000/v1 and identical or
semantically equivalent prompts are served from cache.

Run: uvicorn app:app --reload
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

from cache.store import SemanticCache

load_dotenv()

app = FastAPI(title="LLM Cache Gateway")
cache = SemanticCache()
client = OpenAI()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    prompt = request.messages[-1].content

    cached = cache.lookup(prompt)
    if cached is not None:
        return _as_completion(cached, request.model, from_cache=True)

    upstream = client.chat.completions.create(
        model=request.model,
        messages=[m.model_dump() for m in request.messages],
    )
    answer = upstream.choices[0].message.content
    cache.store(prompt, answer)
    return _as_completion(answer, request.model, from_cache=False)


@app.get("/stats")
def stats():
    return {
        "threshold": cache.threshold,
        "entity_check_enabled": cache.guard,
        "entries": len(cache.entries),
        **cache.stats.as_dict(),
    }


def _as_completion(content: str, model: str, from_cache: bool) -> dict:
    return {
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "cached": from_cache,
    }