# TokenRoute

An OpenAI-compatible gateway that cuts LLM inference cost by routing each request to the cheapest model that can handle it. Point an existing OpenAI client at TokenRoute instead of `api.openai.com` — no client code changes.

**Status:** Active development.

## Features

- Drop-in compatible with the OpenAI SDK via `/v1/chat/completions`
- Cost-aware routing that classifies request difficulty and picks the cheapest sufficient model
- Semantic caching for near-duplicate requests
- Context pruning to trim low-relevance input tokens before forwarding
- Per-request tracking of token counts, cost, latency, and failures
- Automatic failover to a fallback provider on error or timeout

## Architecture

Requests arrive at a FastAPI gateway implementing the OpenAI chat completions schema, so any OpenAI client works unchanged.

The semantic cache runs first. Incoming prompts are embedded and checked against Redis for a near-duplicate above a similarity threshold — a hit skips both classification and the provider call entirely. The threshold is deliberately conservative, since a false hit returns a wrong answer and a miss only costs money.

On a miss, context pruning drops low-relevance segments, then a small classifier scores request difficulty and maps it to the cheapest model tier that can handle it. The premise is that most production traffic — extraction, classification, formatting — is handled identically by models costing far less than frontier ones.

Provider adapters normalize the differences between OpenAI, Anthropic, and Google APIs behind one interface, which is also where failover lives. Every request writes a usage record to PostgreSQL: tokens, cost, latency, chosen model, cache status.

## Installation

```bash
git clone https://github.com/<you>/tokenroute.git
cd tokenroute

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # provider API keys
docker compose up -d      # Postgres + Redis

alembic upgrade head
uvicorn tokenroute.main:app --port 8000
```

## Usage

Point any OpenAI client at the gateway:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tr-local-key",
)

resp = client.chat.completions.create(
    model="auto",                      # let the router choose
    messages=[{"role": "user", "content": "Summarize this changelog..."}],
)
```

Pin a specific model to bypass routing:

```python
resp = client.chat.completions.create(model="gpt-4o", messages=[...])
```

Check usage:

```bash
curl localhost:8000/usage?window=24h
```

Routing config:

```yaml
# config/routing.yaml
tiers:
  cheap:    { model: gpt-4o-mini,  max_difficulty: 0.4 }
  standard: { model: claude-haiku, max_difficulty: 0.75 }
  frontier: { model: gpt-4o,       max_difficulty: 1.0 }

cache:
  similarity_threshold: 0.93
  ttl_seconds: 86400
```
