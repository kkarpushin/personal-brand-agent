# LaoZhang.ai API Integration Skill

> **Last Updated:** January 2026
> **Documentation Source:** https://docs.laozhang.ai/en

## Overview

LaoZhang.ai is an enterprise-grade AI model aggregation service providing unified access to 200+ AI models (OpenAI, Anthropic Claude, Google Gemini, DeepSeek, etc.) through a single OpenAI-compatible API.

**Key Benefits:**
- Single API key for all providers
- 99.9% uptime SLA with auto-failover
- ~20-25% cost savings vs official APIs
- Unified billing

## Base URLs

| Purpose | URL |
|---------|-----|
| **Primary (recommended)** | `https://api.laozhang.ai/v1` |
| **Backup (direct overseas)** | `https://api-vip.laozhang.ai/v1` |
| **Console** | `https://api.laozhang.ai` |
| **API Key Management** | `https://api.laozhang.ai/token` |

## Authentication

```
Authorization: Bearer YOUR_API_KEY
```

API keys start with `sk-` prefix.

---

## Claude Models Integration

### Base URL
```
https://api.laozhang.ai/v1
```
**⚠️ IMPORTANT:** Use WITH `/v1` suffix for Claude!

### SDK Options

**Option 1: Anthropic SDK (Recommended for Claude-specific features)**
```python
import anthropic

client = anthropic.Anthropic(
    api_key="YOUR_LAOZHANG_API_KEY",
    base_url="https://api.laozhang.ai/v1"
)

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(message.content[0].text)
```

**Option 2: OpenAI SDK (For unified codebase)**
```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_LAOZHANG_API_KEY",
    base_url="https://api.laozhang.ai/v1"
)

response = client.chat.completions.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Available Claude Models

| Model | Model ID | Context | Pricing (Input/Output per 1M) |
|-------|----------|---------|-------------------------------|
| **Claude Opus 4.5** | `claude-opus-4-5-20251101` | 200K | $15 / $75 |
| Claude Opus 4.5 Thinking | `claude-opus-4-5-20251101-thinking` | 200K | $15 / $75 |
| **Claude Sonnet 4.5** | `claude-sonnet-4-5-20250929` | 200K | $3 / $15 |
| Claude Sonnet 4.5 Thinking | `claude-sonnet-4-5-20250929-thinking` | 200K | $3 / $15 |
| **Claude Sonnet 4** | `claude-sonnet-4-20250514` | 200K | $3 / $15 |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | 200K | $1 / $5 |
| Claude 3.5 Sonnet | `claude-3-5-sonnet-20241022` | 200K | $3 / $15 |
| Claude 3.5 Haiku | `claude-3-5-haiku-20241022` | 200K | $1 / $5 |
| Claude 3 Haiku | `claude-3-haiku-20240307` | 200K | $0.25 / $1.25 |

### Claude Tool Use (Structured Output)

```python
import anthropic

client = anthropic.Anthropic(
    api_key="YOUR_LAOZHANG_API_KEY",
    base_url="https://api.laozhang.ai/v1"
)

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1000,
    tools=[{
        "name": "analyze_lead",
        "description": "Analyze a sales lead",
        "input_schema": {
            "type": "object",
            "properties": {
                "is_qualified": {"type": "boolean"},
                "confidence": {"type": "number"},
                "requirements": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["is_qualified", "confidence", "requirements"]
        }
    }],
    tool_choice={"type": "tool", "name": "analyze_lead"},
    messages=[{"role": "user", "content": "Analyze: John needs a CRM"}]
)

# Result in message.content[0].input (dict)
result = message.content[0].input
```

---

## Gemini Models Integration

### Base URL
```
https://api.laozhang.ai/v1
```

### SDK: OpenAI SDK Only
```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_LAOZHANG_API_KEY",
    base_url="https://api.laozhang.ai/v1"
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)
```

### Available Gemini Models

| Model | Model ID | Context | Pricing (Input/Output per 1M) |
|-------|----------|---------|-------------------------------|
| **Gemini 3 Pro** | `gemini-3-pro-preview` | 2M | TBD |
| Gemini 3 Pro Thinking | `gemini-3-pro-preview-thinking` | 2M | TBD |
| Gemini 3 Flash | `gemini-3-flash-preview` | 1M | TBD |
| **Gemini 2.5 Pro** | `gemini-2.5-pro` | 2M | $1.25-$2.5 / $10 |
| **Gemini 2.5 Flash** | `gemini-2.5-flash` | 1M | $0.15 / $0.60 |
| Gemini 1.5 Pro | `gemini-1.5-pro-latest` | 2M | $1.25 / $5 |
| Gemini 1.5 Flash | `gemini-1.5-flash` | 1M | $0.075 / $0.30 |

### Gemini Structured Output (json_schema)

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_LAOZHANG_API_KEY",
    base_url="https://api.laozhang.ai/v1"
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "Analyze: John is CEO of TechCorp"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person_analysis",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "company": {"type": "string"}
                },
                "required": ["name", "role"]
            }
        }
    }
)

import json
result = json.loads(response.choices[0].message.content)
```

---

## OpenAI Models Integration

### Available Models

| Model | Model ID | Context | Pricing (Input/Output per 1M) |
|-------|----------|---------|-------------------------------|
| **GPT-5.2** | `gpt-5.2` | 128K | ~$1.25 / $10 |
| GPT-5.1 | `gpt-5.1` | 128K | Similar |
| GPT-5 | `gpt-5` | 128K | Similar |
| **GPT-4.1** | `gpt-4.1` | 128K | $2.50 / $10 |
| GPT-4.1 Mini | `gpt-4.1-mini` | 128K | $0.15 / $0.60 |
| GPT-4o | `gpt-4o` | 128K | $2.50 / $10 |
| GPT-4o Mini | `gpt-4o-mini` | 128K | $0.15 / $0.60 |
| **O3** | `o3` | 200K | $3 / $12 |
| O3 Pro | `o3-pro` | 200K | Higher |
| O4 Mini | `o4-mini` | 200K | $3 / $12 |

### O-Series Restrictions
- **No streaming** - `stream: true` will not work
- **No system role** - Cannot use system messages
- **No temperature/top_p** - These are ignored
- **max_tokens defaults to model maximum**

---

## Image Generation

### Models & Pricing

| Model | Model ID | Price per Image |
|-------|----------|-----------------|
| GPT-4o Image | `gpt-4o-image` | $0.01 |
| Sora Image | `sora-image` | $0.01 |
| DALL-E 2 | `dall-e-2` | $0.02 |
| Nano Banana | `nano-banana` | $0.025 |
| FLUX Pro | `black-forest-labs/flux-pro-v1.1` | $0.035 |
| DALL-E 3 | `dall-e-3` | $0.04 |
| Flux Kontext Pro | `flux-kontext-pro` | $0.04 |
| Flux Kontext Max | `black-forest-labs/flux-kontext-max` | $0.07 |

### Code Example

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_LAOZHANG_API_KEY",
    base_url="https://api.laozhang.ai/v1"
)

response = client.images.generate(
    model="gpt-4o-image",
    prompt="A beautiful sunset over mountains",
    size="1024x1024",
    n=1
)

print(response.data[0].url)
```

**⚠️ FLUX URLs expire in 10 minutes** - download immediately!

---

## Rate Limits

| Limit Type | Value |
|------------|-------|
| Requests per minute (RPM) | 3,000 |
| Tokens per minute (TPM) | 1,000,000 |
| Concurrent requests | 100 |

---

## Error Codes

| HTTP Code | Error | Solution |
|-----------|-------|----------|
| 400 | `invalid_request_error` | Check parameters |
| 401 | `invalid_api_key` | Verify API key |
| 404 | `model_not_found` | Check model ID |
| 429 | `rate_limit_exceeded` | Reduce frequency |
| 429 | `insufficient_quota` | Top up balance |
| 500 | `server_error` | Retry with backoff |

---

## Balance Query API

```python
import requests

response = requests.get(
    "https://api.laozhang.ai/api/user/self",
    headers={"Authorization": "YOUR_ACCESS_TOKEN"}
)
data = response.json()

# Quota conversion: 500,000 quota = $1.00 USD
balance_usd = data["data"]["quota"] / 500000
print(f"Balance: ${balance_usd:.2f}")
```

---

## Streaming

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_LAOZHANG_API_KEY",
    base_url="https://api.laozhang.ai/v1"
)

stream = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## Async Support

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key="YOUR_LAOZHANG_API_KEY",
    base_url="https://api.laozhang.ai/v1"
)

async def generate():
    response = await client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    return response.choices[0].message.content

result = asyncio.run(generate())
```

---

## Vision (Multimodal)

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_LAOZHANG_API_KEY",
    base_url="https://api.laozhang.ai/v1"
)

response = client.chat.completions.create(
    model="gpt-4o",  # or gemini-2.5-flash
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }]
)
```

---

## LangChain Integration

```python
import os
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_BASE"] = "https://api.laozhang.ai/v1"
os.environ["OPENAI_API_KEY"] = "YOUR_LAOZHANG_API_KEY"

llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.7,
    max_tokens=2000
)

response = llm.invoke("Hello!")
print(response.content)
```

---

## Best Practices

### 1. Error Handling with Retry
```python
import time
from openai import OpenAI, RateLimitError

def chat_with_retry(client, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages
            )
        except RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
```

### 2. Environment Variables
```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["LAOZHANG_API_KEY"],
    base_url="https://api.laozhang.ai/v1"
)
```

### 3. Model Selection by Use Case

| Use Case | Recommended Model | Cost |
|----------|-------------------|------|
| Daily tasks | `gpt-4.1-mini`, `gemini-2.5-flash` | Low |
| Complex reasoning | `claude-sonnet-4-20250514`, `o3` | Medium |
| Creative writing | `claude-opus-4-5-20251101` | High |
| Code generation | `gpt-4.1`, `claude-sonnet-4-20250514` | Medium |
| Multimodal | `gpt-4o`, `gemini-2.5-flash` | Medium |

---

## What's NOT Supported

| Feature | Status |
|---------|--------|
| Anthropic Prompt Caching | ❌ Not working (cache_control accepted but no caching) |
| OpenAI Batch API | ❌ Not supported |
| Fine-tuning | ❌ Not supported |
| File management | ❌ Not supported |
| Perplexity models | ❌ Not available |

---

## Changelog Highlights (2025-2026)

| Date | Update |
|------|--------|
| Jan 2026 | Claude 3 Opus retired, GPT-5.2 released |
| Nov 2025 | Claude Opus 4.5 launched |
| Sep 2025 | Claude Sonnet 4.5 released |
| Aug 2025 | GPT-5 full series released |
| Jul 2025 | All Claude models 20% price reduction |
| Jun 2025 | O3 model 80% price reduction |

---

## Support

- **Email:** threezhang.cn@gmail.com
- **Telegram:** @laozhang_cn
- **WeChat:** Kikivivikids
- **Status:** https://status.laozhang.ai
