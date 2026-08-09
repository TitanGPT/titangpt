# TitanGPT Python SDK

[![PyPI version](https://img.shields.io/pypi/v/titangpt.svg)](https://pypi.org/project/titangpt/)
[![Python versions](https://img.shields.io/pypi/pyversions/titangpt.svg)](https://pypi.org/project/titangpt/)
[![License](https://img.shields.io/pypi/l/titangpt.svg)](https://github.com/TitanGPT/titangpt/blob/main/LICENSE)

The official sync and async Python SDK for the OpenAI-compatible
[TitanGPT API](https://platform.titangpt.xyz/).

- Python 3.8+
- Synchronous and asynchronous clients
- Responses accessible as dictionaries or through attributes
- Typed API exceptions with request IDs and OpenAI-compatible error metadata
- Automatic retries for retryable idempotent requests

## Installation

```bash
python -m pip install --upgrade titangpt
```

Set your API key in the environment:

```bash
export TITANGPT_API_KEY="YOUR_API_KEY"
```

PowerShell:

```powershell
$env:TITANGPT_API_KEY = "YOUR_API_KEY"
```

You can also pass `api_key` directly to either client.

## Quick start

### Synchronous client

```python
from titangpt import TitanGPT


with TitanGPT() as client:
    response = client.chat.completions.create(
        model="gpt-5.6-terra",
        messages=[
            {"role": "user", "content": "Write a haiku about Python."},
        ],
    )

print(response.choices[0].message.content)
```

`TitanResponse` behaves like a normal dictionary, so the same value is also
available as `response["choices"][0]["message"]["content"]`.

### Asynchronous client

```python
import asyncio

from titangpt import AsyncTitanGPT


async def main() -> None:
    async with AsyncTitanGPT() as client:
        response = await client.chat.completions.create(
            model="gpt-5.6-terra",
            messages=[{"role": "user", "content": "Explain async I/O briefly."}],
        )
        print(response.choices[0].message.content)


asyncio.run(main())
```

## Configuration

```python
from titangpt import TitanGPT


client = TitanGPT(
    api_key="YOUR_API_KEY",             # or TITANGPT_API_KEY
    base_url="https://api.titangpt.xyz",
    timeout=60,
    max_retries=3,
    user_id="12345",
)
```

`max_retries` applies to retryable `GET`, `HEAD`, and `OPTIONS` requests. The
client retries HTTP 429, 500, 502, 503, and 504 responses with bounded
exponential backoff.

## Error handling

HTTP and transport failures are exposed as typed exceptions:

```python
from titangpt import (
    AuthenticationError,
    ModelNotFoundError,
    RateLimitError,
    TitanGPT,
    TitanGPTException,
)


try:
    with TitanGPT() as client:
        client.chat.completions.create(
            model="unknown-model",
            messages=[{"role": "user", "content": "Hello"}],
        )
except ModelNotFoundError as exc:
    print(exc.message)
    print(exc.code)        # model_not_found
    print(exc.request_id)  # include this when contacting support
except (AuthenticationError, RateLimitError) as exc:
    print(exc.status_code, exc.message)
except TitanGPTException as exc:
    print(f"TitanGPT request failed: {exc}")
```

API exceptions expose these diagnostic attributes:

| Attribute | Description |
| --- | --- |
| `message` | Human-readable API error message |
| `status_code` | HTTP status code, when available |
| `error_type` | OpenAI-compatible error type |
| `param` | Request parameter associated with the error |
| `code` | Stable API error code, such as `model_not_found` |
| `request_id` | Server request ID for diagnostics and support |
| `response_body` | Parsed response body or raw response text |

Available exception classes include `AuthenticationError`,
`AuthorizationError`, `ValidationError`, `RateLimitError`,
`ModelNotFoundError`, `NotFoundError`, `TimeoutError`, `ConnectionError`,
`DataError`, and `APIError`. All inherit from `TitanGPTException`.

## API areas

The SDK currently provides:

- `client.chat.completions`
- `client.models.list()`
- `client.files`
- `client.threads`
- `client.usage`
- `client.audio.transcriptions`
- `client.music.youtube`
- `client.music.yandex`

The async client exposes the same areas with awaitable methods.

## Development

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
python -m build
python -m twine check --strict dist/*
```

See the [changelog](https://github.com/TitanGPT/titangpt/blob/main/CHANGELOG.md)
for release notes.

## Links

- [Website](https://titangpt.xyz/)
- [Documentation](https://platform.titangpt.xyz/)
- [API status](https://status.titangpt.xyz/)
- [Issue tracker](https://github.com/TitanGPT/titangpt/issues)
- [Telegram](https://t.me/titangpt_channel)

## License

MIT
