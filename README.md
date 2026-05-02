# TitanGPT Python Client

[![PyPI version](https://badge.fury.io/py/titangpt.svg)](https://badge.fury.io/py/titangpt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official Python SDK for the TitanGPT API.

## Public API

The package includes:

- `chat.completions`
- `models.list`
- `files`
- `threads`
- `usage`
- `audio.transcriptions`
- `music.youtube`
- `music.yandex`
- sync and async clients

## Installation

```bash
pip install titangpt
```

Python 3.8+ is supported.

## Quick Start

### Sync Client

```python
from titangpt import TitanGPT

with TitanGPT(api_key="YOUR_API_KEY") as client:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum physics in one sentence."},
        ],
    )

    print(response.choices[0].message.content)
```

### Async Client

```python
import asyncio

from titangpt import AsyncTitanGPT


async def main():
    async with AsyncTitanGPT(api_key="YOUR_API_KEY") as client:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Write a haiku about Python."}],
        )
        print(response.choices[0].message.content)


asyncio.run(main())
```

## Configuration

```python
client = TitanGPT(
    api_key="YOUR_API_KEY",
    base_url="https://api.titangpt.xyz",
    timeout=60,
    max_retries=3,
    user_id="12345",
    product="api",
)
```

## Build

```bash
python -m build
python -m twine check dist/*
```

## Links

- Website: [titangpt.ru](https://titangpt.ru)
- Documentation: [platform.titangpt.ru](https://platform.titangpt.ru)
- API endpoint: `https://api.titangpt.xyz`
- Telegram: [@titangpt_channel](https://t.me/titangpt_channel)

## License

MIT
