# TitanGPT Python Client

[![PyPI version](https://badge.fury.io/py/titangpt.svg)](https://badge.fury.io/py/titangpt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[🇬🇧 English Version](#english-version) | [🇷🇺 Русская версия](#русская-версия)**

---

<a name="english-version"></a>
## 🇬🇧 English Version

The official Python library for the **TitanGPT API**.
Designed to be fully compatible with the **OpenAI v1+** syntax, allowing for seamless integration into existing projects.

### Features

- 🚀 **Full Compatibility**: Follows the OpenAI syntax structure (`client.chat.completions.create`).
- 🌍 **UTF-8 Support**: Correctly handles API keys containing non-ASCII characters (e.g., Cyrillic) by automatically encoding headers.
- ⚡ **Sync & Async**: Includes both `TitanGPT` and `AsyncTitanGPT` clients.
- 🎨 **Image Generation**: Support for Flux style generation.
- 🎵 **Music Search**: Unique module for searching tracks via internal API.
- 🔧 **Flexibility**: Supports custom parameters like `verbosity`, `reasoning_effort` (for o-series models), etc.

### Installation

```bash
pip install titangpt
```

### Quick Start

#### Synchronous Client

```python
from titangpt import TitanGPT

# Initialize client (api_key is optional if TITANGPT_API_KEY env var is set)
client = TitanGPT(api_key="YOUR_API_KEY")

# 1. Chat Completion
response = client.chat.completions.create(
    model="gpt-5.2-2025-12-11",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum physics in 1 sentence."}
    ]
)

print(response.choices[0].message.content)

# 2. Image Generation
image = client.images.generate(
    prompt="A futuristic cyberpunk city, neon lights, rain",
    model="flux",
    size="1024x1024"
)

# Output base64 or URL (depending on server response)
print(image.data[0].b64_json[:50] + "...") 
```

#### Asynchronous Client

Ideal for high-load applications (FastAPI, Bots).

```python
import asyncio
from titangpt import AsyncTitanGPT

async def main():
    # Uses aiohttp under the hood
    client = AsyncTitanGPT(api_key="YOUR_API_KEY")

    response = await client.chat.completions.create(
        model="gpt-5.2-2025-12-11",
        messages=[{"role": "user", "content": "Write a haiku about Python."}]
    )
    print(response.choices[0].message.content)
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Usage

You can pass any arbitrary parameters (like `reasoning_effort` or custom dictionaries) using `**kwargs`.

```python
response = client.chat.completions.create(
    model="gpt-5.2-2025-12-11", 
    messages=[{"role": "user", "content": "Solve this logic puzzle..."}],
    
    # Specific parameter for reasoning models
    reasoning_effort="high",
    
    # Custom parameter accepted by TitanGPT backend
    text={
        "verbosity": "high"
    }
)
```

### Music & Audio

```python
# Music Search
track = client.music.search(query="Linkin Park Numb")
print(f"Title: {track.title}, URL: {track.url}")

# Audio Transcription (Whisper)
with open("speech.mp3", "rb") as f:
    transcription = client.audio.transcriptions.create(model="whisper-1", file=f)
    print(transcription.text)
```
---

## 🔗 Links & Support 

- **Website**: [titangpt.ru](https://titangpt.ru)
- **API Endpoint**: `https://api.titangpt.ru`
- **Telegram Channel**: [@titangpt_channel](https://t.me/titangpt_channel)

If you have any questions or need support, join our Telegram channel.

<a name="русская-версия"></a>
## 🇷🇺 Русская версия

Официальная Python библиотека для работы с API **TitanGPT**.
Библиотека спроектирована так, чтобы быть максимально совместимой с синтаксисом **OpenAI (v1+)**, что позволяет легко использовать её в существующих проектах, просто меняя `import`.

### Особенности

- 🚀 **Полная совместимость**: Синтаксис идентичен OpenAI (`client.chat.completions.create`).
- 🇷🇺 **Поддержка UTF-8**: Исправлена проблема с кодировкой заголовков. API-ключи с кириллицей или спецсимволами работают корректно.
- ⚡ **Синхронный и Асинхронный** клиенты (`TitanGPT` и `AsyncTitanGPT`).
- 🎨 **Генерация изображений**: Поддержка Flux style.
- 🎵 **Поиск музыки**: Уникальный модуль для поиска треков.
- 🔧 **Гибкость**: Поддержка любых кастомных параметров (`verbosity`, `reasoning_effort` и др.).

### Установка

```bash
pip install titangpt
```

### Быстрый старт

#### Синхронный клиент

```python
from titangpt import TitanGPT

# Инициализация (ключ можно не передавать, если задан TITANGPT_API_KEY в env)
client = TitanGPT(api_key="ВАШ_КЛЮЧ")

# 1. Чат (Генерация текста)
response = client.chat.completions.create(
    model="gpt-5.2-2025-12-11",
    messages=[
        {"role": "system", "content": "Ты полезный помощник."},
        {"role": "user", "content": "Привет! Расскажи коротко о квантовой физике."}
    ]
)

# Доступ к полям через точку
print(response.choices[0].message.content)

# 2. Генерация изображений
image = client.images.generate(
    prompt="Футуристичный киберпанк город, неон, дождь",
    model="flux",
    size="1024x1024"
)

print(image.data[0].b64_json[:50] + "...") 
```

#### Асинхронный клиент

Идеально для FastAPI и Telegram-ботов.

```python
import asyncio
from titangpt import AsyncTitanGPT

async def main():
    # Используется aiohttp под капотом
    client = AsyncTitanGPT(api_key="ВАШ_КЛЮЧ")

    response = await client.chat.completions.create(
        model="gpt-5.2-2025-12-11",
        messages=[{"role": "user", "content": "Напиши хокку про Python."}]
    )
    print(response.choices[0].message.content)
    
    # Закрытие сессии обязательно
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Продвинутые возможности

Библиотека поддерживает передачу любых параметров, которые принимает API, через `**kwargs`.

```python
response = client.chat.completions.create(
    model="gpt-5.2-2025-12-11", 
    messages=[{"role": "user", "content": "Реши задачу..."}],
    
    # Параметр для o-серии моделей
    reasoning_effort="high",
    
    # Кастомный параметр text (например, для управления детализацией)
    text={
        "verbosity": "low"
    }
)
```

### Музыка и Аудио

```python
# Поиск музыки
track = client.music.search(query="Linkin Park Numb")
print(f"Трек: {track.title}")
print(f"Ссылка: {track.url}")

# Транскрибация (Whisper)
with open("speech.mp3", "rb") as f:
    transcription = client.audio.transcriptions.create(model="whisper-1", file=f)
    print(transcription.text)
```

### Конфигурация

Вы можете настроить базовый URL, таймауты и количество повторных попыток (retries).

```python
client = TitanGPT(
    api_key="...",
    base_url="https://api.titangpt.ru",  # По умолчанию
    timeout=60,                          # Таймаут в секундах
    max_retries=3                        # Количество попыток при ошибках сети
)
```
---

## 🔗 Ссылки и Поддержка

- **Сайт**: [titangpt.ru](https://titangpt.ru)
- **API Endpoint**: `https://api.titangpt.ru`
- **Telegram канал**: [@titangpt_channel](https://t.me/titangpt_channel)
  
Если у вас есть вопросы или нужна помощь, присоединяйтесь к нашему Telegram каналу.

---

## License

MIT License.

*Developed for TitanGPT API.*
