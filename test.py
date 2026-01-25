import os
import asyncio
from titangpt import TitanGPT, AsyncTitanGPT


API_KEY = os.environ.get("TITANGPT_API_KEY")

def test_sync():
    print("--- [SYNC] Testing ---")
    try:
        client = TitanGPT(api_key=API_KEY)
        print("Checking moderation...")
        res = client.moderations.create("test input")
        print(f"Result: {res}")
        print("Searching Yandex Music...")
        search = client.music.yandex.search("Linkin Park")
        if search and 'result' in search and search.result.tracks:
            print(f"Found track: {search.result.tracks.results[0].title}")
        else:
            print(search)

        client.close()
    except Exception as e:
        print(f"Sync Error: {e}")

async def test_async():
    print("\n--- [ASYNC] Testing ---")
    try:
        async with AsyncTitanGPT(api_key=API_KEY) as client:
            # Тест Threads
            print("Creating Thread...")
            thread = await client.threads.create()
            print(f"Thread ID: {thread.id}")
            
            await client.threads.add_message(thread.id, "Hello!")
            print("Message added.")
            
    except Exception as e:
        print(f"Async Error: {e}")

if __name__ == "__main__":
    test_sync()
    asyncio.run(test_async())