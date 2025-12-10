import asyncio
import aiohttp
import json
import argparse
from tqdm.asyncio import tqdm_asyncio

async def fetch_page(session, page, search_term):
    headers = {
        'accept': '*/*',
        'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
        'priority': 'u=1, i',
        'referer': f'https://huggingface.co/models?p=1&sort=trending&search={search_term}',
        'sec-ch-ua': '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Linux"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36'
    }
    url = f'https://huggingface.co/models-json?p={page}&sort=trending&search={search_term}&withCount=true'
    try:
        async with session.get(url, headers=headers) as response:
            return await response.json()
    except Exception as e:
        return {"page": page, "error": str(e)}

async def main():
    parser = argparse.ArgumentParser(description='Fetch HuggingFace models')
    parser.add_argument('-s', '--search', required=True, help='Search term')
    parser.add_argument('-o', '--output', default='result.json', help='Output file (default: result.json)')
    args = parser.parse_args()

    results = []
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_page(session, page, args.search) for page in range(1, 101)]
        results = await tqdm_asyncio.gather(*tasks, desc="Fetching pages")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} pages to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
