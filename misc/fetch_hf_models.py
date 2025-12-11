import asyncio
import aiohttp
import json
import argparse
from tqdm.asyncio import tqdm_asyncio

async def fetch_page(session, page, search_term, sort='modified'):
    headers = {
        'accept': '*/*',
        'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
        'priority': 'u=1, i',
        'referer': f'https://huggingface.co/models?p=1&sort={sort}&search={search_term}',
        'sec-ch-ua': '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Linux"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36'
    }
    url = f'https://huggingface.co/models-json?p={page}&sort={sort}&search={search_term}&withCount=true'
    try:
        async with session.get(url, headers=headers) as response:
            return await response.json()
    except Exception as e:
        return {"page": page, "search_term": search_term, "error": str(e)}

async def main():
    parser = argparse.ArgumentParser(description='Fetch HuggingFace models with deduplication')
    parser.add_argument('-s', '--search', required=True, help='Comma-separated search terms (e.g., "sql,text2sql,nl2sql")')
    parser.add_argument('-o', '--output', default='result.json', help='Output file (default: result.json)')
    parser.add_argument('--sort', default='trending', choices=['trending', 'modified', 'downloads', 'likes'],
                        help='Sort order (default: trending for stable results)')
    args = parser.parse_args()

    search_terms = [term.strip() for term in args.search.split(',')]
    print(f"Searching for: {', '.join(search_terms)}")
    print(f"Sort order: {args.sort}")

    all_models = {}

    async with aiohttp.ClientSession() as session:
        for search_term in search_terms:
            print(f"\nFetching results for '{search_term}'...")
            tasks = [fetch_page(session, page, search_term, args.sort) for page in range(1, 101)]
            results = await tqdm_asyncio.gather(*tasks, desc=f"  {search_term}")

            for page in results:
                if 'models' in page:
                    for model in page['models']:
                        model_id = model.get('id')
                        if model_id and model_id not in all_models:
                            all_models[model_id] = model

            print(f"  Total unique models so far: {len(all_models)}")

    output = {
        'search_terms': search_terms,
        'total_unique_models': len(all_models),
        'models': list(all_models.values())
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Saved {len(all_models)} unique models to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
