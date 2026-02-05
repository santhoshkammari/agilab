import asyncio
from web import search_web
while True:
    query = input("Enter your search query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    print("Starting web search...")
    print(search_web(query=query, max_results=3))

