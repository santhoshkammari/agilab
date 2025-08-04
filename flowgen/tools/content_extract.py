import json
from dataclasses import asdict
from typing import List

from pydantic import BaseModel
from liteauto.parselite import parse
from flowgen.utils.custom_markdownify import custom_markdownify

SYSTEM_PROMPT = """You are a content extraction assistant. You have access to the following tools for extracting and processing web content:

- content_extractor_parse_urls: Parse URLs and extract their HTML content
- content_extractor_html_to_markdown: Convert HTML content to markdown format
- content_extractor_batch_process: Parse multiple URLs and convert them to markdown in batch

Use these tools to extract, parse, and convert web content as requested by the user."""

from trafilatura import fetch_url
def extract_html_from_url(url: str):
    content = fetch_url(url)
    # if isinstance(result, list):
    #     res = [{"url": x.url, "content": x.content} for x in result]
    #     return res
    return {"url": url, "content": content}


def extract_markdown_from_html(html: str):
    return custom_markdownify(html) if html else ""


def extract_markdown_from_url(url: str):
    """Parse multiple list of URLs and convert them to markdown in batch
    Args:
        url: the html url string
    """
    htmls = extract_html_from_url(url)
    if isinstance(htmls,list):
        results = {}
        for x in htmls:
            results[x["url"]] = extract_markdown_from_html(x["content"])
        return json.dumps(results)

    name = url.split("/")[-1].split(".")[0]
    with open(f"{name}.md","w") as f:
        f.write(extract_markdown_from_html(htmls['content']))
    return f"Markdown Saved at {name}.md"

tool_functions = {
    # "extract_html_from_url": extract_html_from_url,
    # "extract_markdown_from_html": extract_markdown_from_html,
    "extract_markdown_from_url": extract_markdown_from_url,
}


def run_example():
    from flowgen.llm.gemini import Gemini
    llm = Gemini(tools=list(tool_functions.values()))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Extract content from https://example.com and convert to markdown"}
    ]

    # Agentic loop - keep calling tools until no more tool calls
    while True:
        response = llm(messages)

        # Check if there are tool calls
        if 'tools' not in response or not response['tools']:
            # No more tool calls, show final content
            print("=== FINAL RESPONSE ===")
            print(response.get('content', 'No content'))
            break

        # Add the assistant message with tool calls first
        messages.append({
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_call.get('id', f"call_{i}"),
                    "function": {
                        "name": tool_call['name'],
                        "arguments": json.dumps(tool_call['arguments'])
                    },
                    "type": "function"
                }
                for i, tool_call in enumerate(response['tools'])
            ]
        })

        # Process each tool call and add tool results
        for i, tool_call in enumerate(response['tools']):
            tool_name = tool_call['name']
            tool_args = tool_call['arguments']
            tool_id = tool_call.get('id', f"call_{i}")

            print(f"Calling tool: {tool_name} with args: {tool_args}")

            # Execute the tool
            tool_result = tool_functions[tool_name](**tool_args)
            print('============')
            print(tool_result)
            print('============')

            # Pretty print tool result
            print(f"=== TOOL RESULT: {tool_name} ===")
            if isinstance(tool_result, dict):
                for key, value in tool_result.items():
                    print(f"{key}:")
                    value_preview = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
                    print(f"  {value_preview}")
            else:
                result_preview = str(tool_result)[:200] + "..." if len(str(tool_result)) > 200 else str(tool_result)
                print(result_preview)
            print("=" * 40)

            # Add tool result back to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": str(tool_result)
            })


def run_sample():
    urls = ["https://example.com"]
    res = extract_markdown_from_url(urls)
    print(res)


if __name__ == '__main__':
    run_sample()
