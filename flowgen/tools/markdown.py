import json
from ..utils.mrkdwn_analysis import MarkdownAnalyzer


SYSTEM_PROMPT = """You are a markdown analysis assistant. You have access to the following tools for analyzing markdown files:

- markdown_analyzer_get_headers: Extract all headers from markdown content with line numbers
- markdown_analyzer_get_paragraphs: Extract all paragraphs from markdown content with line numbers
- markdown_analyzer_get_links: Extract HTTP/HTTPS links from markdown content with line numbers
- markdown_analyzer_get_code_blocks: Extract all code blocks from markdown content with line numbers
- markdown_analyzer_get_tables: Extract all tables from markdown content with line numbers
- markdown_analyzer_get_lists: Extract all lists from markdown content with line numbers
- markdown_analyzer_get_overview: Get complete eagle eye view of document structure and content stats

Each tool takes a file path as parameter and returns data with line numbers for easy navigation. Use these tools to analyze markdown files and provide insights about their structure and content."""

def _get_markdown_analyzer(path:str):
    print(f"DEBUG: Creating MarkdownAnalyzer for {path}")
    import time
    start = time.time()
    analyzer = MarkdownAnalyzer(path)
    end = time.time()
    print(f"DEBUG: MarkdownAnalyzer created in {end-start:.2f} seconds")
    return analyzer

def markdown_analyzer_get_headers(path:str):
    """Extract all headers from markdown content with line numbers"""
    analyzer = _get_markdown_analyzer(path)
    return analyzer.identify_headers()

def markdown_analyzer_get_paragraphs(path:str):
    """Extract all paragraphs from markdown content with line numbers"""
    analyzer = _get_markdown_analyzer(path)
    result = {"Paragraph": []}
    for token in analyzer.tokens:
        if token.type == 'paragraph':
            result["Paragraph"].append({"line": token.line, "content": token.content})
    return result

def markdown_analyzer_get_links(path:str):
    """Extract HTTP/HTTPS links from markdown content with line numbers"""
    analyzer = _get_markdown_analyzer(path)
    links = analyzer.identify_links()
    filter_links = [x for x in links.get('Text link', []) if x.get('url', '').lower().startswith('http')]
    return filter_links

def markdown_analyzer_get_code_blocks(path:str):
    """Extract all code blocks from markdown content with line numbers"""
    analyzer = _get_markdown_analyzer(path)
    return analyzer.identify_code_blocks()

def markdown_analyzer_get_tables(path:str):
    """Extract all tables from markdown content with line numbers"""
    analyzer = _get_markdown_analyzer(path)
    result = {"Table": []}
    for token in analyzer.tokens:
        if token.type == 'table':
            result["Table"].append({
                "line": token.line,
                "header": token.meta["header"],
                "rows": token.meta["rows"]
            })
    return result

def markdown_analyzer_get_lists(path:str):
    """Extract all lists from markdown content with line numbers"""
    analyzer = _get_markdown_analyzer(path)
    result = {"Ordered list": [], "Unordered list": []}
    for token in analyzer.tokens:
        if token.type == 'ordered_list':
            result["Ordered list"].append({"line": token.line, "items": token.meta["items"]})
        elif token.type == 'unordered_list':
            result["Unordered list"].append({"line": token.line, "items": token.meta["items"]})
    return result

def markdown_analyzer_get_overview(path:str):
    """Get eagle eye view of markdown document - complete structure and content overview"""
    analyzer = _get_markdown_analyzer(path)
    
    # Get all components
    headers = analyzer.identify_headers()
    paragraphs = analyzer.identify_paragraphs()
    links = analyzer.identify_links()
    code_blocks = analyzer.identify_code_blocks()
    tables = analyzer.identify_tables()
    lists = analyzer.identify_lists()
    
    # Filter HTTP links
    http_links = [x for x in links.get('Text link', []) if x.get('url', '').lower().startswith('http')]
    
    # Calculate statistics
    paragraph_list = paragraphs.get('Paragraph', [])
    total_paragraphs = len(paragraph_list)
    word_count = sum(len(p.split()) for p in paragraph_list)
    
    # Build complete document structure - ALL headers with line numbers
    header_list = headers.get('Header', [])
    structure = []
    for header in header_list:
        level = header.get('level', 1)
        text = header.get('text', '')
        line = header.get('line', 'N/A')
        structure.append(f"{'  ' * (level-1)}H{level}: {text} (line {line})")
    
    # Build code blocks summary with line numbers
    code_block_summary = []
    for cb in code_blocks.get('Code block', []):
        lang = cb.get('language', 'unknown')
        start = cb.get('start_line', 'N/A')
        end = cb.get('end_line', 'N/A')
        code_block_summary.append(f"{lang} code block (lines {start}-{end})")
    
    # Build tables summary with line numbers
    table_summary = []
    for table in tables.get('Table', []):
        line = table.get('line', 'N/A')
        table_summary.append(f"Table at line {line}")
    
    # Build lists summary with line numbers
    list_summary = []
    from rich import print

    # Create complete overview
    overview_data = {
        "document_title": header_list[0].get('text', 'Untitled') if header_list else 'Untitled',
        "complete_structure": structure,
        "all_headers": [h.get('text', '') for h in header_list],
        "code_blocks_detail": code_block_summary,
        "tables_detail": table_summary,
        # "lists_detail": list_summary,
        "content_stats": {
            "total_sections": len(header_list),
            "paragraphs": total_paragraphs,
            "estimated_words": word_count,
            "code_blocks": len(code_blocks.get('Code block', [])),
            "tables": len(tables.get('Table', [])),
            "lists": len(lists.get('Ordered list', [])) + len(lists.get('Unordered list', [])),
            "external_links": len(http_links)
        },
        "content_types_present": {
            "has_code": len(code_blocks.get('Code block', [])) > 0,
            "has_tables": len(tables.get('Table', [])) > 0,
            "has_lists": len(lists.get('Ordered list', [])) + len(lists.get('Unordered list', [])) > 0,
            "has_links": len(http_links) > 0
        }
    }
    
    # Format as markdown string
    ds = "\n".join(overview_data['complete_structure'])
    hl = "\n".join(f"- {header}" for header in overview_data['all_headers'])
    cb_detail = "\n".join(f"- {cb}" for cb in overview_data['code_blocks_detail']) if overview_data['code_blocks_detail'] else "None"
    table_detail = "\n".join(f"- {table}" for table in overview_data['tables_detail']) if overview_data['tables_detail'] else "None"
    # list_detail = "\n".join(f"- {lst}" for lst in overview_data['lists_detail']) if overview_data['lists_detail'] else "None"
    
    markdown_overview = f"""# Document Overview: {overview_data['document_title']}

## Document Structure
{ds}

## Content Statistics
- **Total Sections**: {overview_data['content_stats']['total_sections']}
- **Paragraphs**: {overview_data['content_stats']['paragraphs']}
- **Estimated Words**: {overview_data['content_stats']['estimated_words']}
- **Code Blocks**: {overview_data['content_stats']['code_blocks']}
- **Tables**: {overview_data['content_stats']['tables']}
- **Lists**: {overview_data['content_stats']['lists']}
- **External Links**: {overview_data['content_stats']['external_links']}

## Code Blocks Detail
{cb_detail}

## Tables Detail
{table_detail}


## Content Types Present
- **Has Code**: {'Yes' if overview_data['content_types_present']['has_code'] else 'No'}
- **Has Tables**: {'Yes' if overview_data['content_types_present']['has_tables'] else 'No'}
- **Has Lists**: {'Yes' if overview_data['content_types_present']['has_lists'] else 'No'}
- **Has External Links**: {'Yes' if overview_data['content_types_present']['has_links'] else 'No'}

## All Headers List
{hl}
"""
    
    return markdown_overview

tool_functions = {
    "markdown_analyzer_get_headers":markdown_analyzer_get_headers,
    "markdown_analyzer_get_paragraphs":markdown_analyzer_get_paragraphs,
    "markdown_analyzer_get_links":markdown_analyzer_get_links,
    "markdown_analyzer_get_code_blocks":markdown_analyzer_get_code_blocks,
    "markdown_analyzer_get_tables":markdown_analyzer_get_tables,
    "markdown_analyzer_get_lists":markdown_analyzer_get_lists,
    "markdown_analyzer_get_overview":markdown_analyzer_get_overview,
}


def run_example():
    from flowgen.llm.gemini import Gemini
    llm = Gemini(tools=list(tool_functions.values()))
    messages = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":"explain the code part in test/transformers.md"}
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
            
            # Pretty print tool result with line numbers and truncated content
            print(f"=== TOOL RESULT: {tool_name} ===")
            if isinstance(tool_result, dict):
                for key, items in tool_result.items():
                    print(f"{key}:")
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                line_info = f"Line {item.get('line', 'N/A')}: " if 'line' in item else ""
                                if 'content' in item:
                                    content_preview = item['content'][:50] + "..." if len(item['content']) > 50 else item['content']
                                    print(f"  {line_info}{content_preview}")
                                elif 'text' in item:
                                    text_preview = item['text'][:50] + "..." if len(item['text']) > 50 else item['text']
                                    print(f"  {line_info}{text_preview}")
                                else:
                                    print(f"  {line_info}{str(item)[:50]}...")
                            else:
                                item_preview = str(item)[:50] + "..." if len(str(item)) > 50 else str(item)
                                print(f"  {item_preview}")
                    else:
                        print(f"  {str(items)[:100]}...")
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
    print("DEBUG: Starting run_sample()")
    import time
    start = time.time()
    
    # res = markdown_analyzer_get_overview('test/transformers.md')
    print("DEBUG: Calling markdown_analyzer_get_code_blocks")
    res = markdown_analyzer_get_code_blocks("/home/ntlpt59/master/own/flowgen/grpo_trainer.md")
    
    end = time.time()
    print(f"DEBUG: run_sample completed in {end-start:.2f} seconds")
    print("DEBUG: Result:")
    print(res)
    print('done')

if __name__ == '__main__':
    run_sample()
