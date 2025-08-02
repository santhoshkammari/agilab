import json
from mrkdwn_analysis import MarkdownAnalyzer


SYSTEM_PROMPT = """You are a markdown analysis assistant. You have access to the following tools for analyzing markdown files:

- markdown_analyzer_get_headers: Extract all headers from markdown content
- markdown_analyzer_get_paragraphs: Extract all paragraphs from markdown content  
- markdown_analyzer_get_links: Extract HTTP/HTTPS links from markdown content
- markdown_analyzer_get_code_blocks: Extract all code blocks from markdown content
- markdown_analyzer_get_tables: Extract all tables from markdown content
- markdown_analyzer_get_lists: Extract all lists from markdown content
- markdown_analyzer_get_overview: Get complete eagle eye view of document structure and content stats

Each tool takes a file path as parameter. Use these tools to analyze markdown files and provide insights about their structure and content."""

def _get_markdown_analyzer(path:str):
    return MarkdownAnalyzer(path)

def markdown_analyzer_get_headers(path:str):
    """Extract all headers from markdown content"""
    analyzer = _get_markdown_analyzer(path)
    return analyzer.identify_headers()

def markdown_analyzer_get_paragraphs(path:str):
    """Extract all paragraphs from markdown content"""
    analyzer = _get_markdown_analyzer(path)
    return analyzer.identify_paragraphs()

def markdown_analyzer_get_links(path:str):
    """Extract HTTP/HTTPS links from markdown content"""
    analyzer = _get_markdown_analyzer(path)
    links = analyzer.identify_links()
    filter_links = [x for x in links['Text link'] if x['url'][:4].lower() == 'http']
    return filter_links

def markdown_analyzer_get_code_blocks(path:str):
    """Extract all code blocks from markdown content"""
    analyzer = _get_markdown_analyzer(path)
    return analyzer.identify_code_blocks()

def markdown_analyzer_get_tables(path:str):
    """Extract all tables from markdown content"""
    analyzer = _get_markdown_analyzer(path)
    return analyzer.identify_tables()

def markdown_analyzer_get_lists(path:str):
    """Extract all lists from markdown content"""
    analyzer = _get_markdown_analyzer(path)
    return analyzer.identify_lists()

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
    
    # Build complete document structure - ALL headers
    header_list = headers.get('Header', [])
    structure = []
    for header in header_list:
        level = header.get('level', 1)
        text = header.get('text', '')
        structure.append(f"{'  ' * (level-1)}H{level}: {text}")
    
    # Create complete overview
    overview_data = {
        "document_title": header_list[0].get('text', 'Untitled') if header_list else 'Untitled',
        "complete_structure": structure,
        "all_headers": [h.get('text', '') for h in header_list],
        "content_stats": {
            "total_sections": len(header_list),
            "paragraphs": total_paragraphs,
            "estimated_words": word_count,
            "code_blocks": len(code_blocks),
            "tables": len(tables),
            "lists": len(lists),
            "external_links": len(http_links)
        },
        "content_types_present": {
            "has_code": len(code_blocks) > 0,
            "has_tables": len(tables) > 0,
            "has_lists": len(lists) > 0,
            "has_links": len(http_links) > 0
        }
    }
    
    # Format as markdown string
    ds = "\n".join(overview_data['complete_structure'])
    hl = "\n".join(f"- {header}" for header in overview_data['all_headers'])
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

if __name__ == '__main__':
    from flowgen.llm.gemini import Gemini
    llm = Gemini(tools=list(tool_functions.values()))
    messages = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":"explain the code part in transformers.md"}
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
            
            # Add tool result back to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": str(tool_result)
            })