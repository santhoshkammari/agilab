from pathlib import Path

v = Path('/home/ntlpt59/master/own/flowgen/test/virat_wiki.html').read_text()
r = Path('/home/ntlpt59/master/own/flowgen/test/rohith_wiki.html').read_text()
s = Path('/home/ntlpt59/master/own/flowgen/test/sbert.html').read_text()


# pip install markitdown
# from markitdown import MarkItDown
# md = MarkItDown()
# res = md.convert(v)
# print(res.markdown)

# pip install markdownify
# from markdownify import markdownify as md
# print(md(r))

# pip install
import textwrap
from tabulate import tabulate
from flowgen.tools.markdown import markdown_analyzer_get_tables


def format_beautiful_table(table_data, max_width=40, tablefmt='grid'):
    """
    Create beautifully formatted tables with proper text wrapping

    Args:
        table_data: Dictionary with 'header' and 'rows' keys
        max_width: Maximum width for each column
        tablefmt: Table format style
    """

    def wrap_text(text, width=max_width):
        """Wrap text to specified width"""
        if not isinstance(text, str):
            text = str(text)
        return '\n'.join(textwrap.wrap(text, width=width, break_long_words=True))

    # Wrap headers
    wrapped_headers = [wrap_text(header) for header in table_data['header']]

    # Wrap all cell content
    wrapped_rows = []
    for row in table_data['rows']:
        wrapped_row = [wrap_text(cell) for cell in row]
        wrapped_rows.append(wrapped_row)

    # Generate beautiful table
    return tabulate(
        wrapped_rows,
        headers=wrapped_headers,
        tablefmt=tablefmt,
        stralign='left'
    )


# Your improved code
tables = markdown_analyzer_get_tables('/home/ntlpt59/master/own/flowgen/rohit_wiki.md')