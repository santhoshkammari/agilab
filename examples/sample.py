from pathlib import Path

from flowgen.utils.custom_markdownify import custom_markdownify
from flowgen.tools.content_extract import extract_markdown_from_url,extract_html_from_url

# md = custom_markdownify(Path("/home/ntlpt59/master/own/flowgen/test/rohith_wiki.html").read_text())
# md = extract_markdown_from_url("https://en.wikipedia.org/wiki/Rohit_Sharma")
html =extract_html_from_url("https://en.wikipedia.org/wiki/Rohit_Sharma")
Path('Rohit_Sharma.html').write_text(html["content"])
md=custom_markdownify(Path('Rohit_Sharma.html').read_text())
print(md)
