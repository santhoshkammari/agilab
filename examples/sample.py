from pathlib import Path

from flowgen.utils.custom_markdownify import custom_markdownify

md = custom_markdownify(Path("/home/ntlpt59/master/own/flowgen/test/rohith_wiki.html").read_text())

print(md)