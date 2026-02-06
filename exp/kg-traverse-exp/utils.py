import pymupdf4llm
from pathlib import Path
from tqdm import tqdm


def extract_pdf2md(pdf_path):
    pdfs = list(Path(pdf_path).glob("[!.]*.pdf"))
    for pdf in tqdm(pdfs):
        md_text = pymupdf4llm.to_markdown(str(pdf))
        md_path = Path(str(pdf).replace(".pdf", ".md").replace("pdfs", "mds"))
        md_path.parent.mkdir(exist_ok=True, parents=True)
        Path(md_path).write_text(md_text)