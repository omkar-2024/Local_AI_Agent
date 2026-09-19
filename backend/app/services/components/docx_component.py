from pathlib import Path

import docx


class DocxComponent:
    """Extracts paragraph text from Microsoft Word documents."""

    def extract(self, path: Path) -> str:
        document = docx.Document(path)
        return "\n".join(p.text for p in document.paragraphs if p.text.strip())


docx_component = DocxComponent()
