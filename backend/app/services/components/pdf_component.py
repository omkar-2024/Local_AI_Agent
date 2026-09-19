from pathlib import Path
from typing import Tuple

import fitz  # PyMuPDF

from app.services.components.image_component import image_component


class PdfComponent:
    """Extracts digital PDF text; falls back to page-image OCR for scanned PDFs."""

    def extract(self, path: Path, workspace_dir: Path) -> Tuple[str, str]:
        doc = fitz.open(path)
        full_text = []
        is_scanned = True

        for page in doc:
            text = page.get_text("text")
            if text.strip():
                is_scanned = False
                full_text.append(text)
        doc.close()

        if is_scanned:
            return "scanned_pdf", self._extract_via_ocr(path, workspace_dir)
        return "digital_pdf", "\n".join(full_text)

    def _extract_via_ocr(self, path: Path, workspace_dir: Path) -> str:
        doc = fitz.open(path)
        extracted_pages = []
        for page_num in range(len(doc)):
            pix = doc[page_num].get_pixmap(dpi=150)
            img_path = workspace_dir / f"temp_{path.stem}_{page_num}.png"
            pix.save(str(img_path))
            extracted_pages.append(image_component.extract(img_path))
            img_path.unlink(missing_ok=True)
        doc.close()
        return "\n".join(extracted_pages)


pdf_component = PdfComponent()
