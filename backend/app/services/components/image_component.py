from pathlib import Path

import cv2


class ImageComponent:
    """Denoises images with OpenCV and extracts text via PaddleOCR."""

    def __init__(self):
        # Lazy-loaded: PaddleOCR is expensive to initialize
        self._ocr_engine = None

    @property
    def ocr_engine(self):
        if self._ocr_engine is None:
            from paddleocr import PaddleOCR
            # use_doc_orientation_classify/unwarping/textline_orientation disabled: unneeded extra model
            # downloads. enable_mkldnn disabled: triggers a PIR/oneDNN inference bug on this Paddle build.
            # tiny det/rec models: lowest-latency variants, traded off against some accuracy.
            self._ocr_engine = PaddleOCR(
                lang='en',
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
                text_detection_model_name='PP-OCRv6_tiny_det',
                text_recognition_model_name='PP-OCRv6_tiny_rec',
            )
        return self._ocr_engine

    def extract(self, path: Path) -> str:
        img = cv2.imread(str(path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        # PaddleOCR expects a 3-channel array
        denoised_bgr = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

        # Pass the array directly: avoids a redundant disk write/read round-trip
        results = self.ocr_engine.predict(input=denoised_bgr)
        lines = []
        for res in results:
            lines.extend(res.get("rec_texts", []))

        return " ".join(lines)


image_component = ImageComponent()
