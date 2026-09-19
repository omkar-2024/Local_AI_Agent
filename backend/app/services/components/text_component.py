import re


class TextComponent:
    """Sanitizes the raw user prompt: strips control chars, normalizes whitespace."""

    def extract(self, text: str) -> str:
        if not text:
            return ""
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        return cleaned.strip()


text_component = TextComponent()
