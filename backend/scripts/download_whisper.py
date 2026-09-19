from pathlib import Path

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


MODEL_ID = "openai/whisper-base"
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "whisper-base"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {MODEL_ID} to {MODEL_DIR}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        use_safetensors=True,
    )

    processor.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR, safe_serialization=True)

    print("Whisper model download complete.")
    print(f"Local model directory: {MODEL_DIR}")


if __name__ == "__main__":
    main()
