import asyncio
import io
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from app.core.config import settings


class SpeechToTextService:
    """
    Fully local Hugging Face Whisper speech-to-text service.

    Audio is decoded and processed locally.
    The Whisper model is loaded from the local filesystem only.

    This implementation intentionally uses Whisper's processor and
    model directly instead of the high-level Transformers ASR
    pipeline. It also keeps generation arguments minimal to avoid
    version-specific Whisper generation errors.
    """

    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._torch: Optional[Any] = None
        self._device: Optional[str] = None
        self._torch_dtype: Optional[Any] = None
        self._load_lock = Lock()

    def _load_model(self) -> tuple[Any, Any, Any]:
        if (
            self._model is not None
            and self._processor is not None
            and self._torch is not None
        ):
            return self._model, self._processor, self._torch

        with self._load_lock:
            if (
                self._model is not None
                and self._processor is not None
                and self._torch is not None
            ):
                return self._model, self._processor, self._torch

            model_dir = Path(settings.WHISPER_MODEL_DIR)

            if (
                not model_dir.exists()
                or not any(model_dir.iterdir())
            ):
                raise RuntimeError(
                    "Local Whisper model is not installed. "
                    f"Place the downloaded '{settings.WHISPER_MODEL_ID}' "
                    f"model in {model_dir}."
                )

            try:
                import torch
                from transformers import (
                    WhisperForConditionalGeneration,
                    WhisperProcessor,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Local speech-to-text dependencies are missing. "
                    "Install transformers, torch, and soundfile."
                ) from exc

            device_setting = str(
                settings.WHISPER_DEVICE or "auto"
            ).strip().lower()

            cuda_available = torch.cuda.is_available()

            if device_setting == "cuda":
                if not cuda_available:
                    raise RuntimeError(
                        "Whisper is configured for CUDA, "
                        "but CUDA is not available."
                    )
                device = "cuda:0"
                torch_dtype = torch.float16

            elif device_setting.isdigit():
                if not cuda_available:
                    raise RuntimeError(
                        "A CUDA device was requested, "
                        "but CUDA is not available."
                    )
                device_index = int(device_setting)
                device = f"cuda:{device_index}"
                torch_dtype = torch.float16

            elif device_setting == "auto" and cuda_available:
                device = "cuda:0"
                torch_dtype = torch.float16

            else:
                device = "cpu"
                torch_dtype = torch.float32

            try:
                processor = WhisperProcessor.from_pretrained(
                    model_dir,
                    local_files_only=True,
                )

                model = WhisperForConditionalGeneration.from_pretrained(
                    model_dir,
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                    local_files_only=True,
                )

                model.to(device)
                model.eval()

                # Do not carry legacy forced decoder IDs from an older
                # checkpoint/config. Whisper can automatically detect
                # the spoken language and perform transcription.
                model.generation_config.forced_decoder_ids = None
                model.generation_config.task = "transcribe"

            except Exception as exc:
                raise RuntimeError(
                    "Unable to load the local Whisper model. "
                    f"Model directory: {model_dir}. "
                    f"Error: {exc}"
                ) from exc

            self._model = model
            self._processor = processor
            self._torch = torch
            self._device = device
            self._torch_dtype = torch_dtype

            return model, processor, torch

    @staticmethod
    def _decode_wav(
        audio_bytes: bytes,
    ) -> tuple[Any, int]:
        try:
            import numpy as np
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError(
                "soundfile and numpy are required "
                "for local speech transcription."
            ) from exc

        try:
            audio, sample_rate = sf.read(
                io.BytesIO(audio_bytes),
                dtype="float32",
                always_2d=False,
            )
        except Exception as exc:
            raise ValueError(
                "The recorded audio could not be decoded. "
                "Please try recording again."
            ) from exc

        if audio is None:
            raise ValueError("The recorded audio is empty.")

        audio = np.asarray(audio, dtype=np.float32)

        if audio.size == 0:
            raise ValueError("The recorded audio is empty.")

        if sample_rate <= 0:
            raise ValueError(
                "The recorded audio has an invalid sample rate."
            )

        # Convert stereo/multichannel audio to mono.
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        audio = audio.reshape(-1).astype(
            np.float32,
            copy=False,
        )

        duration = len(audio) / float(sample_rate)

        if duration < 0.35:
            raise ValueError(
                "The recording is too short. "
                "Please speak for at least a moment."
            )

        if duration > settings.WHISPER_MAX_AUDIO_SECONDS:
            raise ValueError(
                "Voice input is limited to "
                f"{settings.WHISPER_MAX_AUDIO_SECONDS} seconds."
            )

        audio = np.nan_to_num(
            audio,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

        # Remove DC offset.
        audio = audio - np.mean(
            audio,
            dtype=np.float64,
        )

        # Detect genuinely silent recordings before normalization.
        rms = float(
            np.sqrt(
                np.mean(
                    np.square(audio),
                    dtype=np.float64,
                )
            )
        )
        peak = float(np.max(np.abs(audio)))

        if peak < 0.001 or rms < 0.00015:
            raise ValueError(
                "No usable speech was detected. "
                "Please speak clearly into the microphone."
            )

        # Normalize only unusually quiet microphone recordings.
        if 0.0 < peak < 0.25:
            gain = min(4.0, 0.85 / peak)
            audio = np.clip(
                audio * gain,
                -1.0,
                1.0,
            ).astype(np.float32)

        # Whisper expects 16 kHz audio.
        target_rate = 16000

        if sample_rate != target_rate:
            target_length = max(
                1,
                int(
                    round(
                        len(audio)
                        * target_rate
                        / sample_rate
                    )
                ),
            )

            if len(audio) == 1:
                audio = np.repeat(
                    audio,
                    target_length,
                ).astype(np.float32)
            else:
                old_positions = np.linspace(
                    0,
                    len(audio) - 1,
                    num=len(audio),
                    dtype=np.float32,
                )
                new_positions = np.linspace(
                    0,
                    len(audio) - 1,
                    num=target_length,
                    dtype=np.float32,
                )

                audio = np.interp(
                    new_positions,
                    old_positions,
                    audio,
                ).astype(np.float32)

            sample_rate = target_rate

        return audio, sample_rate

    def _transcribe_sync(
        self,
        audio: Any,
        sample_rate: int,
    ) -> str:
        model, processor, torch = self._load_model()

        import numpy as np

        audio = np.asarray(
            audio,
            dtype=np.float32,
        ).reshape(-1)

        if audio.size == 0:
            raise ValueError(
                "No audio samples were available for transcription."
            )

        inputs = processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )

        input_features = inputs.get("input_features")

        if input_features is None:
            raise RuntimeError(
                "Whisper processor did not produce input features."
            )

        # Whisper expects [batch, mel_features, time].
        if input_features.ndim == 2:
            input_features = input_features.unsqueeze(0)

        if input_features.ndim != 3:
            raise RuntimeError(
                "Invalid Whisper input tensor shape: "
                f"{tuple(input_features.shape)}. "
                "Expected [batch, mel_features, time]."
            )

        input_features = input_features.to(
            device=self._device,
            dtype=self._torch_dtype,
        )

        # Keep generation deliberately simple. The previous implementation
        # passed long-form decoding thresholds that can be None-dependent
        # across Transformers versions and caused:
        # "'>' not supported between instances of 'NoneType' and 'float'".
        with torch.inference_mode():
            generated_ids = model.generate(
                input_features=input_features,
                max_new_tokens=128,
                do_sample=False,
            )

        if generated_ids is None:
            return ""

        decoded = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        if not decoded:
            return ""

        return str(decoded[0]).strip()

    async def transcribe(
        self,
        audio_bytes: bytes,
    ) -> dict[str, Any]:
        if not audio_bytes:
            raise ValueError("No audio was provided.")

        max_bytes = (
            settings.WHISPER_MAX_AUDIO_MB
            * 1024
            * 1024
        )

        if len(audio_bytes) > max_bytes:
            raise ValueError(
                "Audio exceeds the "
                f"{settings.WHISPER_MAX_AUDIO_MB}MB limit."
            )

        audio, sample_rate = await asyncio.to_thread(
            self._decode_wav,
            audio_bytes,
        )

        text = await asyncio.to_thread(
            self._transcribe_sync,
            audio,
            sample_rate,
        )

        text = text.strip()

        if not text:
            raise ValueError(
                "No speech could be recognized. "
                "Please speak clearly and try again."
            )

        return {
            "text": text,
            "model": settings.WHISPER_MODEL_ID,
            "local": True,
            "device": self._device,
        }


speech_to_text_service = SpeechToTextService()
