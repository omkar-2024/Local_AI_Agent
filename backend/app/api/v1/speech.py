from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.speech_to_text import speech_to_text_service


router = APIRouter()


@router.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe locally recorded audio using the locally installed
    Whisper model.

    Audio is processed by the local FastAPI backend only.
    No external speech-to-text API is used.
    """

    if not audio.filename:
        raise HTTPException(
            status_code=400,
            detail="No audio file was provided.",
        )

    content_type = str(audio.content_type or "").lower()

    allowed_content_types = {
        "audio/wav",
        "audio/wave",
        "audio/x-wav",
        "audio/webm",
        "audio/ogg",
        "audio/mp4",
        "audio/mpeg",
        "application/octet-stream",
    }

    if content_type and (
        not content_type.startswith("audio/")
        and content_type not in allowed_content_types
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {content_type}",
        )

    try:
        audio_bytes = await audio.read()

        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="The recorded audio is empty.",
            )

        return await speech_to_text_service.transcribe(audio_bytes)

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Speech transcription failed: {exc}",
        ) from exc