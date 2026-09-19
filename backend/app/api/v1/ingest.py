from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
)

from typing import List, Optional

from datetime import (
    datetime,
    timezone,
)

import uuid

from app.core.config import settings

from app.services.purifier import (
    DataPurifierService,
    UnsupportedFileError,
)

from app.services.progress import (
    progress_manager,
)

from app.services.chat_storage import (
    ensure_conversation,
    save_message,
)


router = APIRouter()


purifier = DataPurifierService(
    workspace_dir=settings.WORKSPACE_DIR
)


# ============================================================
# PURIFY REQUEST
# ============================================================

@router.post("/purify")
async def purify_user_request(
    prompt: str = Form(...),
    session_id: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(
        default=[]
    ),
):

    active_session = (
        session_id
        or f"sess_{uuid.uuid4().hex[:10]}"
    )

    active_conversation = (
        conversation_id
        or f"conv_{uuid.uuid4().hex[:12]}"
    )

    request_id = (
        f"req_{uuid.uuid4().hex[:10]}"
    )

    try:

        # ====================================================
        # CONVERSATION
        # ====================================================

        ensure_conversation(
            active_conversation,
            title=(
                " ".join(
                    prompt.strip().split()
                )[:60]
                if prompt.strip()
                else "New Chat"
            ),
        )

        # ====================================================
        # PROGRESS
        # ====================================================

        progress_manager.create(
            active_session
        )

        await progress_manager.emit(
            active_session,
            "REQUEST_RECEIVED",
            "Understanding your request",
            "active",
        )

        # ====================================================
        # SANITIZE
        # ====================================================

        clean_prompt = (
            purifier.sanitize_text(
                prompt
            )
        )

        await progress_manager.emit(
            active_session,
            "REQUEST_UNDERSTOOD",
            "Understanding your request",
            "completed",
        )

        await progress_manager.emit(
            active_session,
            "PURIFYING_REQUEST",
            "Sanitizing input",
            "active",
        )

        await progress_manager.emit(
            active_session,
            "PURIFYING_REQUEST",
            "Sanitizing input",
            "completed",
        )

        # ====================================================
        # ATTACHMENTS
        # ====================================================

        processed_attachments = []

        for upload_file in files:

            filename = (
                upload_file.filename
                or "attachment"
            )

            await progress_manager.emit(
                active_session,
                "PROCESSING_ATTACHMENT",
                f"Processing {filename}",
                "active",
            )

            processed = (
                await purifier.save_and_parse_file(
                    upload_file
                )
            )

            processed_attachments.append(
                processed
            )

            file_type = str(
                processed.get(
                    "file_type",
                    "",
                )
            ).lower()

            if file_type == "image":

                await progress_manager.emit(
                    active_session,
                    "OCR_PROCESSING",
                    "Extracting text with OCR",
                    "completed",
                )

            await progress_manager.emit(
                active_session,
                "PROCESSING_ATTACHMENT",
                f"Processed {filename}",
                "completed",
            )

        # ====================================================
        # SAVE USER MESSAGE
        #
        # Save only after purification succeeded.
        # ====================================================

        attachment_metadata = []

        for attachment in (
            processed_attachments
        ):

            attachment_metadata.append(
                {
                    "filename":
                        attachment.get(
                            "filename"
                        ),
                    "file_type":
                        attachment.get(
                            "file_type"
                        ),
                    "local_path":
                        attachment.get(
                            "local_path"
                        ),
                }
            )

        save_message(
            conversation_id=(
                active_conversation
            ),
            role="user",
            content=clean_prompt,
            request_id=request_id,
            metadata={
                "attachments":
                    attachment_metadata,
            },
        )

        # ====================================================
        # PURIFIED PAYLOAD
        # ====================================================

        purified_payload = {

            "metadata": {

                "request_id":
                    request_id,

                "session_id":
                    active_session,

                "conversation_id":
                    active_conversation,

                "timestamp":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                "status":
                    "PURIFIED_SUCCESS",

                "file_count":
                    len(
                        processed_attachments
                    ),
            },

            "user_request": {

                "raw_prompt":
                    prompt,

                "sanitized_prompt":
                    clean_prompt,
            },

            "attachments":
                processed_attachments,
        }

        return purified_payload

    except UnsupportedFileError as e:

        await progress_manager.emit(
            active_session,
            "ERROR",
            str(e),
            "error",
        )

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:

        await progress_manager.emit(
            active_session,
            "ERROR",
            (
                "Failed to process request: "
                f"{str(e)}"
            ),
            "error",
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to process request: "
                f"{str(e)}"
            ),
        )