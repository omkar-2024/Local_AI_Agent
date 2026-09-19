from pathlib import Path

from typing import (
    Any,
    Dict,
    Optional,
)

from fastapi import (
    APIRouter,
    HTTPException,
)

from fastapi.responses import (
    FileResponse,
    StreamingResponse,
)

from pydantic import BaseModel


from app.agent.orchestrator import (
    run_agent,
    test_llm,
)

from app.agent.sandbox import execute_code

from app.core.config import settings

from app.services.progress import (
    progress_manager,
)

from app.services.chat_storage import (
    get_messages,
    get_recent_messages,
    get_conversations,
    save_message,
    delete_conversation,
)



router = APIRouter()


# ============================================================
# LOCAL CODE EXECUTION
# ============================================================
class ExecuteCodeRequest(BaseModel):
    code: str
    language: str = "python"

@router.post("/execute-code")
async def execute_code_endpoint(request: ExecuteCodeRequest):
    """Execute user-requested code through the existing local sandbox."""
    try:
        return execute_code(
            request.code,
            request.language,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Code execution failed: {exc}",
        )


# ============================================================
# AGENT RUN REQUEST
# ============================================================

class AgentRunRequest(BaseModel):

    purified_json: Dict[str, Any]


# ============================================================
# CODE EXECUTION REQUEST
# ============================================================

class ExecuteCodeRequest(BaseModel):

    code: str
    language: str = "python"


# ============================================================
# BUILD CONVERSATION CONTEXT
# ============================================================

def _build_conversation_context(
    messages: list[dict[str, Any]],
) -> str:

    if not messages:
        return ""

    parts = []

    for message in messages:

        role = str(
            message.get(
                "role",
                "",
            )
        ).upper()

        content = str(
            message.get(
                "content",
                "",
            )
        ).strip()

        if not content:
            continue

        # Keep individual messages bounded.
        if len(content) > 6000:
            content = (
                content[:6000]
                + "\n[message truncated]"
            )

        parts.append(
            f"{role}:\n{content}"
        )

    if not parts:
        return ""

    return (
        "PREVIOUS CONVERSATION CONTEXT\n"
        "================================\n"
        + "\n\n".join(parts)
        + "\n\n"
        "Use this context to understand "
        "follow-up questions. The latest "
        "user request remains the current "
        "request."
    )


# ============================================================
# RUN AGENT
# ============================================================

@router.post("/run")
async def run_agent_endpoint(
    request: AgentRunRequest,
):

    purified_json = request.purified_json

    metadata = (
        purified_json.get(
            "metadata",
            {},
        )
        or {}
    )

    session_id = metadata.get(
        "session_id"
    )

    conversation_id = metadata.get(
        "conversation_id"
    )

    request_id = metadata.get(
        "request_id"
    )

    try:

        # ====================================================
        # PROGRESS SESSION
        # ====================================================

        if session_id:

            progress_manager.set_session(
                session_id
            )

            # Every user request gets a completely fresh live pipeline.
            # This prevents the monitor from mixing the previous request's
            # events with the current request.
            progress_manager.reset_history(
                session_id
            )

            await progress_manager.emit(
                session_id,
                "REQUEST_RECEIVED",
                "Frontend request received by FastAPI",
                "completed",
            )

            await progress_manager.emit(
                session_id,
                "REQUEST_UNDERSTOOD",
                "FastAPI accepted the request",
                "completed",
            )

            await progress_manager.emit(
                session_id,
                "ROUTING_REQUEST",
                "Routing request",
                "active",
            )

        # ====================================================
        # LOAD PREVIOUS CHAT
        #
        # Image analysis is latency-sensitive and Qwen-VL already receives
        # the image itself. Injecting up to 12 previous messages into the
        # vision prompt only increases context processing time. Skip chat
        # history when a real image attachment is present.
        # ====================================================

        attachments = (
            purified_json.get("attachments", [])
            or []
        )

        has_image_attachment = any(
            str(a.get("file_type", "")).lower() == "image"
            for a in attachments
            if isinstance(a, dict)
        )

        previous_messages = []

        if conversation_id and not has_image_attachment:

            previous_messages = (
                get_recent_messages(
                    conversation_id,
                    limit=12,
                    exclude_request_id=(
                        request_id
                    ),
                )
            )

        conversation_context = (
            _build_conversation_context(
                previous_messages
            )
        )

        # ====================================================
        # CONNECT HISTORY TO EXISTING ORCHESTRATOR
        #
        # The existing orchestrator already includes
        # attachment extracted_text inside its purified
        # context. We therefore add history as a synthetic
        # text attachment.
        #
        # This avoids changing the working orchestrator.
        # ====================================================

        if conversation_context:

            attachments = (
                purified_json.setdefault(
                    "attachments",
                    [],
                )
            )

            attachments.append(
                {
                    "filename":
                        "conversation_history.txt",

                    "file_type":
                        "text",

                    "extracted_text":
                        conversation_context,

                    "conversation_context":
                        True,
                }
            )

        # ====================================================
        # RUN EXISTING AGENT
        # ====================================================

        result = await run_agent(
            purified_json
        )

        # ====================================================
        # STORE ASSISTANT RESPONSE
        # ====================================================

        if conversation_id:

            final_answer = (
                result.get(
                    "final_answer",
                    "",
                )
                or result.get(
                    "answer",
                    "",
                )
                or result.get(
                    "response",
                    "",
                )
                or ""
            )

            save_message(
                conversation_id=(
                    conversation_id
                ),
                role="assistant",
                content=final_answer,
                request_id=request_id,
                metadata={
                    "category":
                        result.get(
                            "category"
                        ),

                    "model_used":
                        result.get(
                            "model_used"
                        ),

                    "generated_files":
                        result.get(
                            "generated_files",
                            [],
                        ),

                    "execution_results":
                        result.get(
                            "execution_results",
                            [],
                        ),

                    "tool_calls_made":
                        result.get(
                            "tool_calls_made",
                            0,
                        ),
                },
            )

        # ====================================================
        # PROGRESS COMPLETION
        # ====================================================

        if session_id:

            await progress_manager.emit(
                session_id,
                "ROUTING_REQUEST",
                "Routing request",
                "completed",
            )

            model_name = result.get(
                "model_used",
                "local worker",
            )

            await progress_manager.emit(
                session_id,
                "MODEL_SELECTED",
                f"Selected model: {model_name}",
                "completed",
            )

            generated_files = (
                result.get(
                    "generated_files",
                    [],
                )
            )

            if generated_files:

                for file in generated_files:

                    filename = file.get(
                        "filename",
                        "generated file",
                    )

                    tool_name = file.get(
                        "tool",
                        "",
                    )

                    if (
                        tool_name
                        == "create_word_document"
                    ):

                        message = (
                            "Created Word document: "
                            f"{filename}"
                        )

                    elif (
                        tool_name
                        == "create_excel_document"
                    ):

                        message = (
                            "Created Excel document: "
                            f"{filename}"
                        )

                    elif (
                        tool_name
                        == "create_pptx_document"
                    ):

                        message = (
                            "Created PowerPoint: "
                            f"{filename}"
                        )

                    else:

                        message = (
                            f"Created file: "
                            f"{filename}"
                        )

                    await progress_manager.emit(
                        session_id,
                        "DOCUMENT_CREATED",
                        message,
                        "completed",
                    )

            await progress_manager.emit(
                session_id,
                "GENERATING_RESPONSE",
                "Generating final response",
                "completed",
            )

            await progress_manager.emit(
                session_id,
                "COMPLETED",
                "Request completed",
                "completed",
                {
                    "final": True,
                    "result": result,
                },
            )

        return result

    except Exception as e:

        if session_id:

            await progress_manager.emit(
                session_id,
                "ERROR",
                str(e),
                "error",
            )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Agent run failed: {str(e)}"
            ),
        )


# ============================================================
# GET CURRENT CONVERSATION MESSAGES
# ============================================================

@router.get(
    "/conversations/{conversation_id}/messages"
)
async def conversation_messages(
    conversation_id: str,
):

    try:

        messages = get_messages(
            conversation_id
        )

        return {
            "conversation_id":
                conversation_id,
            "messages":
                messages,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load conversation: "
                f"{str(e)}"
            ),
        )


# ============================================================
# GET CONVERSATION LIST
# ============================================================

@router.get("/conversations")
async def conversations():

    try:

        return {
            "conversations":
                get_conversations()
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load conversations: "
                f"{str(e)}"
            ),
        )

# ============================================================
# DELETE CONVERSATION
# ============================================================

@router.delete(
    "/conversations/{conversation_id}"
)
async def delete_conversation_endpoint(
    conversation_id: str,
):
    try:
        deleted = delete_conversation(
            conversation_id
        )

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        return {
            "success": True,
            "conversation_id":
                conversation_id,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to delete conversation: "
                f"{str(e)}"
            ),
        )
    
# ============================================================
# LIVE PROGRESS SSE
# ============================================================

@router.get(
    "/events/{session_id}"
)
async def agent_progress_events(
    session_id: str,
):

    return StreamingResponse(
        progress_manager.stream(
            session_id
        ),
        media_type=(
            "text/event-stream"
        ),
        headers={
            "Cache-Control":
                "no-cache",

            "Connection":
                "keep-alive",

            "X-Accel-Buffering":
                "no",
        },
    )


# ============================================================
# DOWNLOAD GENERATED FILE
# ============================================================

@router.get(
    "/download/{filename:path}"
)
async def download_generated_file(
    filename: str,
):

    workspace = (
        settings.WORKSPACE_DIR
        .resolve()
    )

    requested_path = (
        workspace / filename
    ).resolve()

    if (
        requested_path != workspace
        and workspace
        not in requested_path.parents
    ):

        raise HTTPException(
            status_code=403,
            detail="Invalid file path.",
        )

    if not requested_path.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Generated file not found."
            ),
        )

    if not requested_path.is_file():

        raise HTTPException(
            status_code=400,
            detail=(
                "Requested path is not a file."
            ),
        )

    return FileResponse(
        path=str(
            requested_path
        ),
        filename=(
            requested_path.name
        ),
        media_type=(
            "application/octet-stream"
        ),
    )


# ============================================================
# TEST LLM
# ============================================================

class TestLLMRequest(BaseModel):

    prompt: str

    model: Optional[str] = None

    system_prompt: Optional[str] = None

    use_tools: bool = False


@router.post("/test-llm")
async def test_llm_endpoint(
    request: TestLLMRequest,
):

    try:

        return await test_llm(
            request.prompt,
            request.model,
            request.system_prompt,
            request.use_tools,
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"LLM test failed: {str(e)}"
            ),
        )