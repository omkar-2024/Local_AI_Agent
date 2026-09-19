from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import base64
import json
import logging
import re

import httpx

from PIL import Image, ImageOps

from app.agent.tools import AGENT_TOOLS
from app.agent.filesystem import execute_filesystem_prompt
from app.core.config import settings
from app.services.llm_router import router_service
from app.services.rag_engine import rag_engine

from app.services.progress import (
    emit_event,
    ROUTING_REQUEST,
    ROUTE_SELECTED,
    MODEL_SELECTED,
    EXECUTING_TOOL,
    RUNNING_PYTHON,
    CREATING_WORD_DOCUMENT,
    CREATING_EXCEL_DOCUMENT,
    CREATING_POWERPOINT,
    GENERATING_RESPONSE,
    COMPLETED,
)


logger = logging.getLogger(__name__)

_ollama_http_client: httpx.AsyncClient | None = None


async def _get_ollama_http_client() -> httpx.AsyncClient:
    global _ollama_http_client
    if _ollama_http_client is None or _ollama_http_client.is_closed:
        _ollama_http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(360.0, connect=20.0),
            limits=httpx.Limits(
                max_connections=8,
                max_keepalive_connections=4,
            ),
        )
    return _ollama_http_client


# =========================================================
# WORKER SYSTEM PROMPT
# =========================================================

AGENT_SYSTEM_PROMPT = """
You are the downstream worker LLM in an on-premises,
air-gapped industrial AI workbench.

The lightweight router has already selected the task
category and capabilities.

Perform the user's task using only:

- the supplied prompt
- purified attachment context
- the tools selected by the router

IMPORTANT:

For actual coding requests:
- provide correct code
- explain only when useful
- do not execute code unless explicitly requested

For explicit RUN, EXECUTE, or TEST requests:
- use run_python_code when available

For engineering calculations:
- ALWAYS use engineering_calculator.
- NEVER perform engineering arithmetic yourself.
- NEVER use run_python_code for engineering calculations.
- engineering_calculator is deterministic.

For local filesystem requests:
- use the selected filesystem tools
- use the exact path supplied by the user
- do not invent or modify filesystem paths

For internal/company knowledge:
- If TRUSTED ORGANIZATION KNOWLEDGE CONTEXT is supplied, use it directly as the
  source of truth. Do not call search_knowledge_base and do not ask the user
  to provide the document again.
- Answer only from the supplied trusted context for organization-specific claims.
- For questions asking for a list, number, name, date, responsibility, definition,
  policy requirement, or other exact fact, extract the answer faithfully from the
  context. Do not replace document terminology with plausible alternatives.
- If multiple source chunks are supplied, prefer the chunk from the explicitly
  named document and preserve the wording/meaning from that source.
- Cite the source name and page number shown in the trusted context when available.
- If the trusted context says that no sufficiently relevant information was found,
  say that the local knowledge base does not contain the requested information.
- Never invent or fill missing organization-specific facts from general knowledge.
- Do not print tool-call syntax such as search_knowledge_base(...).

For document generation:
- actually call the requested document-generation tool.

For Word:
use create_word_document.

For Excel:
use create_excel_document.

For PowerPoint:
use create_pptx_document.

The environment is air-gapped.

Do not use the internet.

Do not invent facts.

Ground responses in the supplied context.
"""

# Keep the vision request prompt intentionally tiny. The full agent prompt is
# useful for tool/coding/RAG tasks but is unnecessary for direct image analysis.
VISION_SYSTEM_PROMPT = """You are a local vision assistant. Analyze the supplied image and answer the user's request accurately and concisely. Do not invent details that are not visible. Do not use external information."""


# =========================================================
# TOOL REGISTRY
# =========================================================

TOOL_BY_NAME = {
    tool.name: tool
    for tool in AGENT_TOOLS
}


# =========================================================
# URL
# =========================================================

def _normalise_base_url(
    api_base: str,
) -> str:

    return (
        api_base
        .rstrip("/")
        .removesuffix("/v1")
        .rstrip("/")
    )


# =========================================================
# WORKSPACE PATH
# =========================================================

def _resolve_workspace_path(
    relative_path: str,
) -> Path:

    raw = Path(relative_path)

    if raw.is_absolute():
        path = raw.resolve()
    else:
        path = (
            settings.BASE_DIR
            / raw
        ).resolve()

    workspace = (
        settings.WORKSPACE_DIR
        .resolve()
    )

    if (
        path != workspace
        and workspace not in path.parents
    ):
        raise ValueError(
            f"Path escapes workspace: "
            f"{relative_path}"
        )

    return path


# =========================================================
# IMAGE
# =========================================================

def _image_base64(
    relative_path: str,
) -> str:
    """Return a vision-optimized JPEG base64 payload.

    Qwen-VL latency grows with image token count. Large screenshots/photos are
    therefore resized before being sent to Ollama. This happens in memory, so
    there is no temporary converted image on disk.
    """

    path = _resolve_workspace_path(relative_path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"Image file not found: {path}"
        )

    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)

            # 896px is a strong latency/quality point for screenshots and
            # general image understanding. Avoid feeding 2K/4K pixels when
            # the task normally does not need them.
            max_side = 896
            if max(image.size) > max_side:
                image.thumbnail(
                    (max_side, max_side),
                    Image.Resampling.LANCZOS,
                )

            # JPEG is substantially smaller than PNG for photographic images
            # and large screenshots, reducing JSON serialization and HTTP
            # transfer overhead. Composite transparency onto white first.
            if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGB", rgba.size, "white")
                background.paste(
                    rgba,
                    mask=rgba.getchannel("A"),
                )
                image = background
            else:
                image = image.convert("RGB")

            import io
            buffer = io.BytesIO()
            image.save(
                buffer,
                format="JPEG",
                quality=82,
                optimize=False,
            )
            return base64.b64encode(
                buffer.getvalue()
            ).decode("ascii")

    except Exception as exc:
        logger.warning(
            "Vision image optimization failed for %s; using original bytes: %s",
            path,
            exc,
        )
        return base64.b64encode(
            path.read_bytes()
        ).decode("ascii")


# =========================================================
# PURIFIED CONTEXT
# =========================================================

def _purified_context(
    purified_json: Dict[str, Any],
) -> str:

    user_request = (
        purified_json.get(
            "user_request"
        )
        or {}
    )

    prompt = (
        user_request.get(
            "sanitized_prompt"
        )
        or user_request.get(
            "prompt"
        )
        or purified_json.get("prompt")
        or purified_json.get("sanitized_prompt")
        or purified_json.get("additionalProp1")
        or ""
    )

    # Swagger can send the request in a generic JSON field such as
    # `additionalProp1`. Keep the exact same fallback used by run_agent so
    # the worker receives the real prompt instead of an empty placeholder.
    if not prompt and isinstance(purified_json, dict):
        for value in purified_json.values():
            if isinstance(value, str) and value.strip():
                prompt = value.strip()
                break

    attachments = (
        purified_json.get(
            "attachments"
        )
        or []
    )

    # Deterministic RAG state. Internal-knowledge requests are retrieved here
    # before the worker LLM runs; the RAG tool is not exposed to that worker.
    knowledge_context = ""
    knowledge_results: list[dict[str, Any]] = []

    parts = [
        f"User request: {prompt}"
    ]

    for attachment in attachments:

        extracted = (
            attachment.get(
                "extracted_text"
            )
            or ""
        )

        parts.append(
            f"[Attachment: "
            f"{attachment.get('filename', 'unknown')} | "
            f"type="
            f"{attachment.get('file_type', 'unknown')}]\n"
            f"{extracted}"
        )

    return "\n\n".join(parts)


# =========================================================
# OLLAMA
# =========================================================

async def _ollama_chat(
    model: str,
    base_url: str,
    messages: list[dict[str, Any]],
    *,
    tools: Optional[
        list[dict[str, Any]]
    ] = None,
    temperature: float = 0.1,
    timeout: float = 360.0,
    keep_alive: str = "15m",
    num_predict: int = 512,
    num_ctx: Optional[int] = None,
) -> dict:

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
        "keep_alive": keep_alive,

        "options": {
            "num_predict": num_predict,
        },
    }

    if num_ctx:
        payload["options"]["num_ctx"] = num_ctx

    if tools:
        payload["tools"] = tools

    client = await _get_ollama_http_client()

    response = await client.post(
        f"{_normalise_base_url(base_url)}/api/chat",
        json=payload,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Ollama request failed "
            f"({response.status_code}): "
            f"{response.text[:2000]}"
        )

    data = response.json()

    if (
        not isinstance(data, dict)
        or not isinstance(
            data.get("message"),
            dict,
        )
    ):
        raise RuntimeError(
            "Ollama returned an invalid "
            f"response: {data!r}"
        )

    return data


# =========================================================
# TOOL SCHEMA
# =========================================================

def _tool_schema(
    tool: Any,
) -> dict[str, Any]:

    schema = None

    try:
        input_model = tool.get_input_schema()

        if hasattr(
            input_model,
            "model_json_schema",
        ):
            schema = (
                input_model
                .model_json_schema()
            )

        elif hasattr(
            input_model,
            "schema",
        ):
            schema = (
                input_model
                .schema()
            )

    except Exception:
        schema = None

    if not schema:

        args_schema = getattr(
            tool,
            "args_schema",
            None,
        )

        if args_schema is not None:

            if hasattr(
                args_schema,
                "model_json_schema",
            ):
                schema = (
                    args_schema
                    .model_json_schema()
                )

            elif hasattr(
                args_schema,
                "schema",
            ):
                schema = (
                    args_schema
                    .schema()
                )

    if not schema:
        schema = {
            "type": "object",
            "properties": {},
        }

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": (
                getattr(
                    tool,
                    "description",
                    "",
                )
                or ""
            ),
            "parameters": schema,
        },
    }


# =========================================================
# TOOL ARGUMENTS
# =========================================================

def _normalise_tool_arguments(
    arguments: Any,
) -> dict[str, Any]:

    if arguments is None:
        return {}

    if isinstance(
        arguments,
        dict,
    ):
        return arguments

    if isinstance(
        arguments,
        str,
    ):

        try:
            parsed = json.loads(
                arguments
            )

            if isinstance(
                parsed,
                dict,
            ):
                return parsed

        except json.JSONDecodeError:
            pass

    return {}


# =========================================================
# SERIALISE TOOL CALL
# =========================================================

def _serialise_tool_call(
    call: dict[str, Any],
) -> dict[str, Any]:

    function = (
        call.get("function")
        or {}
    )

    return {
        "type": call.get(
            "type",
            "function",
        ),
        "function": {
            "name": function.get(
                "name",
                "",
            ),
            "arguments":
                _normalise_tool_arguments(
                    function.get(
                        "arguments"
                    )
                ),
        },
    }


# =========================================================
# TEXTUAL TOOL CALL
# =========================================================

def _extract_textual_tool_call(
    content: str,
    available_tools: dict[str, Any],
) -> Optional[
    dict[str, Any]
]:

    if not content:
        return None

    candidates = [
        content.strip()
    ]

    candidates.extend(
        re.findall(
            r"```(?:json)?\s*(.*?)\s*```",
            content,
            flags=re.IGNORECASE
            | re.DOTALL,
        )
    )

    for candidate in candidates:

        try:
            parsed = json.loads(
                candidate
            )
        except Exception:
            continue

        if not isinstance(
            parsed,
            dict,
        ):
            continue

        name = (
            parsed.get("name")
            or parsed.get("tool")
            or parsed.get("tool_name")
        )

        arguments = (
            parsed.get("arguments")
            or parsed.get("args")
            or parsed.get("parameters")
            or {}
        )

        if name in available_tools:
            return {
                "name": name,
                "arguments":
                    _normalise_tool_arguments(
                        arguments
                    ),
            }

    # Some local Qwen/Ollama builds emit a Python-like function call instead
    # of native tool_calls or JSON. Parse the common search_knowledge_base
    # form so the call is actually executed instead of being shown to users.
    match = re.search(
        r"search_knowledge_base\s*\(\s*query\s*=\s*[\"'](.*?)[\"']\s*\)",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match and "search_knowledge_base" in available_tools:
        return {
            "name": "search_knowledge_base",
            "arguments": {"query": match.group(1).strip()},
        }

    return None


# =========================================================
# TOOL EXECUTION
# =========================================================

async def _execute_tool(
    tool: Any,
    arguments: dict[str, Any],
) -> str:

    # -----------------------------------------------------
    # Progress
    # -----------------------------------------------------

    if tool.name == "engineering_calculator":

        emit_event(
            EXECUTING_TOOL,
            "Running engineering calculation",
        )

    elif tool.name == "run_python_code":

        emit_event(
            RUNNING_PYTHON,
            "Executing Python code",
        )

    elif tool.name == "create_word_document":

        emit_event(
            CREATING_WORD_DOCUMENT,
            "Creating Word document",
        )

    elif tool.name == "create_excel_document":

        emit_event(
            CREATING_EXCEL_DOCUMENT,
            "Creating Excel spreadsheet",
        )

    elif tool.name == "create_pptx_document":

        emit_event(
            CREATING_POWERPOINT,
            "Creating PowerPoint presentation",
        )

    elif tool.name == "filesystem_list_directory":

        emit_event(
            EXECUTING_TOOL,
            "Listing local directory",
        )

    elif tool.name == "filesystem_search_files":

        emit_event(
            EXECUTING_TOOL,
            "Searching local files",
        )

    elif tool.name == "filesystem_get_file_info":

        emit_event(
            EXECUTING_TOOL,
            "Reading local file information",
        )

    elif tool.name == "filesystem_read_file":

        emit_event(
            EXECUTING_TOOL,
            "Reading local file",
        )

    else:

        emit_event(
            EXECUTING_TOOL,
            f"Executing {tool.name}",
        )

    # -----------------------------------------------------
    # Execute
    # -----------------------------------------------------

    try:

        result = await tool.ainvoke(
            arguments
        )

    except Exception as exc:

        logger.exception(
            "Tool %s failed",
            tool.name,
        )

        return (
            f"Tool execution error: "
            f"{type(exc).__name__}: {exc}"
        )

    if isinstance(
        result,
        str,
    ):
        return result

    try:
        return json.dumps(
            result,
            ensure_ascii=False,
            default=str,
        )

    except Exception:
        return str(result)


# =========================================================
# NATIVE TOOL AGENT
# =========================================================

async def _native_tool_agent(
    model: str,
    base_url: str,
    messages: list[dict[str, Any]],
    selected_tools: list[Any],
    *,
    max_rounds: int = 4,
) -> Tuple[
    str,
    list[dict[str, Any]],
    int,
    list[dict[str, Any]],
]:

    tool_schemas = [
        _tool_schema(tool)
        for tool in selected_tools
    ]

    available = {
        tool.name: tool
        for tool in selected_tools
    }

    history = [
        dict(message)
        for message in messages
    ]

    tool_calls_made = 0
    executed_calls = []

    for _ in range(max_rounds):

        response = await _ollama_chat(
            model,
            base_url,
            history,
            tools=tool_schemas,
            temperature=0.1,
            num_predict=512,
            num_ctx=4096,
        )

        assistant = response["message"]

        content = (
            assistant.get(
                "content"
            )
            or ""
        )

        assistant_message = {
            "role": "assistant",
            "content": content,
        }

        raw_calls = (
            assistant.get(
                "tool_calls"
            )
            or []
        )

        # -------------------------------------------------
        # Some Qwen builds return tool calls as JSON text.
        # -------------------------------------------------

        if (
            not raw_calls
            and content
        ):

            textual_call = (
                _extract_textual_tool_call(
                    content,
                    available,
                )
            )

            if textual_call:

                raw_calls = [
                    {
                        "type": "function",
                        "function": {
                            "name":
                                textual_call[
                                    "name"
                                ],
                            "arguments":
                                textual_call[
                                    "arguments"
                                ],
                        },
                    }
                ]

                assistant_message[
                    "content"
                ] = ""

        if raw_calls:

            assistant_message[
                "tool_calls"
            ] = [
                _serialise_tool_call(
                    call
                )
                for call in raw_calls
            ]

        history.append(
            assistant_message
        )

        if not raw_calls:

            return (
                content,
                history,
                tool_calls_made,
                executed_calls,
            )

        # -------------------------------------------------
        # Execute requested tools.
        # -------------------------------------------------

        for raw_call in raw_calls:

            function = (
                raw_call.get(
                    "function"
                )
                or {}
            )

            tool_name = (
                function.get(
                    "name"
                )
            )

            arguments = (
                _normalise_tool_arguments(
                    function.get(
                        "arguments"
                    )
                )
            )

            if not tool_name:
                continue

            tool_calls_made += 1

            if tool_name not in available:

                result = (
                    "Unknown tool requested: "
                    f"{tool_name}"
                )

            else:

                result = await _execute_tool(
                    available[
                        tool_name
                    ],
                    arguments,
                )

            executed_calls.append(
                {
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result,
                }
            )

            history.append(
                {
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": result,
                }
            )

    raise RuntimeError(
        "Worker exceeded maximum "
        f"tool rounds ({max_rounds})."
    )


# =========================================================
# DOCUMENT DETECTION
# =========================================================

def _detect_required_document_tool(
    prompt: str,
) -> Optional[str]:

    text = (
        prompt.lower()
        .strip()
    )

    if any(
        x in text
        for x in [
            "create excel",
            "create an excel",
            "generate excel",
            "generate an excel",
            "download excel",
            "spreadsheet",
            "excel file",
            "xlsx",
            "xlsx file",
        ]
    ):
        return "create_excel_document"

    if any(
        x in text
        for x in [
            "create powerpoint",
            "create a powerpoint",
            "generate powerpoint",
            "generate a powerpoint",
            "download powerpoint",
            "create ppt",
            "generate ppt",
            "pptx",
            "pptx file",
        ]
    ):
        return "create_pptx_document"

    if any(
        x in text
        for x in [
            "downloadable document",
            "download document",
            "downloadable word",
            "download word document",
            "create document",
            "create a document",
            "create word document",
            "create a word document",
            "generate document",
            "generate a document",
            "generate word document",
            "generate a word document",
            "make a document",
            "make a word document",
            "prepare document",
            "prepare a document",
            "prepare word document",
            "write a document",
            "write a word document",
            "export document",
            "save as word",
            "save as docx",
            "docx file",
            "word file",
            "convert to word",
            "convert to docx",
            "modify the document",
            "modify document",
            "edit the document",
            "edit document",
            "revise the document",
            "revise document",
            "generate from this document",
            "create from this document",
            "prepare from this document",
        ]
    ):
        return "create_word_document"

    return None


# =========================================================
# ENGINEERING DETECTION
# =========================================================

def _detect_engineering_request(
    prompt: str,
) -> bool:

    text = (
        prompt.lower()
        .strip()
    )

    signals = [
        "engineering calculation",
        "engineering calculations",
        "engineering problem",
        "pump power",
        "pump calculation",
        "pump calculations",
        "pressure drop",
        "pressure loss",
        "pipe pressure",
        "pipe velocity",
        "flow velocity",
        "reynolds number",
        "reynolds calculation",
        "heat duty",
        "heat transfer",
        "specific heat",
        "fluid mechanics",
        "fluid flow",
        "hydraulic calculation",
        "hydraulic calculations",
        "darcy weisbach",
        "darcy-weisbach",
        "friction factor",
        "flow rate",
        "pipe diameter",
        "pump head",
        "pump efficiency",
    ]

    has_numeric_input = bool(
        re.search(
            r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:\s*[A-Za-z%°²³^/*.-]+)?",
            text,
        )
    )

    calculation_intent = any(
        phrase in text
        for phrase in [
            "calculate",
            "calculation",
            "find the",
            "determine",
            "compute",
            "work out",
            "solve",
            "maximum bending",
            "pressure drop",
            "pressure loss",
            "pump power",
            "heat transfer",
            "heat duty",
            "reynolds number",
        ]
    )

    domain_signal = any(signal in text for signal in signals)
    return bool(domain_signal and (calculation_intent or has_numeric_input))


# =========================================================
# CODE EXECUTION
# =========================================================

def _detect_code_execution(
    prompt: str,
) -> bool:

    text = (
        prompt.lower()
        .strip()
    )

    execution_phrases = [
        "run this",
        "run the code",
        "run code",
        "execute this",
        "execute the code",
        "execute code",
        "test this code",
        "test the code",
        "test code",
        "execute this program",
        "run this program",
        "run the program",
    ]

    return any(
        x in text
        for x in execution_phrases
    )


# =========================================================
# ACTUAL CODING DETECTION
# =========================================================

def _detect_coding_request(
    prompt: str,
) -> bool:

    text = (
        prompt.lower()
        .strip()
    )

    strong_signals = [
        "write code",
        "write a program",
        "generate code",
        "generate a program",
        "write python",
        "write javascript",
        "write typescript",
        "write java",
        "write c++",
        "write c code",
        "debug this",
        "debug the code",
        "debug code",
        "fix this code",
        "fix the code",
        "review this code",
        "review the code",
        "code review",
        "implement this",
        "implement the function",
        "implement a function",
        "create a function",
        "create a class",
        "write a function",
        "write a class",
        "build the code",
        "modify this code",
        "change this code",
        "complete this code",
        "finish this code",
        "sql query",
    ]

    if any(
        signal in text
        for signal in strong_signals
    ):
        return True

    coding_words = [
        "code",
        "coding",
        "program",
        "programming",
        "script",
        "algorithm",
    ]

    action_words = [
        "write",
        "create",
        "generate",
        "implement",
        "build",
        "fix",
        "debug",
        "modify",
        "complete",
        "finish",
    ]

    return (
        any(
            word in text
            for word in coding_words
        )
        and
        any(
            word in text
            for word in action_words
        )
    )
def _detect_internal_knowledge_request(prompt: str) -> bool:
    text = prompt.lower().strip()

    explicit_signals = [
        "according to the knowledge base",
        "according to knowledge base",
        "from the knowledge base",
        "in the knowledge base",
        "according to our sop",
        "according to the sop",
        "according to our manual",
        "according to the manual",
        "according to our policy",
        "company policy",
        "internal document",
        "internal documents",
        "organizational knowledge",
        "past correspondence",
    ]

    if any(signal in text for signal in explicit_signals):
        return True

    organization_signals = [
        "mrpl",
        "mangalore refinery",
        "mangalore refining",
    ]

    knowledge_signals = [
        "vision",
        "mission",
        "policy",
        "policies",
        "sop",
        "manual",
        "procedure",
        "procedures",
        "guideline",
        "guidelines",
        "standard",
        "standards",
        "correspondence",
        "circular",
        "regulation",
        "regulations",
        "history",
        "objective",
        "objectives",
        "organization",
        "company",
    ]

    has_organization = any(signal in text for signal in organization_signals)
    has_knowledge_topic = any(signal in text for signal in knowledge_signals)

    # Internal/company references are frequently implicit. Requiring both
    # "MRPL" and a knowledge keyword caused valid questions such as
    # "What is our company vision?" to bypass RAG.
    first_person_org = bool(
        re.search(
            r"\\b(our|we|us|company's|organisation's|organization's)\\b",
            text,
        )
    )
    internal_document_language = bool(
        re.search(
            r"\\b(document|documents|file|files|report|reports|sop|manual|policy|policies)\\b",
            text,
        )
    )

    if has_organization:
        return True

    if has_knowledge_topic and first_person_org:
        return True

    if has_knowledge_topic and internal_document_language:
        return True

    return False

# =========================================================
# DEFAULT FILENAMES
# =========================================================

def _default_filename(
    tool_name: str,
) -> str:

    return {
        "create_word_document":
            "generated_document.docx",

        "create_excel_document":
            "generated_spreadsheet.xlsx",

        "create_pptx_document":
            "generated_presentation.pptx",

    }.get(
        tool_name,
        "generated_file",
    )


# =========================================================
# GENERATED FILES
# =========================================================

def _collect_generated_files(
    executed_calls:
        list[dict[str, Any]],
) -> list[dict[str, str]]:

    generated = []

    document_tools = {
        "create_word_document",
        "create_excel_document",
        "create_pptx_document",
    }

    for call in executed_calls:

        tool_name = call.get(
            "tool"
        )

        if tool_name not in document_tools:
            continue

        result = call.get(
            "result"
        )

        if not result:
            continue

        candidates = [
            Path(str(result)),
            settings.BASE_DIR / str(result),
            settings.WORKSPACE_DIR / str(result),
        ]

        existing = None

        for candidate in candidates:

            try:
                candidate = candidate.resolve()
            except Exception:
                continue

            workspace = (
                settings.WORKSPACE_DIR
                .resolve()
            )

            if (
                candidate.is_file()
                and workspace in candidate.parents
            ):
                existing = candidate
                break

        if existing is None:
            continue

        try:

            relative = (
                existing
                .relative_to(
                    settings
                    .WORKSPACE_DIR
                    .resolve()
                )
                .as_posix()
            )

        except ValueError:
            continue

        item = {
            "tool": tool_name,
            "path": relative,
            "filename": existing.name,
        }

        if item not in generated:
            generated.append(item)

    return generated


# =========================================================
# PYTHON EXECUTION RESULTS
# =========================================================

def _extract_execution_results(
    executed_calls:
        list[dict[str, Any]],
) -> list[dict[str, Any]]:

    results = []

    for call in executed_calls:

        if (
            call.get("tool")
            != "run_python_code"
        ):
            continue

        results.append(
            {
                "tool": "run_python_code",
                "arguments": call.get(
                    "arguments",
                    {},
                ),
                "result": call.get(
                    "result",
                    "",
                ),
            }
        )

    return results


# =========================================================
# ENGINEERING FALLBACK
# =========================================================

async def _engineering_fallback(
    purified_text: str,
) -> dict[str, Any]:

    tool = TOOL_BY_NAME.get(
        "engineering_calculator"
    )

    if tool is None:
        raise RuntimeError(
            "engineering_calculator "
            "tool is not registered."
        )

    emit_event(
        EXECUTING_TOOL,
        "Running engineering calculation",
    )

    arguments = {
        "calculation": "auto",
        "parameters": purified_text,
    }

    result = await tool.ainvoke(
        arguments
    )

    if not isinstance(
        result,
        str,
    ):
        result = json.dumps(
            result,
            ensure_ascii=False,
            default=str,
        )

    return {
        "tool": "engineering_calculator",
        "arguments": arguments,
        "result": result,
    }


# =========================================================
# DOCUMENT FALLBACK
# =========================================================

async def _document_fallback(
    model: str,
    base_url: str,
    purified_text: str,
    tool_name: str,
) -> tuple[
    str,
    dict[str, Any],
]:

    instructions = {

        "create_word_document": (
            "Return ONLY the document body.\n"
            "Use '# ' for the title.\n"
            "Use '## ' for headings.\n"
            "Use normal lines for paragraphs."
        ),

        "create_excel_document": (
            "Return ONLY valid CSV data.\n"
            "The first row must be the header."
        ),

        "create_pptx_document": (
            "Return ONLY presentation markup.\n"
            "Use '# ' for slide titles.\n"
            "Use '- ' for bullets."
        ),
    }

    messages = [
        {
            "role": "system",
            "content": (
                "Create the requested "
                "deliverable from the supplied "
                "request and attachment context.\n"
                "Do not invent information.\n\n"
                + instructions[tool_name]
            ),
        },
        {
            "role": "user",
            "content": purified_text,
        },
    ]

    response = await _ollama_chat(
        model,
        base_url,
        messages,
        temperature=0.0,
        keep_alive="15m",
        num_predict=512,
        num_ctx=3072,
    )

    content = (
        response["message"]
        .get("content")
        or ""
    ).strip()

    if not content:
        raise RuntimeError(
            "Worker returned no "
            "document content."
        )

    filename = _default_filename(
        tool_name
    )

    if (
        tool_name
        == "create_excel_document"
    ):

        arguments = {
            "filename": filename,
            "csv_data": content,
        }

    else:

        arguments = {
            "filename": filename,
            "content": content,
        }

    tool = TOOL_BY_NAME[
        tool_name
    ]

    result = await _execute_tool(
        tool,
        arguments,
    )

    return content, {
        "tool": tool_name,
        "arguments": arguments,
        "result": result,
    }


# =========================================================
# DETERMINISTIC KNOWLEDGE RETRIEVAL
# =========================================================

async def _retrieve_knowledge_context(
    prompt: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Retrieve trusted organization context before the worker LLM runs."""
    emit_event(
        EXECUTING_TOOL,
        "Searching local knowledge base",
    )

    try:
        results = await rag_engine.retrieve(
            prompt,
            top_k=5,
        )
    except Exception as exc:
        logger.exception("Knowledge-base retrieval failed")
        # Keep the API request alive so the frontend receives a useful response
        # instead of the generic "local AI could not complete" error.
        return (
            "The local knowledge base could not be searched because "
            f"retrieval failed: {type(exc).__name__}: {exc}",
            [],
        )

    context = rag_engine.format_context(results)
    return context, results



# =========================================================
# DIRECT DOCUMENT GENERATION HELPERS
# =========================================================

def _extract_latest_assistant_response(
    attachments: list[dict[str, Any]],
) -> str:
    """Extract the latest previous assistant answer from chat history."""
    history_text = ""

    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        if not attachment.get("conversation_context"):
            continue
        candidate = attachment.get("extracted_text") or ""
        if candidate:
            history_text = str(candidate)
            break

    if not history_text:
        return ""

    matches = re.findall(
        r"(?:^|\n)ASSISTANT:\s*\n(.*?)(?=\n(?:USER|ASSISTANT):|\Z)",
        history_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return str(matches[-1]).strip() if matches else ""


def _is_previous_response_request(prompt: str) -> bool:
    """Detect explicit requests to export the previous assistant answer."""
    text = str(prompt or "").lower().strip()

    signals = (
        "this response",
        "this answer",
        "above response",
        "above answer",
        "previous response",
        "previous answer",
        "last response",
        "last answer",
        "the response above",
        "the answer above",
        "convert this response",
        "convert this answer",
        "make this response",
        "make this answer",
        "generate this response",
        "generate this answer",
        "create this response",
        "create this answer",
        "export this response",
        "export this answer",
        "turn this response",
        "turn this answer",
    )

    return any(signal in text for signal in signals)


def _response_to_excel_csv(content: str) -> str:
    """Convert a previous answer to valid CSV without another LLM call."""
    import csv
    import io

    text = str(content or "").strip()

    if not text:
        return "Content\n"

    table_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("|")
        and line.strip().endswith("|")
    ]

    if len(table_lines) >= 2:
        rows = []

        for line in table_lines:
            cells = [
                cell.strip()
                for cell in line.strip().strip("|").split("|")
            ]

            if cells and all(
                re.fullmatch(r":?-{3,}:?", cell or "")
                for cell in cells
            ):
                continue

            rows.append(cells)

        if rows:
            width = max(len(row) for row in rows)
            rows = [
                row + [""] * (width - len(row))
                for row in rows
            ]

            buffer = io.StringIO()
            csv.writer(buffer).writerows(rows)
            return buffer.getvalue()

    rows = [
        [line.strip()]
        for line in text.splitlines()
        if line.strip()
    ]

    if not rows:
        rows = [[""]]

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Content"])
    writer.writerows(rows)
    return buffer.getvalue()


def _response_to_pptx_markup(content: str) -> str:
    """Create basic PPT markup from an existing answer without an LLM call."""
    text = str(content or "").strip()

    if not text:
        return "# Generated Presentation\n- No content provided."

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    title = "Generated Presentation"
    if lines:
        title = re.sub(r"^#+\s*", "", lines[0]).strip()[:100] or title

    output = [f"# {title}"]
    body_lines = []

    for line in lines[1:]:
        clean = re.sub(r"^[-*•]\s*", "", line)
        clean = re.sub(r"^\d+[.)]\s*", "", clean)
        clean = re.sub(r"^#+\s*", "", clean).strip()
        if clean:
            body_lines.append(clean)

    if body_lines:
        output.extend(f"- {line}" for line in body_lines[:25])
    else:
        output.append("- Generated from the previous response.")

    return "\n".join(output)


async def _direct_create_document(
    tool_name: str,
    content: str,
) -> dict[str, Any]:
    """Create a deliverable directly, bypassing the LLM tool loop."""
    filename = _default_filename(tool_name)

    if tool_name == "create_excel_document":
        arguments = {
            "filename": filename,
            "csv_data": _response_to_excel_csv(content),
        }
    elif tool_name == "create_pptx_document":
        arguments = {
            "filename": filename,
            "content": _response_to_pptx_markup(content),
        }
    else:
        arguments = {
            "filename": filename,
            "content": str(content).strip(),
        }

    tool = TOOL_BY_NAME.get(tool_name)
    if tool is None:
        raise RuntimeError(
            f"Document tool '{tool_name}' is not registered."
        )

    result = await _execute_tool(tool, arguments)

    return {
        "tool": tool_name,
        "arguments": arguments,
        "result": result,
    }



def _detect_filesystem_request(prompt: str) -> bool:
    """Fast local detection for pure filesystem operations."""
    text = str(prompt or "").lower().strip()
    signals = (
        "filesystem",
        "file system",
        "local files",
        "local file",
        "local folder",
        "local directory",
        "list files",
        "list the files",
        "show files",
        "show the files",
        "what files are in",
        "what is in this folder",
        "what's in this folder",
        "folder contents",
        "directory contents",
        "browse folder",
        "browse directory",
        "go into",
        "go to folder",
        "open folder",
        "enter folder",
        "navigate to folder",
        "find file",
        "find the file",
        "find files",
        "search for file",
        "search for files",
        "locate file",
        "read file",
        "read the file",
        "open file",
        "open the file",
        "file information",
        "file info",
        "file metadata",
        "file properties",
        "go back",
        "parent folder",
        "parent directory",
        "current folder",
        "current directory",
        "list here",
        "show here",
    )
    if any(signal in text for signal in signals):
        return True
    return bool(
        re.search(r"\b[A-Za-z]:[\\/]", prompt or "")
        or re.search(r"(?<!\w)/(?:[^/\s]+/)+[^/\s]*", prompt or "")
    )


def _is_pure_filesystem_operation(prompt: str) -> bool:
    """Avoid intercepting requests that need an LLM after reading a file."""
    text = str(prompt or "").lower().strip()

    if not _detect_filesystem_request(text):
        return False

    # These verbs mean the user wants the contents transformed/analyzed by
    # the worker after filesystem access, so keep the normal tool-agent path.
    compound_actions = (
        "summarize",
        "summary",
        "analyze",
        "analyse",
        "explain",
        "translate",
        "rewrite",
        "review",
        "compare",
        "calculate",
        "extract the data",
        "extract data",
        "answer questions from",
        "tell me what it says",
        "what does the file say",
        "based on the file",
        "using the file",
    )
    return not any(action in text for action in compound_actions)


# =========================================================
# MAIN AGENT
# =========================================================

async def run_agent(
    purified_json: Dict[str, Any],
) -> Dict[str, Any]:

    user_request = (
        purified_json.get(
            "user_request"
        )
        or {}
    )

    prompt = (
        user_request.get(
            "sanitized_prompt"
        )
        or user_request.get(
            "prompt"
        )
        or purified_json.get("prompt")
        or purified_json.get("sanitized_prompt")
        or purified_json.get("additionalProp1")
        or ""
    )

    # Swagger may expose the generic JSON field as `additionalProp1`.
    # Accept it for direct Swagger requests so the actual user question is
    # not silently lost before routing.
    if not prompt and isinstance(purified_json, dict):
        for value in purified_json.values():
            if isinstance(value, str) and value.strip():
                prompt = value.strip()
                break

    attachments = (
        purified_json.get(
            "attachments"
        )
        or []
    )

    # Always initialize these so every request has a predictable result shape.
    knowledge_context = ""
    knowledge_results: list[dict[str, Any]] = []

    attachment_types = [
        a.get(
            "file_type",
            "unknown",
        )
        for a in attachments
    ]

    # =====================================================
    # 1. DIRECT VISION FAST PATH
    # =====================================================
    # A real image attachment is an unambiguous vision signal. Do not spend
    # another LLM call asking the 0.5B router to classify something we already
    # know from the attachment metadata.
    actual_images = [
        a
        for a in attachments
        if str(a.get("file_type", "")).lower() == "image"
    ]

    required_document_tool = _detect_required_document_tool(prompt)
    engineering_requested = _detect_engineering_request(prompt)
    filesystem_requested = _is_pure_filesystem_operation(prompt)

    # =====================================================
    # 2. DIRECT FILESYSTEM FAST PATH
    # =====================================================
    # Basic local filesystem navigation is deterministic. Do not spend a
    # 0.5B router call or a 7B tool-agent round on commands such as:
    # "list Documents", "go into Projects", "go back", or "find report.pdf".
    # The filesystem module maintains a per-conversation navigation directory.
    if (
        filesystem_requested
        and not required_document_tool
        and not engineering_requested
        and not actual_images
    ):
        metadata = purified_json.get("metadata", {}) or {}
        state_key = (
            str(metadata.get("conversation_id") or "")
            or str(metadata.get("session_id") or "")
            or "default"
        )

        emit_event(
            ROUTING_REQUEST,
            "Filesystem request detected; using direct local filesystem",
        )
        emit_event(
            ROUTE_SELECTED,
            "LOCAL_FILESYSTEM",
        )
        emit_event(
            MODEL_SELECTED,
            "No model required",
        )

        result = execute_filesystem_prompt(
            prompt,
            state_key=state_key,
        )

        if not result:
            raise RuntimeError(
                "The filesystem request could not be resolved."
            )

        emit_event(
            GENERATING_RESPONSE,
            "Returning local filesystem result",
        )
        emit_event(
            COMPLETED,
            "Processing completed",
        )

        return {
            "category": "TASK_DOCUMENT_SUMMARY",
            "model_used": "local-filesystem",
            "selected_tools": ["filesystem"],
            "fallback_used": False,
            "final_answer": result.strip(),
            "tool_calls_made": 1,
            "message_count": 2,
            "generated_files": [],
            "execution_results": [
                {
                    "tool": "filesystem",
                    "arguments": {"prompt": prompt},
                    "result": result,
                }
            ],
            "knowledge_results": [],
            "filesystem_fast_path": True,
        }

    # =====================================================
    # 3. DIRECT DOCUMENT FAST PATH
    # =====================================================
    # Document intent is deterministic. Skip the general router and the
    # native tool-agent loop. If the user refers to the previous answer,
    # convert that answer directly with zero additional LLM inference.
    if required_document_tool:
        previous_response = _extract_latest_assistant_response(
            attachments
        )

        if (
            previous_response
            and _is_previous_response_request(prompt)
        ):
            emit_event(
                ROUTING_REQUEST,
                "Document request detected; using direct local generation",
            )

            tool_label = {
                "create_word_document": "Word document",
                "create_excel_document": "Excel spreadsheet",
                "create_pptx_document": "PowerPoint presentation",
            }.get(
                required_document_tool,
                "document",
            )

            emit_event(
                ROUTE_SELECTED,
                f"Direct {tool_label} generation",
            )

            executed_call = await _direct_create_document(
                required_document_tool,
                previous_response,
            )

            generated_files = _collect_generated_files(
                [executed_call]
            )

            emit_event(
                GENERATING_RESPONSE,
                "Preparing downloadable document",
            )

            emit_event(
                COMPLETED,
                "Processing completed",
            )

            return {
                "category": "TASK_DOCUMENT_SUMMARY",
                "model_used": "local-document-generator",
                "selected_tools": [required_document_tool],
                "fallback_used": False,
                "final_answer": (
                    "The requested "
                    f"{tool_label.lower()} has been created successfully."
                ),
                "tool_calls_made": 1,
                "message_count": 2,
                "generated_files": generated_files,
                "execution_results": [],
                "knowledge_results": [],
                "document_fast_path": True,
            }

        # Standalone document requests still need content generation, but
        # only one focused model call is used. The 0.5B router and native
        # tool-agent loop are skipped.
        emit_event(
            ROUTING_REQUEST,
            "Document request detected; bypassing general router",
        )

        registry = settings.load_model_registry()
        task_routes = registry.get("task_routes", {})
        route_config = task_routes.get(
            "TASK_DOCUMENT_SUMMARY"
        )

        if route_config is None:
            raise RuntimeError(
                "No TASK_DOCUMENT_SUMMARY route is configured."
            )

        selected_model = route_config["primary_model"]
        base_url = _normalise_base_url(
            route_config["api_base"]
        )

        emit_event(
            ROUTE_SELECTED,
            "Request routed to TASK_DOCUMENT_SUMMARY",
        )
        emit_event(
            MODEL_SELECTED,
            f"Using {selected_model}",
        )

        _, executed_call = await _document_fallback(
            selected_model,
            base_url,
            _purified_context(purified_json),
            required_document_tool,
        )

        generated_files = _collect_generated_files(
            [executed_call]
        )

        emit_event(
            COMPLETED,
            "Processing completed",
        )

        return {
            "category": "TASK_DOCUMENT_SUMMARY",
            "model_used": selected_model,
            "selected_tools": [required_document_tool],
            "fallback_used": False,
            "final_answer": (
                "The requested document has been created successfully."
            ),
            "tool_calls_made": 1,
            "message_count": 2,
            "generated_files": generated_files,
            "execution_results": [],
            "knowledge_results": [],
            "document_fast_path": True,
        }

    # =====================================================
    # 3. DIRECT ENGINEERING FAST PATH
    # =====================================================
    # Engineering arithmetic is deterministic. Skip both the 0.5B router and
    # the 7B native tool-agent loop. This removes an unnecessary inference
    # round and prevents model-dependent tool-call formatting from breaking a
    # calculation.
    #
    # File-backed engineering requests stay on the normal agent path because
    # they may need filesystem_read_file before calculating.
    engineering_has_filesystem_reference = (
        bool(re.search(r"[A-Za-z]:\\", prompt))
        or bool(re.search(r"(?<!\w)/(?:[^/\s]+/)+[^/\s]*", prompt))
        or any(
            word in prompt.lower()
            for word in [
                "read file",
                "read the file",
                "from my computer",
                "from my pc",
                "from my laptop",
                "local file",
                "local folder",
                "filesystem",
            ]
        )
    )

    if engineering_requested and not required_document_tool and not engineering_has_filesystem_reference:
        emit_event(
            ROUTING_REQUEST,
            "Engineering calculation detected; using deterministic local calculator",
        )
        emit_event(
            ROUTE_SELECTED,
            "TASK_ENGINEERING_CALCULATION",
        )
        emit_event(
            MODEL_SELECTED,
            "Deterministic local engineering calculator",
        )

        tool = TOOL_BY_NAME.get("engineering_calculator")
        if tool is None:
            raise RuntimeError("engineering_calculator tool is not registered.")

        arguments = {
            "calculation": "auto",
            "parameters": _purified_context(purified_json),
        }

        try:
            result = await _execute_tool(tool, arguments)
        except Exception as exc:
            logger.exception("Deterministic engineering calculation failed")
            result = f"Engineering calculation error: {type(exc).__name__}: {exc}"

        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False, default=str)

        emit_event(
            GENERATING_RESPONSE,
            "Returning deterministic engineering result",
        )
        emit_event(
            COMPLETED,
            "Processing completed",
        )

        return {
            "category": "TASK_ENGINEERING_CALCULATION",
            "model_used": "deterministic-engineering-calculator",
            "selected_tools": ["engineering_calculator"],
            "fallback_used": False,
            "final_answer": result.strip(),
            "tool_calls_made": 1,
            "message_count": 2,
            "generated_files": [],
            "execution_results": [],
            "knowledge_results": [],
            "engineering_fast_path": True,
            "engineering_deterministic": True,
        }

    if actual_images and not required_document_tool and not engineering_requested:
        emit_event(
            ROUTING_REQUEST,
            "Detected image attachment; bypassing text router",
        )

        registry = settings.load_model_registry()
        task_routes = registry.get("task_routes", {})
        route_config = task_routes.get("TASK_MULTIMODAL_VISION")

        if route_config is None:
            raise RuntimeError(
                "No TASK_MULTIMODAL_VISION route is configured."
            )

        selected_model = route_config["primary_model"]
        base_url = _normalise_base_url(route_config["api_base"])

        emit_event(
            ROUTE_SELECTED,
            "Request routed to TASK_MULTIMODAL_VISION",
        )
        emit_event(
            MODEL_SELECTED,
            f"Using {selected_model}",
        )

        # Keep the vision prompt minimal. Attachment metadata/extracted OCR and
        # long conversation-history text are unnecessary because the image
        # itself is supplied directly to Qwen-VL.
        vision_prompt = (
            str(prompt).strip()
            or "Analyze the attached image and describe the important details."
        )

        message = {
            "role": "user",
            "content": vision_prompt,
        }

        images = []
        for attachment in actual_images:
            local_path = attachment.get("local_path")
            if local_path:
                images.append(_image_base64(local_path))

        if not images:
            raise RuntimeError("No usable image attachment was found.")

        message["images"] = images

        emit_event(
            GENERATING_RESPONSE,
            "Generating response from vision model",
        )

        response = await _ollama_chat(
            selected_model,
            base_url,
            [
                {
                    "role": "system",
                    "content": VISION_SYSTEM_PROMPT,
                },
                message,
            ],
            temperature=0.0,
            keep_alive="15m",
            num_predict=(
                160
                if len(vision_prompt) < 180
                and not any(
                    word in vision_prompt.lower()
                    for word in ("detailed", "in detail", "everything", "all details")
                )
                else 256
            ),
            num_ctx=2048,
        )

        final_answer = (
            response.get("message", {}).get("content", "")
            or "No response was returned by the vision model."
        ).strip()

        emit_event(
            COMPLETED,
            "Processing completed",
        )

        return {
            "category": "TASK_MULTIMODAL_VISION",
            "model_used": selected_model,
            "selected_tools": [],
            "fallback_used": False,
            "final_answer": final_answer,
            "tool_calls_made": 0,
            "message_count": 2,
            "generated_files": [],
            "execution_results": [],
            "knowledge_results": [],
            "vision_fast_path": True,
        }

    # =====================================================
    # 2. ROUTING
    # =====================================================

    emit_event(
        ROUTING_REQUEST,
        "Routing request to task category",
    )

    route = await router_service.classify_and_route(
        prompt,
        attachment_types,
    )

    emit_event(
        ROUTE_SELECTED,
        f"Request routed to "
        f"{route['category']}",
    )

    # =====================================================
    # FAST PATH - ROUTER ANSWERS SIMPLE QUESTIONS
    # =====================================================
    # Simple, tool-free requests are answered directly by the lightweight
    # router. This completely removes the second (7B) inference call.
    # The router service already applies deterministic safety gates before
    # setting fast_path=True.
    if route.get("fast_path") and route.get("direct_answer"):
        router_model = (
            route.get("router_model")
            or "router"
        )

        emit_event(
            MODEL_SELECTED,
            f"Using {router_model} (fast path)",
        )

        emit_event(
            GENERATING_RESPONSE,
            "Returning fast-path response",
        )

        final_answer = str(
            route["direct_answer"]
        ).strip()

        emit_event(
            COMPLETED,
            "Processing completed",
        )

        return {
            "category": route["category"],
            "model_used": router_model,
            "selected_tools": [],
            "fallback_used": route.get(
                "fallback_used",
                False,
            ),
            "final_answer": final_answer,
            "tool_calls_made": 0,
            "message_count": 2,
            "generated_files": [],
            "execution_results": [],
            "knowledge_results": [],
            "fast_path": True,
        }

    registry = settings.load_model_registry()

    task_routes = registry.get(
        "task_routes",
        {},
    )

    route_config = task_routes.get(
        route["category"]
    )

    if route_config is None:

        route_config = task_routes.get(
            "TASK_DOCUMENT_SUMMARY"
        )

        if route_config is None:
            raise RuntimeError(
                "No worker route available."
            )

    selected_model = route_config[
        "primary_model"
    ]

    base_url = _normalise_base_url(
        route_config["api_base"]
    )

    # =====================================================
    # 2. HARD DETECTION
    # =====================================================

    required_document_tool = (
        _detect_required_document_tool(
            prompt
        )
    )

    engineering_requested = (
        _detect_engineering_request(
            prompt
        )
    )

    code_execution_requested = (
        _detect_code_execution(
            prompt
        )
    )

    coding_requested = (
        _detect_coding_request(
            prompt
        )
    )

    internal_knowledge_requested = _detect_internal_knowledge_request(
    prompt
 )
    # 3. INTERNAL KNOWLEDGE / RAG
    # =====================================================
    # Explicit knowledge-base requests always use the text worker and RAG.
    # This prevents a word such as "vision" from selecting the multimodal
    # vision model.
    if internal_knowledge_requested:
        route["category"] = "TASK_DOCUMENT_SUMMARY"
        route["selected_tools"] = []
        route_config = task_routes.get("TASK_DOCUMENT_SUMMARY")
        selected_model = route_config["primary_model"]
        base_url = _normalise_base_url(route_config["api_base"])

        knowledge_context, knowledge_results = (
            await _retrieve_knowledge_context(prompt)
        )

    # =====================================================
    # 4. ENGINEERING
    # =====================================================

    elif (
        engineering_requested
        and not required_document_tool
    ):

        route["category"] = (
            "TASK_ENGINEERING_CALCULATION"
        )

        route["selected_tools"] = [
            "engineering_calculator"
        ]

        filesystem_reference = (
            bool(
                re.search(
                    r"[A-Za-z]:\\",
                    prompt,
                )
            )
            or
            bool(
                re.search(
                    r"(?<!\w)/(?:[^/\s]+/)+[^/\s]*",
                    prompt,
                )
            )
            or
            any(
                word in prompt.lower()
                for word in [
                    "read file",
                    "read the file",
                    "from my computer",
                    "from my pc",
                    "from my laptop",
                    "local file",
                    "local folder",
                    "filesystem",
                ]
            )
        )

        if filesystem_reference:

            route["selected_tools"] = [
                "filesystem_read_file",
                "engineering_calculator",
            ]

        route_config = task_routes.get(
            "TASK_ENGINEERING_CALCULATION"
        )

        if route_config is None:
            route_config = task_routes.get(
                "TASK_DOCUMENT_SUMMARY"
            )

        selected_model = route_config[
            "primary_model"
        ]

        base_url = _normalise_base_url(
            route_config["api_base"]
        )

        code_execution_requested = False

    # =====================================================
    # 4. DOCUMENT
    # =====================================================

    elif required_document_tool:

        route["category"] = (
            "TASK_DOCUMENT_SUMMARY"
        )

        route["selected_tools"] = [
            required_document_tool
        ]

        route_config = task_routes.get(
            "TASK_DOCUMENT_SUMMARY"
        )

        selected_model = route_config[
            "primary_model"
        ]

        base_url = _normalise_base_url(
            route_config["api_base"]
        )

    # =====================================================
    # 5. ACTUAL CODING
    # =====================================================

    elif (
        coding_requested
        and route["category"]
        != "TASK_MULTIMODAL_VISION"
    ):

        route["category"] = (
            "TASK_CODING"
        )

        route_config = task_routes.get(
            "TASK_CODING"
        )

        selected_model = route_config[
            "primary_model"
        ]

        base_url = _normalise_base_url(
            route_config["api_base"]
        )

        if code_execution_requested:

            route["selected_tools"] = [
                "run_python_code"
            ]

        else:

            route["selected_tools"] = []

    # =====================================================
    # MODEL EVENT
    # =====================================================

    emit_event(
        MODEL_SELECTED,
        f"Using {selected_model}",
    )

    # =====================================================
    # 6. VISION
    # =====================================================

    actual_images = [
        a
        for a in attachments
        if str(
            a.get(
                "file_type",
                "",
            )
        ).lower()
        == "image"
    ]

    if (
        actual_images
        and not required_document_tool
        and not engineering_requested
    ):

        route["category"] = (
            "TASK_MULTIMODAL_VISION"
        )

        route_config = task_routes.get(
            "TASK_MULTIMODAL_VISION"
        )

        selected_model = route_config[
            "primary_model"
        ]

        base_url = _normalise_base_url(
            route_config["api_base"]
        )

        emit_event(
            MODEL_SELECTED,
            f"Using {selected_model}",
        )

        message = {
            "role": "user",
            "content": _purified_context(
                purified_json
            ),
        }

        images = []

        for attachment in actual_images:

            local_path = attachment.get(
                "local_path"
            )

            if local_path:

                images.append(
                    _image_base64(
                        local_path
                    )
                )

        if images:
            message["images"] = images

        emit_event(
            GENERATING_RESPONSE,
            "Generating response from vision model",
        )

        response = await _ollama_chat(
            selected_model,
            base_url,
            [
                {
                    "role": "system",
                    "content":
                        AGENT_SYSTEM_PROMPT,
                },
                message,
            ],
            temperature=0.0,
            keep_alive="15m",
            num_predict=256,
            num_ctx=2048,
        )

        emit_event(
            COMPLETED,
            "Processing completed",
        )

        return {
            "category":
                route["category"],

            "model_used":
                selected_model,

            "selected_tools":
                [],

            "fallback_used":
                route.get(
                    "fallback_used",
                    False,
                ),

            "final_answer":
                (
                    response["message"]
                    .get("content")
                    or ""
                ),

            "tool_calls_made":
                0,

            "message_count":
                3,

            "generated_files":
                [],

            "execution_results":
                [],
        }

    # =====================================================
    # 7. SELECT TOOLS
    # =====================================================
    #
    # Keep tools selected by the router, then add deterministic tools required
    # by the request. This is important for combined tasks such as:
    # "Use our SOP and create a Word document" -> RAG + Word tool.
    #
    selected_tool_names = [
        name
        for name in route.get(
            "selected_tools",
            [],
        )
        if name in TOOL_BY_NAME
    ]

    def add_tool(name: str) -> None:
        if name in TOOL_BY_NAME and name not in selected_tool_names:
            selected_tool_names.append(name)

    if engineering_requested:
        add_tool("engineering_calculator")

        filesystem_reference = (
            bool(
                re.search(
                    r"[A-Za-z]:\\",
                    prompt,
                )
            )
            or bool(
                re.search(
                    r"(?<!\w)/(?:[^/\s]+/)+[^/\s]*",
                    prompt,
                )
            )
            or any(
                word in prompt.lower()
                for word in [
                    "read file",
                    "read the file",
                    "from my computer",
                    "from my pc",
                    "from my laptop",
                    "local file",
                    "local folder",
                    "filesystem",
                ]
            )
        )

        if filesystem_reference:
            add_tool("filesystem_read_file")

    if required_document_tool:
        add_tool(required_document_tool)

    if code_execution_requested:
        add_tool("run_python_code")

    # Internal knowledge is retrieved deterministically above and injected into
    # the worker prompt as trusted context. Do NOT expose search_knowledge_base
    # to the worker here; doing so would send the request through the ReAct tool
    # loop and can leave the frontend waiting at a tool-call stage.
    selected_tool_objects = [
        TOOL_BY_NAME[name]
        for name in selected_tool_names
        if name in TOOL_BY_NAME
    ]

    purified_text = _purified_context(
        purified_json
    )

    if knowledge_context:
        purified_text = (
            f"{purified_text}\n\n"
            "TRUSTED ORGANIZATION KNOWLEDGE CONTEXT:\n"
            f"{knowledge_context}\n\n"
            "KNOWLEDGE-BASE RULES:\n"
            "Use the trusted context above as the source of truth "
            "for organization-specific claims. Do not invent missing "
            "policy, procedure, or company facts. Cite source/page "
            "information from the context when relevant."
        )

    messages = [
        {
            "role": "system",
            "content": AGENT_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": purified_text,
        },
    ]

    # =====================================================
    # 8. NO TOOL WORKER
    # =====================================================

    if not selected_tool_objects:

        emit_event(
            GENERATING_RESPONSE,
            "Generating response from model",
        )

        # Keep normal answers compact and fast.
        # This is the main latency improvement for
        # informational questions.
        response = await _ollama_chat(
            selected_model,
            base_url,
            messages,
            temperature=0.0,
            keep_alive="15m",
            num_predict=384,
            num_ctx=4096,
        )

        emit_event(
            COMPLETED,
            "Processing completed",
        )

        final_answer = (
            response["message"].get("content")
            or ""
        ).strip()

        # A local worker can occasionally return an empty content field even
        # though the request itself succeeded. Never send an empty answer to
        # the frontend for a knowledge request.
        if not final_answer and internal_knowledge_requested:
            final_answer = knowledge_context.strip()

        return {
            "category":
                route["category"],

            "model_used":
                selected_model,

            "selected_tools":
                [],

            "fallback_used":
                route.get(
                    "fallback_used",
                    False,
                ),

            "final_answer":
                final_answer,

            "tool_calls_made":
                0,

            "message_count":
                3,

            "generated_files":
                [],

            "execution_results":
                [],
            "knowledge_results": knowledge_results,
        }

    # =====================================================
    # 9. TOOL WORKER
    # =====================================================

    emit_event(
        GENERATING_RESPONSE,
        "Generating response with tools",
    )

    (
        final_answer,
        history,
        tool_calls_made,
        executed_calls,
    ) = await _native_tool_agent(
        selected_model,
        base_url,
        messages,
        selected_tool_objects,
    )

    # =====================================================
    # 10. ENGINEERING GUARANTEE
    # =====================================================

    if engineering_requested:

        already_calculated = any(
            call.get("tool")
            == "engineering_calculator"
            for call in executed_calls
        )

        if not already_calculated:

            fallback_call = (
                await _engineering_fallback(
                    purified_text
                )
            )

            executed_calls.append(
                fallback_call
            )

            final_answer = (
                fallback_call["result"]
            )

    # =====================================================
    # 11. DOCUMENT GUARANTEE
    # =====================================================

    if required_document_tool:

        already_created = any(
            call.get("tool")
            == required_document_tool
            for call in executed_calls
        )

        if not already_created:

            emit_event(
                GENERATING_RESPONSE,
                "Generating document content",
            )

            (
                fallback_content,
                fallback_call,
            ) = await _document_fallback(
                selected_model,
                base_url,
                purified_text,
                required_document_tool,
            )

            executed_calls.append(
                fallback_call
            )

            final_answer = (
                "The requested document "
                "has been created successfully."
            )

    # =====================================================
    # 12. RESULTS
    # =====================================================

    generated_files = (
        _collect_generated_files(
            executed_calls
        )
    )

    execution_results = (
        _extract_execution_results(
            executed_calls
        )
    )

    # =====================================================
    # 13. PYTHON FALLBACK RESPONSE
    # =====================================================

    if (
        code_execution_requested
        and execution_results
        and not (
            final_answer
            or ""
        ).strip()
    ):

        final_answer = (
            "The code was executed "
            "successfully. "
            "The output is shown below."
        )

    # =====================================================
    # 14. FINAL PROGRESS
    # =====================================================

    emit_event(
        GENERATING_RESPONSE,
        "Generating final response",
    )

    emit_event(
        COMPLETED,
        "Processing completed",
    )

    return {
        "category":
            route["category"],

        "model_used":
            selected_model,

        "selected_tools":
            selected_tool_names,

        "fallback_used":
            route.get(
                "fallback_used",
                False,
            ),

        "final_answer":
            final_answer,

        "tool_calls_made":
            tool_calls_made,

        "message_count":
            len(history),

        "generated_files":
            generated_files,

        "execution_results":
            execution_results,
        "knowledge_results": knowledge_results,
    }


# =========================================================
# TEST LLM
# =========================================================

async def test_llm(
    prompt: str,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    use_tools: bool = False,
) -> Dict[str, Any]:

    registry = settings.load_model_registry()

    model_name = (
        model
        or registry[
            "default_router"
        ][
            "model_name"
        ]
    )

    base_url = _normalise_base_url(
        registry[
            "default_router"
        ][
            "api_base"
        ]
    )

    for cfg in (
        registry
        .get(
            "task_routes",
            {},
        )
        .values()
    ):

        if (
            cfg.get(
                "primary_model"
            )
            == model_name
        ):

            base_url = _normalise_base_url(
                cfg["api_base"]
            )

            break

    messages = []

    if system_prompt:

        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    if not use_tools:

        response = await _ollama_chat(
            model_name,
            base_url,
            messages,
            temperature=0.1,
            num_predict=512,
            num_ctx=4096,
        )

        return {
            "model_used":
                model_name,

            "response_content":
                (
                    response["message"]
                    .get("content")
                    or ""
                ),

            "tool_calls":
                (
                    response["message"]
                    .get("tool_calls")
                    or []
                ),
        }

    (
        final_answer,
        _,
        tool_calls_made,
        executed_calls,
    ) = await _native_tool_agent(
        model_name,
        base_url,
        messages,
        AGENT_TOOLS,
    )

    return {
        "model_used":
            model_name,

        "response_content":
            final_answer,

        "tool_calls":
            [
                {
                    "tool":
                        call.get("tool"),

                    "arguments":
                        call.get(
                            "arguments"
                        ),
                }
                for call
                in executed_calls
            ],

        "tool_calls_made":
            tool_calls_made,
    }