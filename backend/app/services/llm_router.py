import httpx
import json
import logging
import re
from typing import List, Optional, Tuple

from app.core.config import settings


logger = logging.getLogger(__name__)

_router_http_client: httpx.AsyncClient | None = None


async def _get_router_http_client() -> httpx.AsyncClient:
    global _router_http_client
    if _router_http_client is None or _router_http_client.is_closed:
        _router_http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(8.0, connect=2.0),
            limits=httpx.Limits(
                max_connections=2,
                max_keepalive_connections=1,
            ),
        )
    return _router_http_client


# =========================================================
# ROUTER SYSTEM PROMPT
# =========================================================

ROUTER_SYSTEM_PROMPT = """
You are the FAST router for a local air-gapped AI assistant.

Return ONLY one JSON object:
{"category":"TASK_DOCUMENT_SUMMARY","tools":[],"fast_path":true,"answer":"..."}

Categories:
- TASK_CODING: generate, debug, review, or implement code.
- TASK_DOCUMENT_SUMMARY: normal/general questions, explanations, summaries, or documents.
- TASK_MULTIMODAL_VISION: only when an actual image attachment exists.
- TASK_ENGINEERING_CALCULATION: engineering/math tasks requiring engineering_calculator.

Tools:
read_workspace_file, write_workspace_file, run_python_code,
create_word_document, create_excel_document, create_pptx_document,
search_knowledge_base, filesystem_list_directory,
filesystem_search_files, filesystem_get_file_info,
filesystem_read_file, engineering_calculator.

FAST PATH:
Set fast_path=true and put the final answer in "answer" ONLY for a short,
simple, general question or casual request that needs no tool, attachment,
company knowledge, calculation, coding, or document generation.
For fast_path=true, tools MUST be [].
For all other requests, fast_path=false and answer="".

Rules:
1. Actual code generation/debugging/review -> TASK_CODING.
2. Normal conceptual/informational questions -> TASK_DOCUMENT_SUMMARY.
3. Engineering calculations -> TASK_ENGINEERING_CALCULATION with engineering_calculator.
4. Vision -> TASK_MULTIMODAL_VISION only with an image attachment.
5. Filesystem requests -> the minimum required filesystem tool.
6. Document generation -> the requested document tool.
7. Internal/company SOPs, manuals, policies, procedures, standards,
   internal correspondence, or organizational knowledge -> search_knowledge_base.
8. "What is Python?" is informational, not coding.
9. Never use a tool unless required.
10. If unsure whether a request is safe for the fast path, set fast_path=false.
"""



# =========================================================
# ALLOWED TOOLS
# =========================================================

ALLOWED_TOOLS = {
    "read_workspace_file",
    "write_workspace_file",
    "run_python_code",
    "create_word_document",
    "create_excel_document",
    "create_pptx_document",
    "search_knowledge_base",
    "filesystem_list_directory",
    "filesystem_search_files",
    "filesystem_get_file_info",
    "filesystem_read_file",
    "engineering_calculator",
}


VALID_CATEGORIES = {
    "TASK_CODING",
    "TASK_DOCUMENT_SUMMARY",
    "TASK_MULTIMODAL_VISION",
    "TASK_ENGINEERING_CALCULATION",
}


# =========================================================
# ROUTER SERVICE
# =========================================================

class LLMRouterService:

    async def classify_and_route(
        self,
        user_prompt: str,
        attachment_types: Optional[List[str]] = None,
    ) -> dict:

        attachment_types = attachment_types or []
        prompt_lower = (user_prompt or "").lower().strip()

        # Deterministic high-priority routes bypass the small router LLM.
        # Engineering calculations are handled locally by a deterministic
        # calculator, so spending an additional 0.5B inference here only adds
        # latency and can introduce tool-call formatting failures.
        internal_knowledge_request = self._is_internal_knowledge_request(
            prompt_lower
        )
        engineering_request = self._is_engineering_request(prompt_lower)

        direct_answer = ""
        fast_path = False
        router_model = ""

        if engineering_request:
            category = "TASK_ENGINEERING_CALCULATION"
            selected_tools = ["engineering_calculator"]
            fallback_used = False
        elif internal_knowledge_request:
            category = "TASK_DOCUMENT_SUMMARY"
            selected_tools = ["search_knowledge_base"]
            fallback_used = False
        else:
            (
                category,
                selected_tools,
                fallback_used,
                fast_path,
                direct_answer,
                router_model,
            ) = await self._call_router_llm(
                user_prompt,
                attachment_types,
            )

        # -----------------------------------------------------
        # Deterministic safety/priority signals
        # -----------------------------------------------------

        engineering_request = self._is_engineering_request(
            prompt_lower
        )

        filesystem_request = self._is_filesystem_request(
            user_prompt,
            prompt_lower,
        )

        document_tool = self._detect_document_tool(
            prompt_lower
        )

        coding_request = self._is_actual_coding_request(
            prompt_lower
        )

        image_request = self._has_image_attachment(
            attachment_types
        )

        internal_knowledge_request = self._is_internal_knowledge_request(
            prompt_lower
        )

        # =====================================================
        # FAST-PATH SAFETY
        # =====================================================
        # The small router may answer only a genuinely simple, tool-free,
        # non-attachment request. Deterministic signals always win.
        if (
            fast_path
            and (
                attachment_types
                or internal_knowledge_request
                or engineering_request
                or filesystem_request
                or document_tool
                or coding_request
                or image_request
                or selected_tools
                or not direct_answer.strip()
            )
        ):
            fast_path = False
            direct_answer = ""

        # =====================================================
        # PRIORITY 1 - DOCUMENT GENERATION
        # =====================================================

        if document_tool:
            category = "TASK_DOCUMENT_SUMMARY"
            selected_tools = [document_tool]

        # =====================================================
        # PRIORITY 2 - ENGINEERING
        # =====================================================

        elif engineering_request:
            category = "TASK_ENGINEERING_CALCULATION"

            selected_tools = [
                "engineering_calculator"
            ]

            if filesystem_request:
                selected_tools = [
                    "filesystem_read_file",
                    "engineering_calculator",
                ]

        # =====================================================
        # PRIORITY 3 - FILESYSTEM
        # =====================================================

        elif filesystem_request:
            category = "TASK_DOCUMENT_SUMMARY"

            if self._contains_any(
                prompt_lower,
                [
                    "list files",
                    "list the files",
                    "show files",
                    "show the files",
                    "list folder",
                    "list directory",
                    "show folder",
                    "show directory",
                    "what files are in",
                    "what is in this folder",
                    "what's in this folder",
                ],
            ):
                selected_tools = [
                    "filesystem_list_directory"
                ]

            elif self._contains_any(
                prompt_lower,
                [
                    "find file",
                    "find the file",
                    "find files",
                    "find the files",
                    "search for file",
                    "search for the file",
                    "search for files",
                    "search for the files",
                    "search file",
                    "search files",
                    "look for file",
                    "look for the file",
                    "locate file",
                    "locate the file",
                ],
            ):
                selected_tools = [
                    "filesystem_search_files"
                ]

            elif self._contains_any(
                prompt_lower,
                [
                    "file info",
                    "file information",
                    "file metadata",
                    "metadata",
                    "information about the file",
                    "information on the file",
                    "details about the file",
                    "details of the file",
                    "properties of the file",
                ],
            ):
                selected_tools = [
                    "filesystem_get_file_info"
                ]

            else:
                selected_tools = [
                    "filesystem_read_file"
                ]

        # =====================================================
        # PRIORITY 4 - ACTUAL CODING
        # =====================================================

        elif (
            coding_request
            and not image_request
        ):
            category = "TASK_CODING"

            if self._contains_any(
                prompt_lower,
                [
                    "run this",
                    "run the code",
                    "run code",
                    "execute this",
                    "execute the code",
                    "execute code",
                    "test this code",
                    "test the code",
                    "test code",
                    "run this program",
                    "execute this program",
                ],
            ):
                selected_tools = [
                    "run_python_code"
                ]
            else:
                selected_tools = []

        # =====================================================
        # PRIORITY 5 - IMAGE
        # =====================================================

        elif image_request:
            category = "TASK_MULTIMODAL_VISION"
            selected_tools = []

        # =====================================================
        # INFORMATIONAL QUESTION
        #
        # If the router LLM selected coding for something that
        # is clearly educational/informational, correct it.
        # =====================================================

        elif self._is_informational_question(
            prompt_lower
        ):
            category = "TASK_DOCUMENT_SUMMARY"
            selected_tools = []

        # =====================================================
        # IMAGE SAFETY
        # =====================================================

        if image_request:
            category = "TASK_MULTIMODAL_VISION"
            selected_tools = []

        # Any post-router deterministic override disables the fast path.
        if (
            fast_path
            and (
                category != "TASK_DOCUMENT_SUMMARY"
                or selected_tools
                or internal_knowledge_request
                or image_request
                or attachment_types
            )
        ):
            fast_path = False
            direct_answer = ""

        # RAG is opt-in. This avoids adding retrieval work/tooling to ordinary
        # questions and keeps latency low. It can coexist with document,
        # coding, filesystem and engineering tools.
        if (
            internal_knowledge_request
            and "search_knowledge_base" not in selected_tools
        ):
            selected_tools.append("search_knowledge_base")

        # An explicit knowledge-base request is text/document knowledge, not
        # multimodal vision. A real image attachment remains the only trigger
        # for TASK_MULTIMODAL_VISION.
        if internal_knowledge_request and not image_request:
            category = "TASK_DOCUMENT_SUMMARY"

        # =====================================================
        # MODEL REGISTRY
        # =====================================================

        registry = settings.load_model_registry()

        task_routes = registry.get(
            "task_routes",
            {},
        )

        route_info = task_routes.get(
            category
        )

        # Engineering is intentionally allowed to fall back
        # to the existing document/text worker because the
        # current model.yaml has no separate engineering model.
        if route_info is None:
            route_info = task_routes.get(
                "TASK_DOCUMENT_SUMMARY"
            )

        if route_info is None:
            raise RuntimeError(
                "TASK_DOCUMENT_SUMMARY route is missing "
                "from model registry."
            )

        return {
            "category": category,
            "selected_model": route_info["primary_model"],
            "api_base": route_info["api_base"],
            "vram_required_gb": route_info.get(
                "vram_required_gb",
                0,
            ),
            "context_length": route_info.get(
                "context_length"
            ),
            "selected_tools": selected_tools,
            "fallback_used": fallback_used,
            "fast_path": bool(fast_path and direct_answer.strip()),
            "direct_answer": direct_answer.strip() if fast_path else "",
            "router_model": router_model,
        }

    # =========================================================
    # SIGNAL HELPERS
    # =========================================================

    @staticmethod
    def _contains_any(
        text: str,
        values: list[str],
    ) -> bool:
        return any(
            value in text
            for value in values
        )

    # =========================================================
    # ENGINEERING
    # =========================================================

    def _is_engineering_request(
        self,
        text: str,
    ) -> bool:

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

        domain_signal = self._contains_any(
            text,
            signals,
        )

        has_numeric_input = bool(
            re.search(
                r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:\s*[A-Za-z%°²³^/*.-]+)?",
                text,
            )
        )

        calculation_intent = self._contains_any(
            text,
            [
                "calculate",
                "calculation",
                "find the",
                "determine",
                "compute",
                "work out",
                "solve",
                "maximum bending",
            ],
        )

        return bool(
            domain_signal
            and (calculation_intent or has_numeric_input)
        )

    # =========================================================
    # FILESYSTEM
    # =========================================================

    def _is_filesystem_request(
        self,
        original_prompt: str,
        text: str,
    ) -> bool:

        signals = [
            "read file",
            "read the file",
            "open file",
            "open the file",
            "access file",
            "access the file",
            "find file",
            "find the file",
            "find files",
            "find the files",
            "search for file",
            "search for the file",
            "search for files",
            "search for the files",
            "search file",
            "search files",
            "list files",
            "list the files",
            "show files",
            "show the files",
            "get file",
            "get the file",
            "file info",
            "file information",
            "file metadata",
            "metadata of the file",
            "take the file",
            "load file",
            "load the file",
            "from my computer",
            "from my pc",
            "from my laptop",
            "from my desktop",
            "local file",
            "local files",
            "local folder",
            "local directory",
            "filesystem",
            "file system",
            "folder contents",
            "directory contents",
            "browse folder",
            "browse directory",
            "go into",
            "go to folder",
            "open folder",
            "enter folder",
            "navigate to folder",
            "go back",
            "parent folder",
            "parent directory",
            "current folder",
            "current directory",
            "list here",
            "show here",
            "what is in",
            "what's in",
            "show folders",
            "list folders",
        ]

        if self._contains_any(text, signals):
            return True

        has_windows_path = bool(
            re.search(
                r"[A-Za-z]:\\",
                original_prompt or "",
            )
        )

        has_unix_path = bool(
            re.search(
                r"(?<!\w)/(?:[^/\s]+/)+[^/\s]*",
                original_prompt or "",
            )
        )

        return (
            has_windows_path
            or has_unix_path
        )

    # =========================================================
    # ACTUAL CODING
    # =========================================================

    def _is_actual_coding_request(
        self,
        text: str,
    ) -> bool:

        # Strong coding signals.
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

        if self._contains_any(
            text,
            strong_signals,
        ):
            return True

        # Code-like requests that explicitly ask for coding.
        coding_verbs = [
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

        has_coding_word = self._contains_any(
            text,
            coding_verbs,
        )

        has_action_word = self._contains_any(
            text,
            action_words,
        )

        if (
            has_coding_word
            and has_action_word
        ):
            return True

        return False

    # =========================================================
    # INTERNAL KNOWLEDGE / RAG
    # =========================================================

    def _is_internal_knowledge_request(
        self,
        text: str,
    ) -> bool:
        """Detect explicit requests for organization-owned knowledge.

        RAG is deliberately opt-in. General questions should not pay the
        retrieval/tool-selection cost unless the user asks for internal
        organizational information.
        """
        signals = [
            "according to the knowledge base",
            "according to knowledge base",
            "from the knowledge base",
            "in the knowledge base",
            "according to the document",
            "according to the documents",
            "according to the pdf",
            "according to the pdfs",
            "from the document",
            "from the documents",
            "according to our sop",
            "according to the sop",
            "according to our manual",
            "according to the manual",
            "according to our policy",
            "according to company policy",
            "according to our procedure",
            "according to our procedures",
            "according to our standards",
            "according to company standards",
            "our internal",
            "internal company",
            "internal document",
            "internal documents",
            "company sop",
            "company manual",
            "company procedure",
            "company policy",
            "company guidelines",
            "organization's sop",
            "organization's policy",
            "organizational knowledge",
            "our guidelines",
            "our process",
            "our maintenance procedure",
            "our safety procedure",
            "past correspondence",
            "internal correspondence",
            "confidential procedure",
            "confidential policy",
        ]
        return any(signal in text for signal in signals)

    # =========================================================
    # INFORMATIONAL QUESTIONS
    # =========================================================

    def _is_informational_question(
        self,
        text: str,
    ) -> bool:

        informational_starts = [
            "tell me about",
            "explain",
            "what is",
            "what are",
            "what does",
            "how does",
            "why is",
            "why are",
            "why does",
            "define",
            "describe",
            "difference between",
            "compare",
            "give me an overview",
            "teach me",
        ]

        return self._contains_any(
            text,
            informational_starts,
        )

    # =========================================================
    # DOCUMENT GENERATION
    # =========================================================

    def _detect_document_tool(
        self,
        text: str,
    ) -> Optional[str]:

        if self._contains_any(
            text,
            [
                "create excel",
                "create an excel",
                "generate excel",
                "generate an excel",
                "download excel",
                "create spreadsheet",
                "generate spreadsheet",
                "xlsx",
                "spreadsheet",
            ],
        ):
            return "create_excel_document"

        if self._contains_any(
            text,
            [
                "create powerpoint",
                "create a powerpoint",
                "generate powerpoint",
                "generate a powerpoint",
                "download powerpoint",
                "create ppt",
                "generate ppt",
                "pptx",
            ],
        ):
            return "create_pptx_document"

        if self._contains_any(
            text,
            [
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
                "docx",
            ],
        ):
            return "create_word_document"

        return None

    # =========================================================
    # IMAGE
    # =========================================================

    @staticmethod
    def _has_image_attachment(
        attachment_types: List[str],
    ) -> bool:

        normalized = {
            str(item).lower().strip()
            for item in attachment_types
        }

        return bool(
            normalized
            & {
                "image",
                "jpg",
                "jpeg",
                "png",
                "webp",
                "bmp",
                "gif",
                "image/jpeg",
                "image/png",
                "image/webp",
                "image/bmp",
                "image/gif",
            }
        )

    # =========================================================
    # CALL 0.5B ROUTER
    # =========================================================

    async def _call_router_llm(
        self,
        prompt: str,
        attachment_types: List[str],
    ) -> Tuple[
        str,
        List[str],
        bool,
        bool,
        str,
        str,
    ]:

        registry = settings.load_model_registry()

        router_cfg = registry[
            "default_router"
        ]

        routing_input = (
            f"User request:\n{prompt}\n\n"
            f"Attachment types: "
            f"{attachment_types or ['none']}"
        )

        native_root = (
            router_cfg["api_base"]
            .rstrip("/")
            .removesuffix("/v1")
            .rstrip("/")
        )

        payload = {
            "model": router_cfg["model_name"],
            "messages": [
                {
                    "role": "system",
                    "content": ROUTER_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": routing_input,
                },
            ],
            "temperature": 0.0,
            "format": "json",
            "stream": False,

            # Keep the router response extremely small.
            "options": {
                "num_predict": 192,
                "num_ctx": 1536,
            },

            "keep_alive": -1,
        }

        try:
            client = await _get_router_http_client()

            response = await client.post(
                f"{native_root}/api/chat",
                json=payload,
            )

            response.raise_for_status()

            data = response.json()

            raw_content = (
                data
                .get("message", {})
                .get("content", "")
            )

            if not raw_content:
                raise ValueError(
                    "Router returned empty content"
                )

            try:
                parsed = json.loads(
                    raw_content
                )
            except json.JSONDecodeError:

                match = re.search(
                    r"\{.*\}",
                    raw_content,
                    re.DOTALL,
                )

                if not match:
                    raise ValueError(
                        "Invalid router response"
                    )

                parsed = json.loads(
                    match.group(0)
                )

            category = parsed.get(
                "category",
                "TASK_DOCUMENT_SUMMARY",
            )

            if category not in VALID_CATEGORIES:
                raise ValueError(
                    f"Invalid router category: {category}"
                )

            raw_tools = parsed.get(
                "tools",
                [],
            )

            if not isinstance(
                raw_tools,
                list,
            ):
                raw_tools = []

            selected_tools = [
                tool
                for tool in raw_tools
                if tool in ALLOWED_TOOLS
            ]

            if category == "TASK_MULTIMODAL_VISION":
                selected_tools = []

            fast_path = bool(parsed.get("fast_path", False))
            direct_answer = str(parsed.get("answer") or "").strip()

            # Never accept a router fast-path answer when it also requested
            # tools. This is a hard correctness guard.
            if selected_tools or category != "TASK_DOCUMENT_SUMMARY":
                fast_path = False
                direct_answer = ""

            logger.info(
                "0.5B router decision: category=%s tools=%s fast_path=%s",
                category,
                selected_tools,
                fast_path,
            )

            return (
                category,
                selected_tools,
                False,
                fast_path,
                direct_answer,
                router_cfg["model_name"],
            )

        except Exception:

            logger.exception(
                "Router LLM failed; "
                "using deterministic fallback"
            )

            return self._deterministic_fallback(
                prompt,
                attachment_types,
            )

    # =========================================================
    # DETERMINISTIC FALLBACK
    # =========================================================

    def _deterministic_fallback(
        self,
        prompt: str,
        attachment_types: List[str],
    ) -> Tuple[
        str,
        List[str],
        bool,
        bool,
        str,
        str,
    ]:

        text = (prompt or "").lower().strip()

        rag_tool = (
            ["search_knowledge_base"]
            if self._is_internal_knowledge_request(text)
            else []
        )

        if self._has_image_attachment(attachment_types):
            return (
                "TASK_MULTIMODAL_VISION",
                [],
                True,
                False,
                "",
                "",
            )

        if self._is_engineering_request(text):
            return (
                "TASK_ENGINEERING_CALCULATION",
                ["engineering_calculator"] + rag_tool,
                True,
                False,
                "",
                "",
            )

        if self._is_filesystem_request(prompt, text):
            if self._contains_any(
                text,
                [
                    "list files",
                    "list the files",
                    "show files",
                    "show the files",
                    "list folder",
                    "list directory",
                ],
            ):
                return (
                    "TASK_DOCUMENT_SUMMARY",
                    ["filesystem_list_directory"] + rag_tool,
                    True,
                    False,
                    "",
                    "",
                )

            if self._contains_any(
                text,
                [
                    "find file",
                    "find the file",
                    "find files",
                    "find the files",
                    "search for file",
                    "search for files",
                    "search file",
                    "search files",
                ],
            ):
                return (
                    "TASK_DOCUMENT_SUMMARY",
                    ["filesystem_search_files"] + rag_tool,
                    True,
                    False,
                    "",
                    "",
                )

            if self._contains_any(
                text,
                [
                    "file info",
                    "file information",
                    "metadata",
                    "information about the file",
                ],
            ):
                return (
                    "TASK_DOCUMENT_SUMMARY",
                    ["filesystem_get_file_info"] + rag_tool,
                    True,
                    False,
                    "",
                    "",
                )

            return (
                "TASK_DOCUMENT_SUMMARY",
                ["filesystem_read_file"] + rag_tool,
                True,
                False,
                "",
                "",
            )

        document_tool = self._detect_document_tool(text)

        if document_tool:
            return (
                "TASK_DOCUMENT_SUMMARY",
                [document_tool] + rag_tool,
                True,
                False,
                "",
                "",
            )

        if self._is_actual_coding_request(text):
            return (
                "TASK_CODING",
                [],
                True,
                False,
                "",
                "",
            )

        return (
            "TASK_DOCUMENT_SUMMARY",
            rag_tool,
            True,
            False,
            "",
            "",
        )


# =========================================================
# GLOBAL ROUTER
# =========================================================

router_service = LLMRouterService()