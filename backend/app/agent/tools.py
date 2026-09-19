from pathlib import Path

from app.agent.filesystem import (
    list_directory,
    search_files,
    get_file_info,
    read_file,
)

from app.agent.engineering_calculator import (
    engineering_calculation,
)

from langchain_core.tools import tool

from app.agent.document_generator import (
    generate_excel_document,
    generate_pptx_document,
    generate_word_document,
)

from app.agent.sandbox import execute_python_code
from app.core.config import settings
from app.services.rag_engine import rag_engine


def _resolve_safe_path(
    relative_path: str,
) -> Path:
    """
    Resolves a path relative to the workspace,
    rejecting attempts to escape it.
    """

    target = (
        settings.WORKSPACE_DIR
        / relative_path
    ).resolve()

    workspace_root = (
        settings.WORKSPACE_DIR.resolve()
    )

    if (
        workspace_root not in target.parents
        and target != workspace_root
    ):
        raise ValueError(
            f"Path '{relative_path}' "
            "escapes the sandboxed workspace"
        )

    return target


# =========================================================
# WORKSPACE FILE TOOLS
# =========================================================

@tool
def read_workspace_file(
    relative_path: str,
) -> str:
    """
    Reads and returns the text content of
    a file inside the sandboxed workspace.
    """

    path = _resolve_safe_path(
        relative_path
    )

    if not path.exists():

        return (
            f"Error: file "
            f"'{relative_path}' "
            "does not exist in the workspace."
        )

    return path.read_text(
        encoding="utf-8",
        errors="ignore",
    )


@tool
def write_workspace_file(
    relative_path: str,
    content: str,
) -> str:
    """
    Writes text content to a file in the
    sandboxed workspace.
    """

    path = _resolve_safe_path(
        relative_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        content,
        encoding="utf-8",
    )

    return (
        f"Wrote {len(content)} characters "
        f"to '{relative_path}'."
    )


# =========================================================
# PYTHON EXECUTION
# =========================================================

@tool
def run_python_code(
    code: str,
) -> str:
    """
    Executes Python code in a sandboxed
    subprocess with no outbound network access.
    """

    return execute_python_code(code)


# =========================================================
# DOCUMENT GENERATION
# =========================================================

@tool
def create_word_document(
    filename: str,
    content: str,
) -> str:
    """
    Creates a Word (.docx) deliverable.
    content uses '# ' for the title,
    '## ' for headings, and normal lines
    for paragraphs.
    """

    return generate_word_document(
        filename,
        content,
    )


@tool
def create_excel_document(
    filename: str,
    csv_data: str,
) -> str:
    """
    Creates an Excel (.xlsx) deliverable
    from CSV-formatted text.
    """

    return generate_excel_document(
        filename,
        csv_data,
    )


@tool
def create_pptx_document(
    filename: str,
    content: str,
) -> str:
    """
    Creates a PowerPoint (.pptx) deliverable.
    content uses '# ' for slide titles
    and '- ' lines for bullets.
    """

    return generate_pptx_document(
        filename,
        content,
    )


# =========================================================
# KNOWLEDGE BASE
# =========================================================

@tool
async def search_knowledge_base(
    query: str,
) -> str:
    """
    Searches the organization's local
    knowledge base for relevant context.
    """

    return await rag_engine.answer_context(
        query,
        top_k=4,
    )


# =========================================================
# DIRECT LOCAL FILESYSTEM TOOLS
# =========================================================

@tool
def filesystem_list_directory(
    path: str,
) -> str:
    """
    Lists files and folders in any directory
    on the local computer.
    """

    return list_directory(path)


@tool
def filesystem_search_files(
    path: str,
    filename: str,
) -> str:
    """
    Searches recursively for files by name
    under a local computer directory.
    """

    return search_files(
        path,
        filename,
    )


@tool
def filesystem_get_file_info(
    path: str,
) -> str:
    """
    Gets metadata such as size, extension,
    and MIME type for a local file.
    """

    return get_file_info(path)


@tool
def filesystem_read_file(
    path: str,
) -> str:
    """
    Reads the text content of a local file
    using its absolute filesystem path.
    """

    return read_file(path)


# =========================================================
# ENGINEERING CALCULATOR
# =========================================================

@tool
def engineering_calculator(
    calculation: str,
    parameters: str,
) -> str:
    """
    Performs deterministic engineering calculations.

    Supported calculations:
    - pump_power
    - pressure_drop
    - pipe_velocity
    - reynolds_number
    - heat_duty
    - auto

    parameters may be valid JSON or natural-language
    engineering parameters.
    """

    return engineering_calculation(
        calculation,
        parameters,
    )


# =========================================================
# MASTER TOOL REGISTRY
# =========================================================

AGENT_TOOLS = [

    # Workspace
    read_workspace_file,
    write_workspace_file,

    # Python
    run_python_code,

    # Documents
    create_word_document,
    create_excel_document,
    create_pptx_document,

    # Knowledge base
    search_knowledge_base,

    # Direct filesystem
    filesystem_list_directory,
    filesystem_search_files,
    filesystem_get_file_info,
    filesystem_read_file,

    # Engineering
    engineering_calculator,
]