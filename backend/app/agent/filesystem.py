from pathlib import Path
import mimetypes
import os
import re

import fitz


# ============================================================
# USER / WINDOWS DIRECTORY HELPERS
# ============================================================

def _home_directory() -> Path:
    """Return the current user's home directory."""
    return Path.home().resolve()


def _common_directories() -> list[Path]:
    """
    Return common user directories.

    Supports normal Windows folders as well as OneDrive-
    redirected Desktop/Documents folders.
    """

    home = _home_directory()

    candidates = [
        # Standard locations
        home / "Desktop",
        home / "Downloads",
        home / "Documents",
        home / "Pictures",
        home / "Videos",
        home / "Music",

        # OneDrive locations
        home / "OneDrive" / "Desktop",
        home / "OneDrive" / "Downloads",
        home / "OneDrive" / "Documents",
        home / "OneDrive" / "Pictures",
        home / "OneDrive" / "Videos",
        home / "OneDrive" / "Music",

        # OneDrive variants sometimes used by Windows
        home / "OneDrive - Personal" / "Desktop",
        home / "OneDrive - Personal" / "Documents",
    ]

    result = []

    seen = set()

    for path in candidates:
        try:
            resolved = path.resolve()

            if (
                resolved.exists()
                and resolved.is_dir()
                and str(resolved).lower() not in seen
            ):
                result.append(resolved)
                seen.add(str(resolved).lower())

        except (OSError, RuntimeError):
            continue

    return result


# ============================================================
# TEXT CLEANING
# ============================================================

def _clean_path_text(path: str) -> str:
    """
    Clean a path supplied by the LLM.

    Handles values such as:

        text.txt
        "text.txt"
        the file text.txt
        file text.txt
        my desktop
        the desktop
    """

    if not path:
        return ""

    value = str(path).strip()

    # Remove surrounding quotes.
    value = value.strip("\"'")

    prefixes = [
        "the file ",
        "file ",
        "the folder ",
        "folder ",
        "the directory ",
        "directory ",
    ]

    lowered = value.lower()

    for prefix in prefixes:
        if lowered.startswith(prefix):
            value = value[len(prefix):].strip()
            lowered = value.lower()

    return value.strip("\"'")


# ============================================================
# NATURAL LOCATION RESOLUTION
# ============================================================

def _location_directory(location: str) -> Path | None:
    """
    Resolve natural locations such as:

        desktop
        my desktop
        downloads
        my downloads
        documents
        my documents
    """

    if not location:
        return None

    home = _home_directory()

    value = location.lower().strip()

    value = re.sub(
        r"^(my|the)\s+",
        "",
        value,
    ).strip()

    aliases = {
        "desktop": [
            home / "Desktop",
            home / "OneDrive" / "Desktop",
            home / "OneDrive - Personal" / "Desktop",
        ],

        "downloads": [
            home / "Downloads",
            home / "OneDrive" / "Downloads",
        ],

        "download": [
            home / "Downloads",
            home / "OneDrive" / "Downloads",
        ],

        "documents": [
            home / "Documents",
            home / "OneDrive" / "Documents",
            home / "OneDrive - Personal" / "Documents",
        ],

        "document": [
            home / "Documents",
            home / "OneDrive" / "Documents",
            home / "OneDrive - Personal" / "Documents",
        ],

        "docs": [
            home / "Documents",
            home / "OneDrive" / "Documents",
        ],

        "pictures": [
            home / "Pictures",
            home / "OneDrive" / "Pictures",
        ],

        "photos": [
            home / "Pictures",
            home / "OneDrive" / "Pictures",
        ],

        "images": [
            home / "Pictures",
            home / "OneDrive" / "Pictures",
        ],

        "videos": [
            home / "Videos",
            home / "OneDrive" / "Videos",
        ],

        "music": [
            home / "Music",
            home / "OneDrive" / "Music",
        ],

        "home": [
            home,
        ],
        "this pc": [
            _filesystem_root(),
        ],
        "my computer": [
            _filesystem_root(),
        ],
        "computer": [
            _filesystem_root(),
        ],
        "root": [
            _filesystem_root(),
        ],
    }

    possible_paths = aliases.get(value, [])

    for path in possible_paths:
        try:
            resolved = path.resolve()

            if resolved.exists() and resolved.is_dir():
                return resolved

        except (OSError, RuntimeError):
            continue

    return None


# ============================================================
# FILE SEARCH INSIDE DIRECTORY
# ============================================================

def _find_file_in_directory(
    directory: Path,
    filename: str,
) -> Path | None:
    """
    Find a file inside a directory.

    First checks directly inside the directory,
    then recursively searches using case-insensitive
    filename matching.
    """

    if not directory.exists() or not directory.is_dir():
        return None

    filename = filename.strip().strip("\"'")

    if not filename:
        return None

    # --------------------------------------------------------
    # Direct path
    # --------------------------------------------------------

    direct = directory / filename

    try:
        if direct.exists() and direct.is_file():
            return direct.resolve()
    except (OSError, RuntimeError):
        pass

    # --------------------------------------------------------
    # Case-insensitive recursive search
    # --------------------------------------------------------

    target_name = Path(filename).name.lower()

    try:
        for item in directory.rglob("*"):

            try:
                if (
                    item.is_file()
                    and item.name.lower() == target_name
                ):
                    return item.resolve()

            except (OSError, RuntimeError):
                continue

    except (PermissionError, OSError, RuntimeError):
        pass

    return None


# ============================================================
# FULL PATH RESOLUTION
# ============================================================

def _resolve_path(path: str) -> Path:
    """
    Resolve a user-provided filesystem path.

    Supported:

        C:\\Users\\User\\Desktop\\text.txt

        ~/Desktop/text.txt

        text.txt

        text.txt in my desktop

        report.pdf from downloads

        notes.txt on my desktop

        desktop
    """

    if not path or not str(path).strip():
        raise ValueError("Path cannot be empty.")

    original = str(path).strip()
    cleaned = _clean_path_text(original)

    # ========================================================
    # 1. DIRECT PATH
    # ========================================================

    # Normal Windows / Unix / home-relative path.
    try:
        direct = Path(cleaned).expanduser()

        resolved_direct = direct.resolve()

        if resolved_direct.exists():
            return resolved_direct

    except (OSError, RuntimeError):
        pass

    # ========================================================
    # 2. WINDOWS PATH HANDLING
    # ========================================================

    # This is important if the backend receives a Windows path
    # containing escaped backslashes.
    #
    # Example:
    #
    # C:\Users\nvvar\Desktop\text.txt
    #

    windows_path_match = re.match(
        r"^[A-Za-z]:[\\/].+",
        cleaned,
    )

    if windows_path_match:

        windows_path = cleaned.replace("/", "\\")

        try:
            windows_target = Path(windows_path)

            if windows_target.exists():
                return windows_target.resolve()

        except (OSError, RuntimeError):
            pass

    # ========================================================
    # 3. "FILE IN DESKTOP" / "FILE FROM DOWNLOADS"
    # ========================================================

    location_patterns = [
        r"^(.*?)\s+(?:in|inside|from|on)\s+"
        r"(?:my\s+|the\s+)?"
        r"(desktop|downloads?|documents?|docs|pictures?|"
        r"photos?|images?|videos?|music|home)\s*$",

        r"^(.*?)\s+(?:in|inside|from|on)\s+"
        r"(desktop|downloads?|documents?|docs|pictures?|"
        r"photos?|images?|videos?|music|home)\s*$",
    ]

    for pattern in location_patterns:

        match = re.match(
            pattern,
            cleaned,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        filename = match.group(1).strip().strip("\"'")
        location = match.group(2).strip()

        directory = _location_directory(location)

        if not directory:
            continue

        found = _find_file_in_directory(
            directory,
            filename,
        )

        if found:
            return found

        raise FileNotFoundError(
            f"Could not find '{filename}' in '{directory}'."
        )

    # ========================================================
    # 4. NATURAL DIRECTORY
    # ========================================================

    directory = _location_directory(cleaned)

    if directory:
        return directory

    # ========================================================
    # 5. FILENAME ONLY
    #
    # Example:
    #
    # text.txt
    #
    # Search:
    #
    # Desktop
    # Downloads
    # Documents
    # Pictures
    # Videos
    # Music
    # OneDrive versions
    # ========================================================

    filename = cleaned.strip().strip("\"'")

    if (
        filename
        and "\\" not in filename
        and "/" not in filename
    ):

        for directory in _common_directories():

            found = _find_file_in_directory(
                directory,
                filename,
            )

            if found:
                return found

    # ========================================================
    # 6. CURRENT WORKING DIRECTORY
    # ========================================================

    try:

        cwd_candidate = (
            Path.cwd() / filename
        ).resolve()

        if cwd_candidate.exists():
            return cwd_candidate

    except (OSError, RuntimeError):
        pass

    # ========================================================
    # 7. FAIL
    # ========================================================

    raise FileNotFoundError(
        f"Could not find the requested path or file: "
        f"'{original}'"
    )



# ============================================================
# PROMPT-DRIVEN LOCAL FILESYSTEM NAVIGATION
# ============================================================

# The active directory is kept per conversation/session so a user can
# navigate naturally:
#   "list Documents"
#   "go into Projects"
#   "go back"
#   "list the files here"
# without asking the LLM to invent/remember filesystem paths.
_FILESYSTEM_NAVIGATION: dict[str, Path] = {}


def _filesystem_root() -> Path:
    """Return a useful local filesystem root for the current OS."""
    if os.name == "nt":
        # Prefer the current Windows system drive.
        drive = os.environ.get("SystemDrive", "C:")
        return Path(f"{drive}\\").resolve()
    return Path("/").resolve()


def _normalise_natural_target(value: str) -> str:
    """Remove conversational wrappers around a folder/file target."""
    value = str(value or "").strip().strip("\"'")
    value = re.sub(
        r"^(?:the|my|this|that)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()
    value = re.sub(
        r"\s+(?:folder|directory|location)$",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()
    return value.strip("\"'").strip()


def _extract_explicit_path(text: str) -> str:
    """Extract a Windows/Unix path from natural language."""
    text = str(text or "")

    windows = re.search(
        r'(?:"([^"]+)")|(?:\'([^\']+)\')|([A-Za-z]:[\\/][^"\']+)',
        text,
        flags=re.IGNORECASE,
    )
    if windows:
        candidate = next(
            (group for group in windows.groups() if group),
            "",
        ).strip().rstrip(".,;")
        if re.match(r"^[A-Za-z]:[\\/]", candidate):
            return candidate

    unix = re.search(
        r'(?<!\w)(/(?:[^/\s]+/)*[^/\s]+)',
        text,
    )
    if unix:
        return unix.group(1).rstrip(".,;")

    return ""


def _extract_location_from_prompt(prompt: str) -> str:
    """
    Extract the most likely folder/path target from a filesystem prompt.

    This deliberately handles common human wording instead of requiring
    the LLM to produce a tool-call argument.
    """
    text = str(prompt or "").strip()
    lowered = text.lower()

    explicit = _extract_explicit_path(text)
    if explicit:
        return explicit

    # Common Windows user-folder aliases.
    aliases = (
        "desktop",
        "downloads",
        "documents",
        "docs",
        "pictures",
        "photos",
        "images",
        "videos",
        "music",
        "home",
        "this pc",
        "my computer",
        "computer",
        "entire computer",
        "all drives",
        "root",
    )

    # Prefer a known location when it appears explicitly.
    for alias in aliases:
        if re.search(
            rf"\b(?:my|the|this)?\s*{re.escape(alias)}\b",
            lowered,
        ):
            return alias

    # "in/inside/from/on <target>"
    match = re.search(
        r"\b(?:in|inside|from|on|under|within)\s+"
        r"(.+?)(?:\s+(?:folder|directory))?"
        r"(?:\s+(?:and|then)\b|[?.!,;]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        candidate = _normalise_natural_target(match.group(1))
        candidate = re.sub(
            r"\b(?:please|can you|could you)\b",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()
        if candidate:
            return candidate

    # "go into/open/navigate to X"
    match = re.search(
        r"\b(?:go|move|navigate|enter)\s+(?:into|to)\s+"
        r"(.+?)(?:\s+(?:folder|directory))?"
        r"(?:[?.!,;]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return _normalise_natural_target(match.group(1))

    match = re.search(
        r"\b(?:open|browse|access)\s+"
        r"(.+?)(?:\s+(?:folder|directory))?"
        r"(?:[?.!,;]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        candidate = _normalise_natural_target(match.group(1))
        if candidate.lower() not in {
            "file",
            "the file",
            "a file",
            "files",
        }:
            return candidate

    return ""


def _resolve_navigation_target(
    target: str,
    current: Path | None,
) -> Path | None:
    """Resolve an extracted target against aliases, current directory, or home."""
    cleaned = _normalise_natural_target(target)

    if not cleaned:
        return current

    lowered = cleaned.lower()

    if lowered in {"root", "this pc", "my computer", "computer", "entire computer", "all drives"}:
        return _filesystem_root()

    # Reuse the existing robust natural-location resolver.
    natural = _location_directory(cleaned)
    if natural:
        return natural

    # Direct absolute/relative path.
    try:
        direct = Path(cleaned).expanduser()
        if direct.is_absolute() and direct.exists():
            return direct.resolve()
    except (OSError, RuntimeError):
        pass

    # Relative to the current navigated directory.
    if current is not None:
        try:
            candidate = (current / cleaned).resolve()
            if candidate.exists():
                return candidate
        except (OSError, RuntimeError):
            pass

    # Relative to the home directory.
    try:
        candidate = (_home_directory() / cleaned).resolve()
        if candidate.exists():
            return candidate
    except (OSError, RuntimeError):
        pass

    # If the target is just a folder/file name, search common user locations.
    for directory in _common_directories():
        try:
            candidate = (directory / cleaned).resolve()
            if candidate.exists():
                return candidate
        except (OSError, RuntimeError):
            continue

    return None


def _is_filesystem_prompt(prompt: str) -> bool:
    """Detect local filesystem intent without invoking an LLM."""
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
        "find ",
        "search for ",
        "locate file",
        "read file",
        "read the file",
        "open file",
        "open the file",
        "show the contents of",
        "file information",
        "file metadata",
        "properties of the file",
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

    if re.search(
        r"\b(?:list|show|browse)\s+(?:the\s+|my\s+)?"
        r"(?:desktop|downloads?|documents?|docs|pictures?|photos?|"
        r"images?|videos?|music|home|folders?|directories?)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True

    if re.search(r"\b(?:find|locate)\s+[^\s]+\.[A-Za-z0-9]{1,8}\b", text):
        return True

    return bool(
        re.search(r"\b[A-Za-z]:[\\/]", prompt or "")
        or re.search(r"(?<!\w)/(?:[^/\s]+/)+[^/\s]*", prompt or "")
    )


def _filesystem_action(prompt: str) -> str:
    """Determine the deterministic filesystem operation."""
    text = str(prompt or "").lower().strip()

    if any(
        x in text
        for x in (
            "go back",
            "parent folder",
            "parent directory",
            "one folder back",
            "up one folder",
            "go up",
        )
    ):
        return "back"

    if any(
        x in text
        for x in (
            "find file",
            "find the file",
            "find files",
            "search for file",
            "search for files",
            "search file",
            "search files",
            "locate file",
            "locate the file",
        )
    ) or bool(
        re.search(
            r"\b(?:find|locate|search\s+for)\s+[^\s]+\.[A-Za-z0-9]{1,8}\b",
            text,
            flags=re.IGNORECASE,
        )
    ):
        return "search"

    if any(
        x in text
        for x in (
            "file information",
            "file info",
            "file metadata",
            "metadata of the file",
            "properties of the file",
            "file properties",
            "details of the file",
        )
    ):
        return "info"

    if any(
        x in text
        for x in (
            "read file",
            "read the file",
            "open file",
            "open the file",
            "show the contents of",
            "display the contents of",
            "contents of the file",
        )
    ):
        return "read"

    if any(
        x in text
        for x in (
            "go into",
            "go to folder",
            "open folder",
            "enter folder",
            "navigate to folder",
            "browse folder",
        )
    ):
        return "enter"

    return "list"


def _extract_filename_from_prompt(prompt: str, target: str) -> str:
    """Extract a filename for deterministic recursive search."""
    text = str(prompt or "").strip()

    # Common wording: "find resume.pdf in Documents".
    if target:
        pattern = re.escape(target)
        before = re.split(
            rf"\b(?:in|inside|from|under)\s+{pattern}\b",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
    else:
        before = text

    match = re.search(
        r"\b(?:find|search(?:\s+for)?|locate)\s+"
        r"(?:the\s+|a\s+)?(?:file\s+)?[\"']?([^\"']+?)[\"']?"
        r"(?:\s+(?:in|inside|from|under)\b|$)",
        before,
        flags=re.IGNORECASE,
    )
    if match:
        candidate = match.group(1).strip()
        candidate = re.sub(
            r"\b(?:please|can you|could you)\b",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()
        if candidate:
            return candidate

    # If the prompt contains a recognizable extension, use that token.
    ext_match = re.search(
        r"([^\s\"']+\.(?:pdf|docx?|xlsx?|pptx?|txt|log|py|json|csv|jpg|jpeg|png|webp|bmp))",
        text,
        flags=re.IGNORECASE,
    )
    if ext_match:
        return ext_match.group(1).strip(".,;")

    return ""


def execute_filesystem_prompt(
    prompt: str,
    state_key: str = "",
) -> str:
    """
    Execute a natural-language local filesystem request deterministically.

    This is intentionally independent of the LLM. It makes basic filesystem
    navigation fast and reliable while keeping write operations out of this
    prompt-driven explorer.
    """
    if not _is_filesystem_prompt(prompt):
        return ""

    key = str(state_key or "default")
    current = _FILESYSTEM_NAVIGATION.get(key)

    action = _filesystem_action(prompt)
    target_text = _extract_location_from_prompt(prompt)

    # --------------------------------------------------------
    # Navigation
    # --------------------------------------------------------
    if action == "back":
        if current is None:
            current = _home_directory()
        else:
            parent = current.parent
            current = parent if parent != current else current
        _FILESYSTEM_NAVIGATION[key] = current
        action = "list"

    # --------------------------------------------------------
    # Resolve target
    # --------------------------------------------------------
    target = _resolve_navigation_target(
        target_text,
        current,
    )

    if target is None:
        if action == "list" and current is not None and any(
            x in str(prompt).lower()
            for x in ("here", "current folder", "current directory")
        ):
            target = current
        else:
            # Friendly aliases with no explicit target use the current location,
            # falling back to the user's home directory.
            target = current or _home_directory()

    # For an "enter/open/browse folder" request, the target itself becomes
    # the active directory.
    if action == "enter":
        if not target.is_dir():
            return f"Error: '{target}' is not a directory."
        _FILESYSTEM_NAVIGATION[key] = target
        return (
            f"Current directory: {target}\n\n"
            f"{list_directory(str(target))}"
        )

    # A request such as "open README.txt" should read the file, not list its
    # parent directory.
    if action in {"read", "info"}:
        if target.is_dir():
            if action == "info":
                return (
                    f"Path: {target}\n"
                    f"Name: {target.name or target}\n"
                    f"Type: directory"
                )
            _FILESYSTEM_NAVIGATION[key] = target
            return (
                f"Current directory: {target}\n\n"
                f"{list_directory(str(target))}"
            )

        if action == "read":
            return read_file(str(target))
        return get_file_info(str(target))

    if action == "search":
        if target.is_file():
            search_root = target.parent
        else:
            search_root = target

        filename = _extract_filename_from_prompt(
            prompt,
            target_text,
        )

        if not filename:
            return (
                "Error: please specify the file name to search for, "
                "for example 'find report.pdf in Documents'."
            )

        _FILESYSTEM_NAVIGATION[key] = search_root
        return search_files(
            str(search_root),
            filename,
        )

    # --------------------------------------------------------
    # LIST
    # --------------------------------------------------------
    if not target.is_dir():
        return f"Error: '{target}' is not a directory."

    _FILESYSTEM_NAVIGATION[key] = target

    result = list_directory(str(target))

    return (
        f"Current directory: {target}\n\n"
        f"{result}"
    )

# ============================================================
# LIST DIRECTORY
# ============================================================

def list_directory(path: str) -> str:
    """List files and directories at a local filesystem path."""

    try:
        target = _resolve_path(path)

    except Exception as exc:
        return f"Error: {exc}"

    if not target.is_dir():
        return (
            f"Error: '{target}' is not a directory."
        )

    entries = []

    try:

        for item in sorted(
            target.iterdir(),
            key=lambda p: (
                not p.is_dir(),
                p.name.lower(),
            ),
        ):

            kind = (
                "DIR"
                if item.is_dir()
                else "FILE"
            )

            entries.append(
                f"[{kind}] {item.name}"
            )

    except PermissionError:
        return (
            f"Error: Permission denied for "
            f"'{target}'."
        )

    except OSError as exc:
        return (
            f"Error reading directory "
            f"'{target}': {exc}"
        )

    if not entries:
        return (
            f"Directory '{target}' is empty."
        )

    return (
        f"Directory: {target}\n"
        + "\n".join(entries)
    )


# ============================================================
# SEARCH FILES
# ============================================================

def search_files(
    path: str,
    filename: str,
) -> str:
    """Search recursively for files matching a filename."""

    try:
        target = _resolve_path(path)

    except Exception as exc:
        return f"Error: {exc}"

    if not target.is_dir():
        return (
            f"Error: '{target}' is not a directory."
        )

    if not filename or not filename.strip():
        return (
            "Error: filename cannot be empty."
        )

    filename = (
        filename
        .strip()
        .strip("\"'")
    )

    matches = []

    try:

        # First exact recursive search.
        for item in target.rglob(filename):

            try:

                if item.is_file():

                    matches.append(
                        str(item.resolve())
                    )

                    if len(matches) >= 100:
                        break

            except (OSError, RuntimeError):
                continue

    except (PermissionError, OSError, RuntimeError):
        pass

    # --------------------------------------------------------
    # Case-insensitive fallback
    # --------------------------------------------------------

    if not matches:

        target_name = (
            Path(filename)
            .name
            .lower()
        )

        try:

            for item in target.rglob("*"):

                try:

                    if (
                        item.is_file()
                        and item.name.lower()
                        == target_name
                    ):

                        matches.append(
                            str(item.resolve())
                        )

                        if len(matches) >= 100:
                            break

                except (OSError, RuntimeError):
                    continue

        except (
            PermissionError,
            OSError,
            RuntimeError,
        ):
            pass

    if not matches:

        return (
            f"No files matching '{filename}' "
            f"were found under '{target}'."
        )

    return "\n".join(matches)


# ============================================================
# FILE INFORMATION
# ============================================================

def get_file_info(path: str) -> str:
    """Return metadata about a local filesystem file."""

    try:
        target = _resolve_path(path)

    except Exception as exc:
        return f"Error: {exc}"

    if not target.is_file():
        return (
            f"Error: '{target}' is not a file."
        )

    try:

        stat = target.stat()

        mime_type, _ = mimetypes.guess_type(
            str(target)
        )

        return (
            f"Path: {target}\n"
            f"Name: {target.name}\n"
            f"Size: {stat.st_size} bytes\n"
            f"Extension: "
            f"{target.suffix or 'none'}\n"
            f"MIME type: "
            f"{mime_type or 'unknown'}"
        )

    except OSError as exc:
        return (
            f"Error getting information for "
            f"'{target}': {exc}"
        )


# ============================================================
# READ FILE
# ============================================================

def read_file(path: str) -> str:
    """Read a local filesystem file as text or extract text from PDF."""

    try:
        target = _resolve_path(path)

    except Exception as exc:
        return f"Error: {exc}"

    if not target.is_file():
        return (
            f"Error: '{target}' is not a file."
        )

    # --------------------------------------------------------
    # 10 MB protection
    # --------------------------------------------------------

    max_size = 10 * 1024 * 1024

    try:
        file_size = target.stat().st_size

    except OSError as exc:
        return (
            f"Error accessing '{target}': {exc}"
        )

    if file_size > max_size:
        return (
            f"Error: '{target}' is larger than "
            f"the 10 MB read limit. "
            f"Use a document-specific processing "
            f"pipeline instead."
        )

    try:

        # ====================================================
        # PDF
        # ====================================================

        if target.suffix.lower() == ".pdf":

            document = fitz.open(
                str(target)
            )

            pages = []

            try:

                for page_number, page in enumerate(
                    document,
                    start=1,
                ):

                    text = page.get_text()

                    if text.strip():

                        pages.append(
                            f"--- Page {page_number} ---\n"
                            f"{text}"
                        )

            finally:
                document.close()

            if not pages:

                return (
                    f"PDF '{target}' contains no "
                    f"extractable text. "
                    f"It may be a scanned/"
                    f"image-only PDF."
                )

            return "\n\n".join(pages)

        # ====================================================
        # NORMAL TEXT FILE
        # ====================================================

        return target.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    except UnicodeDecodeError:

        return (
            f"Error: '{target}' does not appear "
            f"to be a readable text file."
        )

    except PermissionError:

        return (
            f"Error: Permission denied while "
            f"reading '{target}'."
        )

    except Exception as exc:

        return (
            f"Error reading '{target}': {exc}"
        )