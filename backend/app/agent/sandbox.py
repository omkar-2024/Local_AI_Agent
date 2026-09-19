from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from app.core.config import settings


# Hard upper bound for every interactive run.
TIMEOUT_SECONDS = 10

# Maximum amount of output returned to the browser.
MAX_OUTPUT_CHARS = 20_000

# Only these languages are exposed by the Run button.
LANGUAGE_ALIASES = {
    "py": "python",
    "python": "python",
    "js": "javascript",
    "jsx": "javascript",
    "javascript": "javascript",
    "node": "javascript",
    "nodejs": "javascript",
    "c": "c",
    "cpp": "cpp",
    "c++": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "java": "java",
}


def _normalize_language(language: str) -> str:
    value = str(language or "").strip().lower()
    return LANGUAGE_ALIASES.get(value, value)


def _command_exists(*commands: str) -> str | None:
    for command in commands:
        found = shutil.which(command)
        if found:
            return found
    return None


def _clean_environment() -> dict[str, str]:
    """
    Give child processes a minimal environment.

    The child is intentionally launched without shell=True and with a
    temporary working directory. This is a safety boundary for the
    workbench, not a substitute for an OS/container sandbox.
    """
    allowed = {
        "PATH",
        "SystemRoot",
        "WINDIR",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "HOME",
        "LANG",
        "LC_ALL",
        "PYTHONIOENCODING",
        "PYTHONUNBUFFERED",
        "JAVA_HOME",
        "NODE_PATH",
    }

    env = {
        key: value
        for key, value in os.environ.items()
        if key in allowed
    }

    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    return env


def _truncate(value: str) -> str:
    if len(value) <= MAX_OUTPUT_CHARS:
        return value

    return (
        value[:MAX_OUTPUT_CHARS]
        + "\n\n[output truncated]"
    )


def _startupinfo() -> Any:
    """
    Hide compiler/interpreter windows on Windows.
    """
    if os.name != "nt":
        return None

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return startupinfo


def _creationflags() -> int:
    if os.name != "nt":
        return 0

    return (
        getattr(subprocess, "CREATE_NO_WINDOW", 0)
        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    )


def _run_process(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
) -> tuple[int, str, str, bool, int]:
    started = time.perf_counter()
    timed_out = False

    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_SECONDS,
            shell=False,
            startupinfo=_startupinfo(),
            creationflags=_creationflags(),
        )

        returncode = completed.returncode
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""

    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = -1

        stdout = (
            exc.stdout.decode("utf-8", "replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )

        stderr = (
            exc.stderr.decode("utf-8", "replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )

        stderr = (
            stderr
            + (
                "\nExecution terminated after "
                f"{TIMEOUT_SECONDS} seconds."
            )
        )

    except FileNotFoundError as exc:
        returncode = -1
        stdout = ""
        stderr = str(exc)

    except Exception as exc:
        returncode = -1
        stdout = ""
        stderr = f"{type(exc).__name__}: {exc}"

    duration_ms = int(
        (time.perf_counter() - started) * 1000
    )

    return (
        returncode,
        _truncate(stdout),
        _truncate(stderr),
        timed_out,
        duration_ms,
    )


def execute_code(
    code: str,
    language: str,
) -> dict[str, Any]:
    """
    Execute a supported code snippet locally.

    Supported:
      - Python
      - JavaScript / Node.js
      - C
      - C++
      - Java

    Every execution gets its own temporary directory under the local
    sandbox directory. Compilers/interpreters are invoked directly;
    shell=True is never used.
    """

    normalized = _normalize_language(language)

    if normalized not in {
        "python",
        "javascript",
        "c",
        "cpp",
        "java",
    }:
        raise ValueError(
            "Unsupported language. "
            "Supported languages: Python, JavaScript, C, C++, Java."
        )

    if not isinstance(code, str) or not code.strip():
        raise ValueError("Code cannot be empty.")

    run_id = uuid.uuid4().hex
    run_dir = (
        settings.SANDBOX_DIR
        / f"run_{run_id}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    env = _clean_environment()

    try:
        # ====================================================
        # PYTHON
        # ====================================================

        if normalized == "python":
            source = run_dir / "main.py"
            source.write_text(
                code,
                encoding="utf-8",
            )

            command = [
                sys.executable,
                "-I",
                str(source),
            ]

            (
                exit_code,
                stdout,
                stderr,
                timed_out,
                duration_ms,
            ) = _run_process(
                command,
                run_dir,
                env,
            )

        # ====================================================
        # JAVASCRIPT
        # ====================================================

        elif normalized == "javascript":
            node = _command_exists(
                "node",
                "node.exe",
            )

            if not node:
                raise RuntimeError(
                    "Node.js was not found on this computer."
                )

            source = run_dir / "main.js"
            source.write_text(
                code,
                encoding="utf-8",
            )

            command = [
                node,
                str(source),
            ]

            (
                exit_code,
                stdout,
                stderr,
                timed_out,
                duration_ms,
            ) = _run_process(
                command,
                run_dir,
                env,
            )

        # ====================================================
        # C
        # ====================================================

        elif normalized == "c":
            compiler = _command_exists(
                "gcc",
                "clang",
                "gcc.exe",
                "clang.exe",
            )

            if not compiler:
                raise RuntimeError(
                    "No C compiler was found. Install GCC or Clang."
                )

            source = run_dir / "main.c"
            binary = (
                run_dir / "main.exe"
                if os.name == "nt"
                else run_dir / "main"
            )

            source.write_text(
                code,
                encoding="utf-8",
            )

            compile_command = [
                compiler,
                str(source),
                "-O2",
                "-o",
                str(binary),
            ]

            (
                compile_exit,
                compile_stdout,
                compile_stderr,
                compile_timeout,
                compile_ms,
            ) = _run_process(
                compile_command,
                run_dir,
                env,
            )

            if compile_timeout or compile_exit != 0:
                exit_code = compile_exit
                stdout = compile_stdout
                stderr = compile_stderr
                timed_out = compile_timeout
                duration_ms = compile_ms
            else:
                (
                    exit_code,
                    stdout,
                    run_stderr,
                    timed_out,
                    run_ms,
                ) = _run_process(
                    [str(binary)],
                    run_dir,
                    env,
                )

                stderr = (
                    compile_stderr
                    + run_stderr
                )

                duration_ms += run_ms

        # ====================================================
        # C++
        # ====================================================

        elif normalized == "cpp":
            compiler = _command_exists(
                "g++",
                "clang++",
                "g++.exe",
                "clang++.exe",
            )

            if not compiler:
                raise RuntimeError(
                    "No C++ compiler was found. Install MinGW/G++ or Clang."
                )

            source = run_dir / "main.cpp"
            binary = (
                run_dir / "main.exe"
                if os.name == "nt"
                else run_dir / "main"
            )

            source.write_text(
                code,
                encoding="utf-8",
            )

            compile_command = [
                compiler,
                str(source),
                "-std=c++17",
                "-O2",
                "-o",
                str(binary),
            ]

            (
                compile_exit,
                compile_stdout,
                compile_stderr,
                compile_timeout,
                compile_ms,
            ) = _run_process(
                compile_command,
                run_dir,
                env,
            )

            if compile_timeout or compile_exit != 0:
                exit_code = compile_exit
                stdout = compile_stdout
                stderr = compile_stderr
                timed_out = compile_timeout
                duration_ms = compile_ms
            else:
                (
                    exit_code,
                    stdout,
                    run_stderr,
                    timed_out,
                    run_ms,
                ) = _run_process(
                    [str(binary)],
                    run_dir,
                    env,
                )

                stderr = (
                    compile_stderr
                    + run_stderr
                )

                duration_ms += run_ms

        # ====================================================
        # JAVA
        # ====================================================

        else:
            javac = _command_exists(
                "javac",
                "javac.exe",
            )
            java = _command_exists(
                "java",
                "java.exe",
            )

            if not javac or not java:
                raise RuntimeError(
                    "Java JDK was not found. "
                    "Install a JDK with javac and java."
                )

            source = run_dir / "Main.java"
            source.write_text(
                code,
                encoding="utf-8",
            )

            compile_command = [
                javac,
                "-encoding",
                "UTF-8",
                str(source),
            ]

            (
                compile_exit,
                compile_stdout,
                compile_stderr,
                compile_timeout,
                compile_ms,
            ) = _run_process(
                compile_command,
                run_dir,
                env,
            )

            if compile_timeout or compile_exit != 0:
                exit_code = compile_exit
                stdout = compile_stdout
                stderr = compile_stderr
                timed_out = compile_timeout
                duration_ms = compile_ms
            else:
                (
                    exit_code,
                    stdout,
                    run_stderr,
                    timed_out,
                    run_ms,
                ) = _run_process(
                    [
                        java,
                        "-cp",
                        str(run_dir),
                        "Main",
                    ],
                    run_dir,
                    env,
                )

                stderr = (
                    compile_stderr
                    + run_stderr
                )

                duration_ms += run_ms

        success = (
            exit_code == 0
            and not timed_out
        )

        combined_output = (
            stdout
            if stdout
            else stderr
        )

        return {
            "success": success,
            "language": normalized,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "output": _truncate(
                combined_output
                or "Program finished with no output."
            ),
            "timed_out": timed_out,
            "duration_ms": duration_ms,
        }

    finally:
        shutil.rmtree(
            run_dir,
            ignore_errors=True,
        )


# Backward-compatible Python helper used by the existing agent tool.
def execute_python_code(code: str) -> str:
    result = execute_code(
        code,
        "python",
    )

    return (
        f"exit_code: {result['exit_code']}\n"
        f"stdout:\n{result['stdout']}\n"
        f"stderr:\n{result['stderr']}"
    )
