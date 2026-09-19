import { API_BASE_URL } from "../config";

/**
 * ============================================================
 * CENTRAL API REQUEST HELPER
 * ============================================================
 */

async function request(path, options = {}) {
  let response;

  try {
    response = await fetch(`${API_BASE_URL}${path}`, options);
  } catch (error) {
    const networkError = new Error(
      "Unable to connect to the local AI backend. Make sure the FastAPI server is running."
    );

    networkError.status = 0;
    networkError.cause = error;

    throw networkError;
  }

  const contentType =
    response.headers.get("content-type") || "";

  let data;

  try {
    if (contentType.includes("application/json")) {
      data = await response.json();
    } else {
      data = await response.text();
    }
  } catch {
    data = null;
  }

  if (!response.ok) {
    let message =
      `Request failed with status ${response.status}`;

    if (
      typeof data === "string" &&
      data.trim()
    ) {
      message = data;
    } else if (
      data &&
      typeof data === "object"
    ) {
      if (
        typeof data.detail === "string"
      ) {
        message = data.detail;
      } else if (
        Array.isArray(data.detail)
      ) {
        message = data.detail
          .map((item) => {
            const location = item.loc
              ? item.loc.join(" → ")
              : "request";

            return (
              `${location}: ` +
              `${item.msg || "Validation error"}`
            );
          })
          .join("\n");
      } else {
        message = JSON.stringify(data);
      }
    }

    const error = new Error(message);

    error.status = response.status;
    error.details = data;

    throw error;
  }

  return data;
}


/**
 * ============================================================
 * STEP 1
 * PURIFY USER REQUEST
 *
 * Backend:
 * POST /api/v1/files/purify
 *
 * FastAPI expects:
 * prompt
 * session_id
 * conversation_id
 * files
 * ============================================================
 */

export async function purifyFiles(
  prompt,
  sessionId,
  files = [],
  conversationId = null
) {
  const formData = new FormData();

  /*
   * FastAPI declares prompt as Form(...),
   * therefore always send it.
   */
  formData.append(
    "prompt",
    prompt ?? ""
  );

  /*
   * Keep frontend session consistent
   */
  if (sessionId) {
    formData.append(
      "session_id",
      sessionId
    );
  }

  /*
   * Persistent SQLite conversation
   */
  if (conversationId) {
    formData.append(
      "conversation_id",
      conversationId
    );
  }

  /*
   * Backend expects "files"
   */
  for (const file of files) {
    if (file instanceof File) {
      formData.append(
        "files",
        file,
        file.name
      );
    }
  }

  /*
   * Debug information
   */
  console.log(
    "=== PURIFY REQUEST ==="
  );

  console.log(
    "URL:",
    `${API_BASE_URL}/api/v1/ingest/purify`
  );

  console.log(
    "Prompt:",
    prompt
  );

  console.log(
    "Session ID:",
    sessionId
  );

  console.log(
    "Conversation ID:",
    conversationId
  );

  console.log(
    "Files:",
    files
  );

  console.log(
    "File count:",
    files.length
  );

  for (
    const [key, value]
    of formData.entries()
  ) {
    if (value instanceof File) {
      console.log(
        `${key}:`,
        value.name,
        value.type,
        `${value.size} bytes`
      );
    } else {
      console.log(
        `${key}:`,
        value
      );
    }
  }

  /*
   * IMPORTANT:
   *
   * Do NOT manually set Content-Type.
   * Browser adds multipart boundary automatically.
   */

  return request(
    "/api/v1/files/purify",
    {
      method: "POST",
      body: formData,
    }
  );
}


/**
 * ============================================================
 * STEP 2
 * RUN AGENT
 *
 * Backend:
 * POST /api/v1/agent/run
 * ============================================================
 */

export async function runAgent(
  purifiedJson
) {
  return request(
    "/api/v1/agent/run",
    {
      method: "POST",

      headers: {
        "Content-Type":
          "application/json",
      },

      body: JSON.stringify({
        purified_json:
          purifiedJson,
      }),
    }
  );
}


/**
 * ============================================================
 * STEP 2 - STREAMING HELPER
 *
 * Kept for compatibility.
 *
 * Current live progress implementation
 * uses /agent/events/{session_id} from App.jsx.
 *
 * ============================================================
 */

export async function* runAgentStream(
  purifiedJson
) {
  let response;

  try {
    response = await fetch(
      `${API_BASE_URL}/api/v1/agent/run-stream`,
      {
        method: "POST",

        headers: {
          "Content-Type":
            "application/json",
        },

        body: JSON.stringify({
          purified_json:
            purifiedJson,
        }),
      }
    );
  } catch (error) {
    const networkError = new Error(
      "Unable to connect to the local AI backend. Make sure the FastAPI server is running."
    );

    networkError.status = 0;
    networkError.cause = error;

    throw networkError;
  }

  if (!response.ok) {
    let errorMessage =
      `Request failed with status ${response.status}`;

    try {
      const errorData =
        await response.json();

      if (errorData.detail) {
        errorMessage =
          errorData.detail;
      }
    } catch {
      // Ignore JSON parsing errors
    }

    throw new Error(
      errorMessage
    );
  }

  if (!response.body) {
    throw new Error(
      "Streaming response body is unavailable."
    );
  }

  const reader =
    response.body.getReader();

  const decoder =
    new TextDecoder();

  let buffer = "";

  try {
    while (true) {
      const {
        done,
        value,
      } = await reader.read();

      if (done) {
        break;
      }

      buffer += decoder.decode(
        value,
        {
          stream: true,
        }
      );

      const chunks =
        buffer.split("\n\n");

      buffer =
        chunks.pop() || "";

      for (
        const chunk of chunks
      ) {
        if (
          chunk.startsWith(
            "data: "
          )
        ) {
          const jsonStr =
            chunk.slice(6);

          try {
            const event =
              JSON.parse(
                jsonStr
              );

            yield event;
          } catch (error) {
            console.error(
              "Failed to parse SSE event:",
              jsonStr,
              error
            );
          }
        }
      }
    }

    if (
      buffer.startsWith(
        "data: "
      )
    ) {
      const jsonStr =
        buffer.slice(6);

      try {
        yield JSON.parse(
          jsonStr
        );
      } catch (error) {
        console.error(
          "Failed to parse final SSE event:",
          jsonStr,
          error
        );
      }
    }
  } finally {
    reader.releaseLock();
  }
}


/**
 * ============================================================
 * CODE EXECUTION
 *
 * Backend:
 * POST /api/v1/agent/execute-code
 *
 * Used by the Run button in generated code blocks.
 *
 * Supported languages depend on the backend sandbox:
 * Python, JavaScript, C, C++, Java, etc.
 *
 * Request:
 * {
 *   code: "...",
 *   language: "python"
 * }
 *
 * Response is returned directly from the backend.
 * ============================================================
 */

export async function executeCode(
  code,
  language = "python"
) {
  const normalizedCode =
    typeof code === "string"
      ? code
      : String(code ?? "");

  const normalizedLanguage =
    typeof language === "string" &&
    language.trim()
      ? language.trim().toLowerCase()
      : "python";

  if (!normalizedCode.trim()) {
    const error = new Error(
      "No code was provided for execution."
    );

    error.status = 400;

    throw error;
  }

  return request(
    "/api/v1/agent/execute-code",
    {
      method: "POST",

      headers: {
        "Content-Type":
          "application/json",
      },

      body: JSON.stringify({
        code: normalizedCode,
        language: normalizedLanguage,
      }),
    }
  );
}


/**
 * ============================================================
 * LOCAL SPEECH-TO-TEXT
 *
 * Backend:
 * POST /api/v1/speech/transcribe
 *
 * Audio is sent only to the local FastAPI server. The server
 * runs the downloaded Whisper model locally.
 * ============================================================
 */

export async function transcribeAudio(audioBlob) {
  if (!(audioBlob instanceof Blob) || audioBlob.size === 0) {
    const error = new Error("No audio was recorded.");
    error.status = 400;
    throw error;
  }

  const formData = new FormData();
  formData.append("audio", audioBlob, "voice-input.wav");

  return request("/api/v1/speech/transcribe", {
    method: "POST",
    body: formData,
  });
}


/**
 * ============================================================
 * HEALTH CHECK
 * ============================================================
 */

export function checkHealth() {
  return request(
    "/health"
  );
}


/**
 * ============================================================
 * COMPLETE WORKFLOW HELPER
 * ============================================================
 */

export async function sendMessage(
  prompt,
  sessionId,
  files = [],
  conversationId = null
) {
  const purified =
    await purifyFiles(
      prompt,
      sessionId,
      files,
      conversationId
    );

  return runAgent(
    purified
  );
}