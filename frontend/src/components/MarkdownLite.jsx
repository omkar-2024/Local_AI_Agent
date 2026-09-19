import { useState } from "react";
import { Play, RotateCw, CheckCircle2, XCircle } from "lucide-react";

// Lightweight Markdown renderer with local code execution support.
// Code is executed only when the user explicitly clicks Run.

const LANGUAGE_ALIASES = {
  py: "python",
  python: "python",
  js: "javascript",
  jsx: "javascript",
  javascript: "javascript",
  node: "javascript",
  nodejs: "javascript",
  c: "c",
  cpp: "cpp",
  "c++": "cpp",
  cc: "cpp",
  cxx: "cpp",
  java: "java",
};

const RUNNABLE_LANGUAGES = new Set([
  "python",
  "javascript",
  "c",
  "cpp",
  "java",
]);

function normalizeLanguage(language) {
  const value = String(language || "").trim().toLowerCase();
  return LANGUAGE_ALIASES[value] || value;
}

function renderInline(text, keyPrefix) {
  const parts = text
    .split(/(`[^`]+`|\*\*[^*]+\*\*)/g)
    .filter(Boolean);

  return parts.map((part, i) => {
    if (part.startsWith("`") && part.endsWith("`")) {
      return (
        <code key={`${keyPrefix}-${i}`}>
          {part.slice(1, -1)}
        </code>
      );
    }

    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={`${keyPrefix}-${i}`}>
          {part.slice(2, -2)}
        </strong>
      );
    }

    return (
      <span key={`${keyPrefix}-${i}`}>
        {part}
      </span>
    );
  });
}

function renderBlock(block, blockIndex) {
  const lines = block.split("\n");

  const isUnordered = lines.every(
    (line) =>
      /^\s*[-*]\s+/.test(line) ||
      line.trim() === ""
  );

  const isOrdered = lines.every(
    (line) =>
      /^\s*\d+[.)]\s+/.test(line) ||
      line.trim() === ""
  );

  if (
    isUnordered &&
    lines.some((line) => line.trim())
  ) {
    const items = lines
      .filter((line) => line.trim())
      .map((line) =>
        line.replace(/^\s*[-*]\s+/, "")
      );

    return (
      <ul key={blockIndex}>
        {items.map((item, i) => (
          <li key={i}>
            {renderInline(
              item,
              `${blockIndex}-${i}`
            )}
          </li>
        ))}
      </ul>
    );
  }

  if (
    isOrdered &&
    lines.some((line) => line.trim())
  ) {
    const items = lines
      .filter((line) => line.trim())
      .map((line) =>
        line.replace(
          /^\s*\d+[.)]\s+/,
          ""
        )
      );

    return (
      <ol key={blockIndex}>
        {items.map((item, i) => (
          <li key={i}>
            {renderInline(
              item,
              `${blockIndex}-${i}`
            )}
          </li>
        ))}
      </ol>
    );
  }

  return (
    <p key={blockIndex}>
      {lines.map((line, i) => (
        <span key={i}>
          {renderInline(
            line,
            `${blockIndex}-${i}`
          )}
          {i < lines.length - 1 && <br />}
        </span>
      ))}
    </p>
  );
}

function CodeExecutionPanel({
  result,
}) {
  if (!result) {
    return null;
  }

  const success =
    Boolean(result.success) &&
    Number(result.exit_code) === 0 &&
    !result.timed_out;

  const stdout =
    String(result.stdout || "");

  const stderr =
    String(result.stderr || "");

  const output =
    stdout || stderr || result.output || "No output.";

  return (
    <div
      className={`code-execution-result ${
        success
          ? "code-execution-result--success"
          : "code-execution-result--error"
      }`}
    >
      <div className="code-execution-result__header">
        <div className="code-execution-result__title">
          {success ? (
            <CheckCircle2 size={14} />
          ) : (
            <XCircle size={14} />
          )}

          <span>
            {success
              ? "Output"
              : "Execution error"}
          </span>
        </div>

        <div className="code-execution-result__meta">
          {result.exit_code !== undefined &&
            `exit ${result.exit_code}`}
          {result.duration_ms !== undefined &&
            ` · ${result.duration_ms} ms`}
        </div>
      </div>

      <pre className="code-execution-result__body">
        {output}
      </pre>

      {stderr && stdout && (
        <details className="code-execution-result__stderr">
          <summary>stderr</summary>
          <pre>{stderr}</pre>
        </details>
      )}

      {result.timed_out && (
        <div className="code-execution-result__warning">
          Execution stopped because the time limit was exceeded.
        </div>
      )}
    </div>
  );
}

export default function MarkdownLite({
  text,
  onRunCode,
  codeExecutionDisabled = false,
}) {
  const content = String(text ?? "");

  const segments = content
    .split(/(```[\s\S]*?```)/g)
    .filter((segment) => segment !== "");

  const [runningBlock, setRunningBlock] =
    useState(null);

  const [results, setResults] =
    useState({});

  const runBlock = async (
    blockKey,
    code,
    language
  ) => {
    if (
      !onRunCode ||
      runningBlock !== null
    ) {
      return;
    }

    const normalized =
      normalizeLanguage(language);

    if (
      !RUNNABLE_LANGUAGES.has(normalized)
    ) {
      return;
    }

    setRunningBlock(blockKey);

    try {
      const result = await onRunCode(
        code,
        normalized
      );

      setResults((previous) => ({
        ...previous,
        [blockKey]: result,
      }));
    } catch (error) {
      setResults((previous) => ({
        ...previous,
        [blockKey]: {
          success: false,
          exit_code: -1,
          stdout: "",
          stderr:
            error?.message ||
            "Unable to execute code.",
          output:
            error?.message ||
            "Unable to execute code.",
        },
      }));
    } finally {
      setRunningBlock(null);
    }
  };

  return (
    <div className="md">
      {segments.map(
        (segment, index) => {
          if (
            segment.startsWith("```")
          ) {
            const match =
              segment.match(
                /^```([^\s]*)?\n?([\s\S]*?)```$/
              );

            const lang =
              match?.[1] || "";

            const code =
              (
                match?.[2] ??
                segment.replace(
                  /^```|```$/g,
                  ""
                )
              ).replace(/\n$/, "");

            const normalized =
              normalizeLanguage(lang);

            const canRun =
              Boolean(onRunCode) &&
              RUNNABLE_LANGUAGES.has(
                normalized
              );

            const blockKey =
              `code-${index}`;

            const isRunning =
              runningBlock === blockKey;

            return (
              <div
                key={blockKey}
                className="md-code-container"
              >
                <pre
                  data-lang={lang}
                  className="md-code-block"
                >
                  <div className="md-code-block__toolbar">
                    <span>
                      {lang || "code"}
                    </span>

                    {canRun && (
                      <button
                        type="button"
                        className="code-run-button"
                        onClick={() =>
                          runBlock(
                            blockKey,
                            code,
                            normalized
                          )
                        }
                        disabled={
                          codeExecutionDisabled ||
                          runningBlock !== null
                        }
                        title={
                          codeExecutionDisabled
                            ? "Code execution is temporarily disabled"
                            : isRunning
                              ? "Running code"
                              : "Run code locally"
                        }
                      >
                        {isRunning ? (
                          <RotateCw
                            size={13}
                            className="code-run-spinner"
                          />
                        ) : (
                          <Play size={13} />
                        )}

                        {isRunning
                          ? "Running"
                          : "Run"}
                      </button>
                    )}
                  </div>

                  <code>
                    {code}
                  </code>
                </pre>

                <CodeExecutionPanel
                  result={results[blockKey]}
                />
              </div>
            );
          }

          const blocks =
            segment
              .split(/\n{2,}/)
              .filter(
                (block) =>
                  block.trim() !== ""
              );

          return blocks.map(
            (block, i) =>
              renderBlock(
                block,
                `${index}-${i}`
              )
          );
        }
      )}
    </div>
  );
}
