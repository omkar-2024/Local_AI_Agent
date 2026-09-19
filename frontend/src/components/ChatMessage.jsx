import { Download, FileText, Image as ImageIcon, Play } from "lucide-react";
import MarkdownLite from "./MarkdownLite";
import { API_BASE_URL } from "../config";

export default function ChatMessage({ message, onRunCode, codeExecutionDisabled = false }) {
  const isUser = message.role === "user";

  const generatedFiles =
    message.generated_files || [];

  const executionResults =
    message.execution_results || [];

  return (
    <div
      className={`msg-row ${
        isUser
          ? "msg-row--user"
          : "msg-row--assistant"
      }`}
    >
      <div
        className={`msg ${
          isUser
            ? "msg--user"
            : "msg--assistant"
        }`}
      >
        {isUser && Array.isArray(message.attachments) && message.attachments.length > 0 && (
          <div
            className="message-attachments"
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "8px",
              marginBottom: message.content ? "10px" : 0,
            }}
          >
            {message.attachments.map((attachment, index) => {
              const isImage =
                Boolean(attachment?.isImage) ||
                String(attachment?.type || "").startsWith("image/");

              if (isImage && attachment?.previewUrl) {
                return (
                  <div
                    key={attachment.id || `${attachment.name}-${index}`}
                    style={{
                      overflow: "hidden",
                      borderRadius: "12px",
                      border: "1px solid rgba(128,128,128,0.22)",
                      background: "rgba(128,128,128,0.08)",
                    }}
                  >
                    <img
                      src={attachment.previewUrl}
                      alt={attachment.name || "Attached image"}
                      style={{
                        display: "block",
                        width: "100%",
                        maxWidth: "420px",
                        maxHeight: "320px",
                        objectFit: "contain",
                      }}
                    />
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "7px",
                        padding: "7px 9px",
                        fontSize: "12px",
                        opacity: 0.8,
                      }}
                    >
                      <ImageIcon size={14} />
                      <span
                        style={{
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {attachment.name || "Image"}
                      </span>
                    </div>
                  </div>
                );
              }

              return (
                <div
                  key={attachment.id || `${attachment.name}-${index}`}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "9px",
                    padding: "9px 11px",
                    borderRadius: "10px",
                    border: "1px solid rgba(128,128,128,0.22)",
                    background: "rgba(128,128,128,0.08)",
                    maxWidth: "420px",
                  }}
                >
                  <FileText size={18} />
                  <div
                    style={{
                      minWidth: 0,
                      display: "flex",
                      flexDirection: "column",
                      gap: "2px",
                    }}
                  >
                    <span
                      style={{
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                        fontSize: "13px",
                        fontWeight: 600,
                      }}
                    >
                      {attachment.name || "Attachment"}
                    </span>
                    <span style={{ fontSize: "11px", opacity: 0.65 }}>
                      {attachment.size
                        ? `${(attachment.size / 1024).toFixed(attachment.size < 1024 * 1024 ? 1 : 2)} ${attachment.size < 1024 * 1024 ? "KB" : "MB"}`
                        : "Document"}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        <MarkdownLite
          text={message.content || ""}
          onRunCode={!isUser ? onRunCode : undefined}
          codeExecutionDisabled={codeExecutionDisabled}
        />

        {/* =========================================
            CODE EXECUTION OUTPUT
        ========================================= */}

        {!isUser &&
          executionResults.length > 0 && (
            <div
              className="execution-output"
              style={{
                marginTop: "14px",
                borderRadius: "10px",
                overflow: "hidden",
                border: "1px solid rgba(128,128,128,0.25)",
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  padding: "9px 12px",
                  fontWeight: 600,
                  fontSize: "13px",
                }}
              >
                <Play size={15} />
                Execution Output
              </div>

              {executionResults.map(
                (execution, index) => (
                  <pre
                    key={index}
                    style={{
                      margin: 0,
                      padding: "12px",
                      overflowX: "auto",
                      whiteSpace: "pre-wrap",
                      wordBreak: "break-word",
                      fontFamily:
                        "Consolas, Monaco, monospace",
                      fontSize: "13px",
                      lineHeight: 1.5,
                      borderTop:
                        "1px solid rgba(128,128,128,0.2)",
                    }}
                  >
                    {execution.result || "No output."}
                  </pre>
                )
              )}
            </div>
          )}

        {/* =========================================
            GENERATED FILES
        ========================================= */}

        {!isUser &&
          generatedFiles.length > 0 && (
            <div
              className="generated-files"
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "8px",
                marginTop: "14px",
              }}
            >
              {generatedFiles.map(
                (file, index) => {
                  const cleanPath = String(
                    file.path || ""
                  )
                    .replaceAll("\\", "/")
                    .replace(
                      /^.*?data\/workspace\//,
                      ""
                    );

                  const encodedPath = cleanPath
                    .split("/")
                    .map(encodeURIComponent)
                    .join("/");

                  const url =
                    `${API_BASE_URL}` +
                    `/api/v1/agent/download/` +
                    encodedPath;

                  const filename =
                    file.filename ||
                    cleanPath
                      .split("/")
                      .pop() ||
                    "generated_file";

                  return (
                    <a
                      key={`${filename}-${index}`}
                      href={url}
                      download={filename}
                      target="_blank"
                      rel="noreferrer"
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "10px",
                        textDecoration: "none",
                      }}
                    >
                      <FileText size={18} />

                      <span
                        style={{
                          flex: 1,
                          overflow: "hidden",
                          textOverflow:
                            "ellipsis",
                          whiteSpace:
                            "nowrap",
                        }}
                      >
                        {filename}
                      </span>

                      <Download size={17} />
                    </a>
                  );
                }
              )}
            </div>
          )}
      </div>
    </div>
  );
}