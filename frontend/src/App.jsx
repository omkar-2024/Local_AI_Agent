import {
  useEffect,
  useState,
} from "react";

import {
  Plus,
  MessageSquare,
  Trash2,
  PanelLeftClose,
  PanelLeftOpen,
  Sparkles,
  MoreHorizontal,
  Sun,
  Moon,
} from "lucide-react";

import NetworkMonitor from "./components/NetworkMonitor";
import ChatWindow from "./components/ChatWindow";
import ChatInput from "./components/ChatInput";

import {
  runAgent,
  executeCode,
} from "./services/api";

import {
  API_BASE_URL,
} from "./config";


const WELCOME = {
  role: "assistant",
  content:
    "Hello! I'm your local AI assistant.\nHow can I help you today?",
};


const INITIAL_STEPS = [
  {
    id: "request",
    label: "Understanding your request",
    status: "pending",
  },
  {
    id: "sanitize",
    label: "Sanitizing input",
    status: "pending",
  },
  {
    id: "attachment",
    label: "Processing attachment",
    status: "pending",
  },
  {
    id: "routing",
    label: "Routing request",
    status: "pending",
  },
  {
    id: "model",
    label: "Selecting local model",
    status: "pending",
  },
  {
    id: "tool",
    label: "Executing required tools",
    status: "pending",
  },
  {
    id: "response",
    label: "Generating final response",
    status: "pending",
  },
];


// ============================================================
// SESSION ID
// ============================================================

function getSessionId() {
  const key = "workbench-session-id";

  let sessionId =
    sessionStorage.getItem(key);

  if (!sessionId) {
    sessionId =
      `sess_${Date.now()}_${Math.random()
        .toString(36)
        .slice(2, 8)}`;

    sessionStorage.setItem(
      key,
      sessionId
    );
  }

  return sessionId;
}


// ============================================================
// CONVERSATION ID
// ============================================================

function createConversationId() {
  return (
    `conv_${Date.now()}_${Math.random()
      .toString(36)
      .slice(2, 10)}`
  );
}


function getConversationId() {
  const key =
    "workbench-conversation-id";

  let conversationId =
    localStorage.getItem(key);

  if (!conversationId) {
    conversationId =
      createConversationId();

    localStorage.setItem(
      key,
      conversationId
    );
  }

  return conversationId;
}


// ============================================================
// ASSISTANT CONTENT
// ============================================================

function buildAssistantContent(answer) {
  return (
    answer ||
    "The agent returned no final answer."
  );
}


// ============================================================
// LOAD CONVERSATION
// ============================================================

async function loadConversation(
  conversationId
) {
  const response =
    await fetch(
      `${API_BASE_URL}/api/v1/agent/conversations/${encodeURIComponent(
        conversationId
      )}/messages`
    );

  if (!response.ok) {
    throw new Error(
      "Failed to load conversation."
    );
  }

  return response.json();
}


// ============================================================
// LOAD CONVERSATION LIST
// ============================================================

async function loadConversations() {
  const response =
    await fetch(
      `${API_BASE_URL}/api/v1/agent/conversations`
    );

  if (!response.ok) {
    throw new Error(
      "Failed to load conversations."
    );
  }

  const data =
    await response.json();

  return Array.isArray(
    data?.conversations
  )
    ? data.conversations
    : [];
}


// ============================================================
// PURIFY WITH CONVERSATION ID
// ============================================================

async function purifyWithConversation(
  prompt,
  sessionId,
  conversationId,
  files
) {
  const formData =
    new FormData();

  formData.append(
    "prompt",
    prompt
  );

  formData.append(
    "session_id",
    sessionId
  );

  formData.append(
    "conversation_id",
    conversationId
  );

  for (
    const file of files
  ) {
    formData.append(
      "files",
      file
    );
  }

  const response =
    await fetch(
      `${API_BASE_URL}/api/v1/files/purify`,
      {
        method: "POST",
        body: formData,
      }
    );

  if (!response.ok) {
    let message =
      "Unable to process the request.";

    try {
      const errorData =
        await response.json();

      message =
        errorData.detail ||
        message;
    } catch {
      // Ignore invalid error body.
    }

    const error =
      new Error(message);

    error.status =
      response.status;

    throw error;
  }

  return response.json();
}


// ============================================================
// EVENT → STEP
// ============================================================

function mapEventToStep(event) {
  switch (event) {
    case "REQUEST_RECEIVED":
    case "REQUEST_UNDERSTOOD":
      return "request";

    case "PURIFYING_REQUEST":
      return "sanitize";

    case "PROCESSING_ATTACHMENT":
    case "OCR_PROCESSING":
      return "attachment";

    case "ROUTING_REQUEST":
    case "ROUTE_SELECTED":
      return "routing";

    case "MODEL_SELECTED":
      return "model";

    case "EXECUTING_TOOL":
    case "RUNNING_PYTHON":
    case "CREATING_WORD_DOCUMENT":
    case "CREATING_EXCEL_DOCUMENT":
    case "CREATING_POWERPOINT":
    case "DOCUMENT_CREATED":
      return "tool";

    case "GENERATING_RESPONSE":
      return "response";

    default:
      return null;
  }
}


// ============================================================
// UPDATE PROCESSING STEPS
// ============================================================

function updateSteps(
  previous,
  event
) {
  if (
    event.event ===
    "COMPLETED"
  ) {
    return previous.map(
      (step) => {
        if (
          step.status ===
          "error"
        ) {
          return step;
        }

        return {
          ...step,
          status: "completed",
        };
      }
    );
  }

  if (
    event.event ===
    "ERROR"
  ) {
    const activeIndex =
      previous.findIndex(
        (step) =>
          step.status ===
          "active"
      );

    if (
      activeIndex === -1
    ) {
      return previous;
    }

    return previous.map(
      (
        step,
        index
      ) => {
        if (
          index ===
          activeIndex
        ) {
          return {
            ...step,
            status: "error",
            label:
              event.message ||
              step.label,
          };
        }

        return step;
      }
    );
  }

  const stepId =
    mapEventToStep(
      event.event
    );

  if (!stepId) {
    return previous;
  }

  const index =
    previous.findIndex(
      (step) =>
        step.id ===
        stepId
    );

  if (index === -1) {
    return previous;
  }

  const next =
    previous.map(
      (step) => ({
        ...step,
      })
    );

  if (event.message) {
    next[index].label =
      event.message;
  }

  if (
    event.status ===
    "active"
  ) {
    next[index].status =
      "active";

    for (
      let i = 0;
      i < index;
      i++
    ) {
      if (
        next[i].status !==
        "error"
      ) {
        next[i].status =
          "completed";
      }
    }
  }

  if (
    event.status ===
    "completed"
  ) {
    next[index].status =
      "completed";

    for (
      let i = 0;
      i < index;
      i++
    ) {
      if (
        next[i].status !==
        "error"
      ) {
        next[i].status =
          "completed";
      }
    }
  }

  if (
    event.status ===
    "error"
  ) {
    next[index].status =
      "error";
  }

  return next;
}


// ============================================================
// FORMAT DATE
// ============================================================

function formatConversationDate(
  value
) {
  if (!value) {
    return "";
  }

  const date =
    new Date(value);

  if (
    Number.isNaN(
      date.getTime()
    )
  ) {
    return "";
  }

  const now =
    new Date();

  const sameDay =
    date.toDateString() ===
    now.toDateString();

  if (sameDay) {
    return date.toLocaleTimeString(
      [],
      {
        hour: "numeric",
        minute: "2-digit",
      }
    );
  }

  return date.toLocaleDateString(
    [],
    {
      month: "short",
      day: "numeric",
    }
  );
}


// ============================================================
// APP
// ============================================================

export default function App() {
  const [
    conversationId,
    setConversationId,
  ] = useState(
    getConversationId()
  );

  const [
    messages,
    setMessages,
  ] = useState([
    WELCOME,
  ]);

  const [
    conversations,
    setConversations,
  ] = useState([]);

  const [
    prompt,
    setPrompt,
  ] = useState("");

  const [
    files,
    setFiles,
  ] = useState([]);

  const [
    loadingLabel,
    setLoadingLabel,
  ] = useState("");

  const [
    processingSteps,
    setProcessingSteps,
  ] = useState(
    INITIAL_STEPS
  );

  // Request-scoped events for the live Local Execution monitor.
  // Cleared for every new prompt and every new conversation.
  const [
    monitorEvents,
    setMonitorEvents,
  ] = useState([]);

  const [
    sidebarOpen,
    setSidebarOpen,
  ] = useState(true);

  // ==========================================================
  // THEME
  // Light is the existing/default theme.
  // Dark is a clean near-black version of the same UI.
  // ==========================================================
  const [
    darkTheme,
    setDarkTheme,
  ] = useState(() => {
    try {
      return localStorage.getItem(
        "workbench-theme"
      ) === "dark";
    } catch {
      return false;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(
        "workbench-theme",
        darkTheme
          ? "dark"
          : "light"
      );
    } catch {
      // Ignore localStorage errors.
    }
  }, [darkTheme]);

  const toggleTheme = () => {
    setDarkTheme(
      (current) => !current
    );
  };

  const [
    historyLoading,
    setHistoryLoading,
  ] = useState(false);

  const busy =
    Boolean(
      loadingLabel
    );


  // ==========================================================
  // SAVE CURRENT CONVERSATION ID
  // ==========================================================

  useEffect(() => {
    localStorage.setItem(
      "workbench-conversation-id",
      conversationId
    );
  }, [
    conversationId,
  ]);


  // ==========================================================
  // LOAD HISTORY LIST
  // ==========================================================

  const refreshConversations =
    async () => {
      try {
        setHistoryLoading(
          true
        );

        const list =
          await loadConversations();

        setConversations(
          list
        );
      } catch (error) {
        console.error(
          "Unable to load conversation history:",
          error
        );
      } finally {
        setHistoryLoading(
          false
        );
      }
    };


  useEffect(() => {
    refreshConversations();
  }, []);


  // ==========================================================
  // LOAD CURRENT CHAT
  // ==========================================================

  useEffect(() => {
    let cancelled =
      false;

    async function loadSavedChat() {
      try {
        const data =
          await loadConversation(
            conversationId
          );

        if (cancelled) {
          return;
        }

        const storedMessages =
          Array.isArray(
            data?.messages
          )
            ? data.messages
            : [];

        if (
          storedMessages.length ===
          0
        ) {
          setMessages([
            WELCOME,
          ]);

          return;
        }

        const restored =
          storedMessages.map(
            (message) => ({
              role:
                message.role,

              content:
                message.content,

              generated_files:
                Array.isArray(
                  message.generated_files
                )
                  ? message.generated_files
                  : [],

              execution_results:
                Array.isArray(
                  message.execution_results
                )
                  ? message.execution_results
                  : [],

              attachments:
                Array.isArray(message.attachments)
                  ? message.attachments
                  : [],
            })
          );

        setMessages([
          WELCOME,
          ...restored,
        ]);
      } catch (error) {
        console.error(
          "Unable to load saved conversation:",
          error
        );

        if (!cancelled) {
          setMessages([
            WELCOME,
          ]);
        }
      }
    }

    loadSavedChat();

    return () => {
      cancelled = true;
    };
  }, [
    conversationId,
  ]);


  // ==========================================================
  // NEW CHAT
  // ==========================================================

  const handleNewChat =
    () => {
      if (busy) {
        return;
      }

      const newId =
        createConversationId();

      setConversationId(
        newId
      );

      localStorage.setItem(
        "workbench-conversation-id",
        newId
      );

      setMessages([
        WELCOME,
      ]);

      setPrompt("");
      setFiles([]);

      setProcessingSteps(
        INITIAL_STEPS.map(
          (step) => ({
            ...step,
            status:
              "pending",
          })
        )
      );

      setMonitorEvents([]);

      setLoadingLabel("");

      if (
        window.innerWidth <=
        900
      ) {
        setSidebarOpen(
          false
        );
      }
    };


  // ==========================================================
  // SELECT CHAT
  // ==========================================================
// console.log("HANDLE SEND IS RUNNING");
  const handleSelectConversation =
    (id) => {
      
      if (
        busy ||
        id === conversationId
      ) {
        return;
      }

      setConversationId(
        id
      );

      if (
        window.innerWidth <=
        900
      ) {
        setSidebarOpen(
          false
        );
      }
    };


  // ==========================================================
  // DELETE CHAT
  // ==========================================================

  const handleDeleteConversation =
    async (
      event,
      id
    ) => {
      event.stopPropagation();

      if (busy) {
        return;
      }

      const confirmed =
        window.confirm(
          "Delete this conversation?"
        );

      if (!confirmed) {
        return;
      }

      try {
        const response =
          await fetch(
            `${API_BASE_URL}/api/v1/agent/conversations/${encodeURIComponent(
              id
            )}`,
            {
              method: "DELETE",
            }
          );

        if (!response.ok) {
          throw new Error(
            "Failed to delete conversation."
          );
        }

        if (
          id ===
          conversationId
        ) {
          handleNewChat();
        }

        await refreshConversations();
      } catch (error) {
        console.error(
          "Unable to delete conversation:",
          error
        );

        window.alert(
          "Unable to delete this conversation."
        );
      }
    };


  // ==========================================================
  // SEND
  // ==========================================================

  const handleSend =
    async () => {
      console.log("HANDLE SEND IS RUNNING");
      const trimmedPrompt =
        prompt.trim();

      if (
        !trimmedPrompt &&
        files.length === 0
      ) {
        return;
      }

      const effectivePrompt =
        trimmedPrompt ||
        "Analyze the attached file and provide a detailed summary of its contents.";

      const attachedFiles =
        [...files];

      const userLabel =
        trimmedPrompt ||
        `Analyze ${attachedFiles.length} attached file${
          attachedFiles.length > 1
            ? "s"
            : ""
        }.`;

      setMessages(
        (previous) => [
          ...previous,
          {
            role: "user",
            content: userLabel,
            attachments: attachedFiles.map((file) => ({
              id: `${file.name}-${file.size}-${file.lastModified}-${Math.random().toString(36).slice(2, 8)}`,
              name: file.name || "attachment",
              size: file.size || 0,
              type: file.type || "application/octet-stream",
              isImage: Boolean(file.type && file.type.startsWith("image/")),
              previewUrl:
                file.type && file.type.startsWith("image/")
                  ? URL.createObjectURL(file)
                  : null,
            })),
          },
        ]
      );

      setPrompt("");
      setFiles([]);

      const sessionId =
        getSessionId();

      setProcessingSteps(
        INITIAL_STEPS.map(
          (step) => ({
            ...step,
            status:
              "pending",
          })
        )
      );

      setLoadingLabel(
        "Processing your request"
      );

      // Every submitted prompt starts a completely fresh execution trace.
      setMonitorEvents([]);

      // ========================================================
      // OPEN SSE BEFORE REQUEST
      // ========================================================

      const eventSource =
        new EventSource(
          `${API_BASE_URL}/api/v1/agent/events/${encodeURIComponent(
            sessionId
          )}`
        );

      eventSource.addEventListener(
        "progress",
        (event) => {
          try {
            const data =
              JSON.parse(
                event.data
              );

            // These are the live SSE events for this request only.
            // Historical audit events are intentionally not reloaded here.
            setMonitorEvents(
              (previous) => [
                ...previous,
                data,
              ]
            );

            setProcessingSteps(
              (previous) =>
                updateSteps(
                  previous,
                  data
                )
            );

            if (
              data.event ===
                "COMPLETED" &&
              data.data?.final ===
                true
            ) {
              eventSource.close();
            }

            if (
              data.event ===
              "ERROR"
            ) {
              eventSource.close();
            }
          } catch (error) {
            console.error(
              "Invalid progress event:",
              error
            );
          }
        }
      );

      eventSource.onerror =
        () => {
          // EventSource reconnects automatically.
          // Don't show a false error while backend is running.
        };

      let purified;

      // ========================================================
      // PURIFICATION
      // ========================================================

      try {
        purified =
          await purifyWithConversation(
            effectivePrompt,
            sessionId,
            conversationId,
            attachedFiles
          );
          console.log("=== PURIFIED RESPONSE ===");
console.log(JSON.stringify(purified, null, 2));
      } catch (error) {
        eventSource.close();
purified =
  await purifyWithConversation(
    effectivePrompt,
    sessionId,
    conversationId,
    attachedFiles
  );


        setMessages(
          (previous) => [
            ...previous,
            {
              role:
                "assistant",

              content:
                error.status ===
                0
                  ? "Unable to connect to the local AI backend.\nMake sure the FastAPI server is running."
                  : (
                    error.message ||
                    "Unable to process the uploaded file."
                  ),

              error:
                error.message,
            },
          ]
        );

        setLoadingLabel("");

        return;
      }

      // ========================================================
      // AGENT
      // ========================================================

      try {
        const result =
          await runAgent(
            purified
          );
          console.log("=== AGENT RESPONSE ===");
console.log(JSON.stringify(result, null, 2));

        const answer =
          result?.final_answer ||
          result?.answer ||
          result?.response ||
          "The agent returned no final answer.";

        const generatedFiles =
          Array.isArray(
            result?.generated_files
          )
            ? result.generated_files
            : [];

        const executionResults =
          Array.isArray(
            result?.execution_results
          )
            ? result.execution_results
            : [];

        setMessages(
          (previous) => [
            ...previous,
            {
              role:
                "assistant",

              content:
                buildAssistantContent(
                  answer
                ),

              generated_files:
                generatedFiles,

              execution_results:
                executionResults,
            },
          ]
        );

        // Refresh sidebar so the
        // conversation title/time updates.
        await refreshConversations();
      } catch (error) {
        setMessages(
          (previous) => [
            ...previous,
            {
              role:
                "assistant",

              content:
                error.status ===
                0
                  ? "Unable to connect to the local AI backend.\nMake sure the FastAPI server is running."
                  : "The local AI could not complete this request.",

              error:
                error.message,
            },
          ]
        );
      } finally {
        eventSource.close();
        setLoadingLabel("");
      }
    };


  // ==========================================================
  // LOCAL CODE EXECUTION
  // ==========================================================

  const handleRunCode = async (code, language) => {
    return executeCode(code, language);
  };

  // ==========================================================
  // UI
  // ==========================================================

  return (
    <div
      className={`app ${
        sidebarOpen
          ? "app--sidebar-open"
          : "app--sidebar-closed"
      }`}
      data-theme={
        darkTheme
          ? "dark"
          : "light"
      }
    >

      {/* ======================================================
          SIDEBAR
      ====================================================== */}

      <aside
        className={`sidebar ${
          sidebarOpen
            ? "sidebar--open"
            : "sidebar--closed"
        }`}
      >

        <div className="sidebar__top">

        

          <button
            className="sidebar__collapse"
            type="button"
            onClick={() =>
              setSidebarOpen(
                (value) => !value
              )
            }
            title={
              sidebarOpen
                ? "Collapse sidebar"
                : "Open sidebar"
            }
            aria-label={
              sidebarOpen
                ? "Collapse sidebar"
                : "Open sidebar"
            }
          >
            {sidebarOpen ? (
              <PanelLeftClose
                size={18}
              />
            ) : (
              <PanelLeftOpen
                size={18}
              />
            )}
          </button>

        </div>


        <div className="sidebar__content">

          <button
            className="new-chat-button"
            type="button"
            onClick={
              handleNewChat
            }
            disabled={busy}
            title="New chat"
          >
            <Plus size={18} />

            {sidebarOpen && (
              <span>
                New chat
              </span>
            )}
          </button>


          {sidebarOpen && (
            <div className="history-section">

              <div className="history-header">
                <span>
                  Recent chats
                </span>

                {conversations.length >
                  0 && (
                  <span className="history-count">
                    {conversations.length}
                  </span>
                )}
              </div>


              <div className="history-list">

                {historyLoading ? (
                  <div className="history-empty">
                    <span className="history-spinner" />
                    Loading chats...
                  </div>
                ) : conversations.length ===
                  0 ? (
                  <div className="history-empty">
                    <MessageSquare
                      size={18}
                    />

                    <span>
                      Your conversations
                      will appear here.
                    </span>
                  </div>
                ) : (
                  conversations.map(
                    (conversation) => (
                      <button
                        key={
                          conversation.id
                        }
                        type="button"
                        className={`history-item ${
                          conversation.id ===
                          conversationId
                            ? "history-item--active"
                            : ""
                        }`}
                        onClick={() =>
                          handleSelectConversation(
                            conversation.id
                          )
                        }
                        disabled={busy}
                      >

                        <div className="history-item__icon">
                          <MessageSquare
                            size={15}
                          />
                        </div>

                        <div className="history-item__body">

                          <div className="history-item__title">
                            {conversation.title ||
                              "New Chat"}
                          </div>

                          <div className="history-item__date">
                            {formatConversationDate(
                              conversation.updated_at
                            )}
                          </div>

                        </div>

                        <div className="history-item__actions">

                          <span className="history-item__more">
                            <MoreHorizontal
                              size={15}
                            />
                          </span>

                          <span
                            className="history-item__delete"
                            role="button"
                            tabIndex={busy ? -1 : 0}
                            onClick={(
                              event
                            ) =>
                              handleDeleteConversation(
                                event,
                                conversation.id
                              )
                            }
                            onKeyDown={(
                              event
                            ) => {
                              if (
                                event.key ===
                                "Enter"
                              ) {
                                handleDeleteConversation(
                                  event,
                                  conversation.id
                                );
                              }
                            }}
                            title="Delete conversation"
                            aria-label="Delete conversation"
                          >
                            <Trash2
                              size={14}
                            />
                          </span>

                        </div>

                      </button>
                    )
                  )
                )}

              </div>

            </div>
          )}

        </div>


        <div className="sidebar__footer">

          {sidebarOpen ? (
            <>
              <div className="local-status">
                <span className="local-status__dot" />

                <div>
                  <div className="local-status__title">
                    Local AI
                  </div>

                  <div className="local-status__text">
                    Air-gapped workspace
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="local-status local-status--collapsed">
              <span className="local-status__dot" />
            </div>
          )}

        </div>

      </aside>


      {/* ======================================================
          MAIN AREA
      ====================================================== */}

      <section className="main-area">

  <header className="topbar">

    <div className="topbar__left">

      <span className="topbar__title">
        {conversations.find(
          (item) =>
            item.id === conversationId
        )?.title ||
          "Sovereign Agentic AI Workbench"}
      </span>

    </div>


    <div className="topbar__actions">

      <button
        type="button"
        className={`theme-switch ${
          darkTheme
            ? "theme-switch--dark"
            : ""
        }`}
        onClick={toggleTheme}
        role="switch"
        aria-checked={darkTheme}
        aria-label={
          darkTheme
            ? "Switch to light theme"
            : "Switch to dark theme"
        }
        title={
          darkTheme
            ? "Switch to light theme"
            : "Switch to dark theme"
        }
      >
        <span className="theme-switch__sun">
          <Sun size={13} />
        </span>

        <span className="theme-switch__track">
          <span className="theme-switch__knob" />
        </span>

        <span className="theme-switch__moon">
          <Moon size={13} />
        </span>
      </button>

      <div className="topbar__status">

        <i className="status-dot" />

        Local AI

      </div>

    </div>

  </header>


  <div className="workspace-row">

    <main className="chat-shell">

      <ChatWindow
        messages={
          messages
        }

        loadingLabel={
          loadingLabel
        }

        processingSteps={
          processingSteps
        }

        onRunCode={
          handleRunCode
        }

        codeExecutionDisabled={
          busy
        }
      />


      <ChatInput
        prompt={
          prompt
        }

        setPrompt={
          setPrompt
        }

        files={
          files
        }

        setFiles={
          setFiles
        }

        onSend={
          handleSend
        }

        disabled={
          busy
        }
      />

    </main>


    <NetworkMonitor
      isProcessing={busy}
      resetToken={conversationId}
      events={monitorEvents}
    />

  </div>


</section>
</div>
);
}