import { useEffect, useMemo, useState } from "react";

import {
  Activity,
  CircleAlert,
  CircleCheck,
  Cpu,
  FileCog,
  FileText,
  GlobeLock,
  HardDrive,
  Network,
  Server,
  ShieldCheck,
  Sparkles,
  Wrench,
} from "lucide-react";

import { API_BASE_URL } from "../config";

const INITIAL_TELEMETRY = {
  guard_active: false,
  air_gapped: true,
  live_external_attempts: 0,
  live_blocked_attempts: 0,
  live_local_connections: 0,
};

function iconForEvent(eventName) {
  switch (eventName) {
    case "REQUEST_RECEIVED":
      return Server;
    case "REQUEST_UNDERSTOOD":
      return Network;
    case "PURIFYING_REQUEST":
      return ShieldCheck;
    case "PROCESSING_ATTACHMENT":
    case "OCR_PROCESSING":
      return FileText;
    case "ROUTING_REQUEST":
    case "ROUTE_SELECTED":
      return Activity;
    case "MODEL_SELECTED":
      return Cpu;
    case "EXECUTING_TOOL":
      return Wrench;
    case "RUNNING_PYTHON":
      return Sparkles;
    case "CREATING_WORD_DOCUMENT":
    case "CREATING_EXCEL_DOCUMENT":
    case "CREATING_POWERPOINT":
    case "DOCUMENT_CREATED":
      return FileCog;
    case "NETWORK_ACTIVITY":
      return GlobeLock;
    case "GENERATING_RESPONSE":
      return Cpu;
    case "COMPLETED":
      return CircleCheck;
    case "ERROR":
      return CircleAlert;
    default:
      return HardDrive;
  }
}

function eventLabel(eventName, message = "") {
  switch (eventName) {
    case "REQUEST_RECEIVED":
      return "Frontend → FastAPI";
    case "REQUEST_UNDERSTOOD":
      return "FastAPI accepted request";
    case "PURIFYING_REQUEST":
      return "FastAPI → Purifier";
    case "PROCESSING_ATTACHMENT":
      return "Local file processing";
    case "OCR_PROCESSING":
      return "Local OCR processing";
    case "ROUTING_REQUEST":
      return "Agent → Router";
    case "ROUTE_SELECTED":
      return "Local route selected";
    case "MODEL_SELECTED":
      return "Router → Ollama";
    case "EXECUTING_TOOL":
      return "Local tool execution";
    case "RUNNING_PYTHON":
      return "Local Python execution";
    case "CREATING_WORD_DOCUMENT":
      return "Creating Word document";
    case "CREATING_EXCEL_DOCUMENT":
      return "Creating Excel spreadsheet";
    case "CREATING_POWERPOINT":
      return "Creating PowerPoint";
    case "DOCUMENT_CREATED":
      return "Document created locally";
    case "NETWORK_ACTIVITY":
      return "Local network activity";
    case "GENERATING_RESPONSE":
      return "Ollama → Agent";
    case "COMPLETED":
      return "Agent → Frontend";
    case "ERROR":
      return "Local execution error";
    default:
      return message || "Local execution";
  }
}

function normalizeModel(message = "") {
  const text = String(message);
  const match = text.match(
    /(?:Using|Selected model:)\s+(.+?)(?:\s+\(fast path\))?$/i
  );
  return match?.[1]?.trim() || "local model";
}

function normalizeRoute(message = "") {
  const text = String(message);
  const match = text.match(/TASK_[A-Z0-9_]+/i);
  return (
    match?.[0] ||
    text.replace(/^Request routed to\s*/i, "").trim() ||
    "Local task route"
  );
}

function eventDetail(eventName, message = "") {
  const text = String(message || "").trim();

  switch (eventName) {
    case "REQUEST_RECEIVED":
      return "Request received by local API";
    case "REQUEST_UNDERSTOOD":
      return "FastAPI accepted the request";
    case "PURIFYING_REQUEST":
      return "Input sanitized locally";
    case "PROCESSING_ATTACHMENT":
      return text || "Attachment processed locally";
    case "OCR_PROCESSING":
      return text || "OCR running locally";
    case "ROUTING_REQUEST":
      return text || "Selecting local task route";
    case "ROUTE_SELECTED":
      return normalizeRoute(text);
    case "MODEL_SELECTED":
      return `${normalizeModel(text)} · localhost`;
    case "GENERATING_RESPONSE":
      return text || "Generating local response";
    case "COMPLETED":
      return "Response returned to frontend";
    case "ERROR":
      return text || "The local execution failed";
    default:
      return text || "Local execution";
  }
}

function isSupportedEvent(eventName) {
  return Boolean(
    eventName &&
      eventName !== "CONNECTED" &&
      eventName !== "PING"
  );
}

/*
 * Build the monitor from the REAL event stream of the latest request.
 * There is deliberately no fixed pipeline definition here.
 *
 * A request may therefore produce:
 *   Frontend → FastAPI
 *   Purifier
 *   Router
 *   Model
 *   RAG/tool #1
 *   Python/tool #2
 *   Model again
 *   Response
 *
 * If the backend emits it, the monitor can display it.
 */
function buildDynamicPipeline(events = [], isProcessing = false) {
  const source = Array.isArray(events) ? events : [];
  const rows = [];

  // Some backend events are emitted once as active and later as completed.
  // Keep one visual row for those lifecycle events and update that row.
  const lifecycleRows = new Map();

  source.forEach((event, sourceIndex) => {
    const eventName = String(event?.event || "");

    if (!isSupportedEvent(eventName)) {
      return;
    }

    const toolEvent = [
      "EXECUTING_TOOL",
      "RUNNING_PYTHON",
      "CREATING_WORD_DOCUMENT",
      "CREATING_EXCEL_DOCUMENT",
      "CREATING_POWERPOINT",
      "DOCUMENT_CREATED",
    ].includes(eventName);

    // Response generation can happen more than once in a tool workflow,
    // so keep each occurrence. Network events are also kept individually.
    const canUpdateLifecycle =
      !toolEvent &&
      eventName !== "NETWORK_ACTIVITY" &&
      eventName !== "GENERATING_RESPONSE";

    const Icon = iconForEvent(eventName);
    const status = String(event?.status || "active");

    let state = "done";
    if (eventName === "ERROR" || status === "error") {
      state = "blocked";
    } else if (status === "active") {
      state = "active";
    }

    const row = {
      id: canUpdateLifecycle
        ? eventName
        : `${eventName}-${sourceIndex}`,
      label: eventLabel(eventName, event?.message),
      detail: eventDetail(eventName, event?.message),
      state,
      icon: Icon,
    };

    if (canUpdateLifecycle && lifecycleRows.has(eventName)) {
      rows[lifecycleRows.get(eventName)] = row;
      return;
    }

    if (canUpdateLifecycle) {
      lifecycleRows.set(eventName, rows.length);
    }

    rows.push(row);
  });

  // While processing, if the backend has an active event, visually mark
  // earlier active rows as completed. The actual order still comes from SSE.
  if (isProcessing) {
    let activeIndex = -1;

    for (let index = rows.length - 1; index >= 0; index -= 1) {
      if (rows[index].state === "active") {
        activeIndex = index;
        break;
      }
    }

    if (activeIndex >= 0) {
      return rows.map((row, index) =>
        index < activeIndex && row.state === "active"
          ? { ...row, state: "done" }
          : row
      );
    }
  }

  return rows;
}

export default function NetworkMonitor({
  isProcessing = false,
  resetToken = "",
  events = [],
}) {
  const [telemetry, setTelemetry] = useState(INITIAL_TELEMETRY);
  const [connected, setConnected] = useState(false);

  // Conversation changes are a hard UI boundary. App clears events for a
  // new prompt as well, so the pipeline always represents the latest request.
  useEffect(() => {
    setConnected(false);
  }, [resetToken]);

  // Network telemetry is independent from the request pipeline. It is kept
  // small and only answers the security question: did anything external occur?
  useEffect(() => {
    let cancelled = false;

    async function refreshNetworkTelemetry() {
      try {
        const response = await fetch(
          `${API_BASE_URL}/api/v1/system/network-audit?limit=100`,
          { cache: "no-store" }
        );

        if (response.ok && !cancelled) {
          const data = await response.json();
          setTelemetry((previous) => ({
            ...previous,
            ...(data?.telemetry || {}),
          }));
          setConnected(true);
        }
      } catch (error) {
        if (!cancelled) {
          setConnected(false);
          console.error("Network telemetry error:", error);
        }
      }
    }

    refreshNetworkTelemetry();
    const interval = window.setInterval(refreshNetworkTelemetry, 750);

    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, []);

  const liveExternalAttempts = Number(
    telemetry?.live_external_attempts ?? 0
  );
  const liveBlockedAttempts = Number(
    telemetry?.live_blocked_attempts ?? 0
  );
  const guardActive = telemetry?.guard_active === true;
  const airGapped = telemetry?.air_gapped !== false;

  const externalActivity =
    liveExternalAttempts > 0 || liveBlockedAttempts > 0;

  const pipeline = useMemo(
    () => buildDynamicPipeline(events, isProcessing),
    [events, isProcessing]
  );

  const statusKind = !guardActive || !airGapped
    ? "checking"
    : externalActivity
      ? "blocked"
      : isProcessing
        ? "processing"
        : "local";

  const statusLabel =
    statusKind === "checking"
      ? "VERIFYING"
      : statusKind === "blocked"
        ? "EXTERNAL BLOCKED"
        : statusKind === "processing"
          ? "LOCAL PROCESSING"
          : "LOCAL ONLY";

  const statusTitle =
    statusKind === "checking"
      ? "Checking local guard"
      : statusKind === "blocked"
        ? "External access blocked"
        : statusKind === "processing"
          ? "Agent running locally"
          : "All systems local";

  const statusDetail =
    statusKind === "blocked"
      ? `${liveBlockedAttempts} external attempt${
          liveBlockedAttempts === 1 ? "" : "s"
        } blocked in the live window`
      : "No external communication detected";

  return (
    <aside
      className={`local-monitor local-monitor--${statusKind}`}
      aria-label="Live local execution monitor"
    >
      <div className="local-monitor__header">
        <div className="local-monitor__title">
          <ShieldCheck size={16} />
          <span>Local Execution</span>
        </div>

        <span className="local-monitor__live">
          <span className="local-monitor__live-dot" />
          {connected ? "LIVE" : "CONNECTING"}
        </span>
      </div>

      <section className="local-monitor__status">
        <div className="local-monitor__status-icon">
          {statusKind === "blocked" ? (
            <CircleAlert size={22} />
          ) : (
            <CircleCheck size={22} />
          )}
        </div>

        <div className="local-monitor__eyebrow">{statusLabel}</div>
        <div className="local-monitor__headline">{statusTitle}</div>
        <div className="local-monitor__detail">{statusDetail}</div>

        <div className="local-monitor__network-count">
          <span className="local-monitor__network-dot" />
          <strong>{externalActivity ? liveBlockedAttempts : 0}</strong>
          <span>{externalActivity ? "blocked" : "external calls"}</span>
        </div>
      </section>

      <section className="local-monitor__pipeline">
        <div className="local-monitor__section-label">LIVE PIPELINE</div>

        <div className="local-monitor__events">
          {pipeline.length === 0 ? (
            <div className="local-monitor__empty">
              <span className="local-monitor__empty-dot" />
              <div>
                <strong>
                  {isProcessing
                    ? "Starting local execution…"
                    : "No active pipeline"}
                </strong>
                <small>
                  {isProcessing
                    ? "Waiting for the first execution event"
                    : "Submit a prompt to begin"}
                </small>
              </div>
            </div>
          ) : (
            pipeline.map((item) => {
              const Icon = item.icon;

              return (
                <div
                  className={`local-monitor__event local-monitor__event--${item.state}`}
                  key={item.id}
                >
                  <span className="local-monitor__event-icon">
                    <Icon size={13} />
                  </span>

                  <div className="local-monitor__event-body">
                    <span className="local-monitor__event-label">
                      {item.label}
                    </span>
                    <span className="local-monitor__event-detail">
                      {item.detail}
                    </span>
                  </div>

                  {item.state === "active" && (
                    <span className="local-monitor__event-active-dot" />
                  )}

                  {item.state === "blocked" && (
                    <CircleAlert
                      className="local-monitor__event-check"
                      size={12}
                    />
                  )}

                  {item.state === "done" && (
                    <CircleCheck
                      className="local-monitor__event-check"
                      size={12}
                    />
                  )}
                </div>
              );
            })
          )}
        </div>
      </section>

      <section className="local-monitor__network">
        <div className="local-monitor__network-head">
          <span>NETWORK</span>
          <span>LOCAL</span>
        </div>

        <div
          className={`local-monitor__network-status ${
            externalActivity
              ? "local-monitor__network-status--blocked"
              : ""
          }`}
        >
          {externalActivity ? (
            <CircleAlert size={13} />
          ) : (
            <CircleCheck size={13} />
          )}

          <span>
            {externalActivity
              ? "External access blocked"
              : "0 external connections"}
          </span>
        </div>
      </section>
    </aside>
  );
}
