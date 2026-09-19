import {
  useEffect,
  useRef,
} from "react";

import ChatMessage from "./ChatMessage";
import LoadingIndicator from "./LoadingIndicator";
import ProcessingStatus from "./ProcessingStatus";


export default function ChatWindow({
  messages,
  loadingLabel,
  processingSteps = [],
  onRunCode,
  codeExecutionDisabled,
}) {

  const endRef =
    useRef(null);


  useEffect(() => {

    endRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });

  }, [
    messages,
    loadingLabel,
    processingSteps,
  ]);


  return (

    <div className="chat-window">

      <div className="chat-window__inner">

        {messages.map(
          (
            message,
            index
          ) => (

            <ChatMessage
              key={index}
              message={message}
              onRunCode={onRunCode}
              codeExecutionDisabled={
                codeExecutionDisabled
              }
            />

          )
        )}


        {loadingLabel &&
        processingSteps.length > 0 ? (

          <div className="processing-status-row">
            <ProcessingStatus
              steps={processingSteps}
            />
          </div>

        ) : loadingLabel ? (

          <div className="loading-status-row">
            <LoadingIndicator
              label={loadingLabel}
            />
          </div>

        ) : null}


        <div ref={endRef} />

      </div>

    </div>

  );
}