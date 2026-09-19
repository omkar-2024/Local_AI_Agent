export default function LoadingIndicator({ label = 'AI is thinking' }) {
  return (
    <div className="msg-row msg-row--assistant">
      <div className="msg msg--assistant msg--loading">
        <span className="typing-dots" aria-hidden="true">
          <span />
          <span />
          <span />
        </span>
        <span className="typing-label">{label}</span>
      </div>
    </div>
  );
}
