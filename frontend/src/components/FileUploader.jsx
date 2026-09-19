import { X, FileText } from 'lucide-react';

const ACCEPT = '.pdf,.docx,.doc,.txt,.png,.jpg,.jpeg,.csv,.json,.py,.log';

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function FileUploader({ files, onChange, disabled, inputRef }) {
  const addFiles = (event) => {
    const picked = Array.from(event.target.files || []);
    if (picked.length) onChange([...files, ...picked]);
    event.target.value = '';
  };

  const removeFile = (index) => onChange(files.filter((_, i) => i !== index));

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={ACCEPT}
        onChange={addFiles}
        disabled={disabled}
        className="file-input-hidden"
        aria-label="Attach file"
      />
      {files.length > 0 && (
        <div className="attachments">
          {files.map((file, index) => (
            <div className="attachment-chip" key={`${file.name}-${index}`}>
              <FileText size={14} />
              <span className="attachment-chip__name">{file.name}</span>
              <span className="attachment-chip__size">{formatSize(file.size)}</span>
              <button
                type="button"
                className="attachment-chip__remove"
                onClick={() => removeFile(index)}
                disabled={disabled}
                aria-label={`Remove ${file.name}`}
              >
                <X size={13} />
              </button>
            </div>
          ))}
        </div>
      )}
    </>
  );
}
