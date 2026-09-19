import { useRef, useState } from 'react';
import {
  Paperclip,
  ArrowUp,
  Mic,
  Square,
  LoaderCircle,
} from 'lucide-react';
import FileUploader from './FileUploader';
import { transcribeAudio } from '../services/api';

export default function ChatInput({
  prompt,
  setPrompt,
  files,
  setFiles,
  onSend,
  disabled,
}) {
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);

  const audioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const processorNodeRef = useRef(null);
  const muteGainRef = useRef(null);

  const audioChunksRef = useRef([]);
  const sampleRateRef = useRef(48000);
  const recordingStartedAtRef = useRef(0);

  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [voiceError, setVoiceError] = useState('');

  const canSend =
    !disabled &&
    !isRecording &&
    !isTranscribing &&
    (prompt.trim().length > 0 || files.length > 0);

  const handleSubmit = (event) => {
    event.preventDefault();

    if (canSend) {
      onSend();
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();

      if (canSend) {
        onSend();
      }
    }
  };

  const autoGrow = (event) => {
    setPrompt(event.target.value);

    const element = event.target;

    element.style.height = 'auto';
    element.style.height = `${Math.min(element.scrollHeight, 200)}px`;
  };

  /*
   * Convert captured Float32 microphone samples into
   * a standard 16-bit PCM WAV file.
   */
  const encodeWav = (chunks, sampleRate) => {
    if (!chunks || chunks.length === 0) {
      throw new Error(
        'No microphone samples were captured. Please try recording again.'
      );
    }

    const totalSamples = chunks.reduce(
      (total, chunk) => total + chunk.length,
      0
    );

    if (totalSamples <= 0) {
      throw new Error(
        'No microphone samples were captured. Please check your microphone.'
      );
    }

    const mono = new Float32Array(totalSamples);

    let writePosition = 0;

    for (const chunk of chunks) {
      mono.set(chunk, writePosition);
      writePosition += chunk.length;
    }

    /*
     * Remove DC offset.
     */
    let mean = 0;

    for (let i = 0; i < mono.length; i += 1) {
      mean += mono[i];
    }

    mean /= mono.length;

    /*
     * Find peak after removing DC offset.
     */
    let peak = 0;

    for (let i = 0; i < mono.length; i += 1) {
      const value = Math.abs(mono[i] - mean);

      if (value > peak) {
        peak = value;
      }
    }

    /*
     * Normalize only if the microphone signal is unusually quiet.
     */
    let gain = 1;

    if (peak > 0.0001 && peak < 0.25) {
      gain = Math.min(4, 0.85 / peak);
    }

    const dataSize = mono.length * 2;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    const writeString = (offset, value) => {
      for (let i = 0; i < value.length; i += 1) {
        view.setUint8(offset + i, value.charCodeAt(i));
      }
    };

    /*
     * WAV header.
     */
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);

    writeString(8, 'WAVE');

    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);

    // PCM
    view.setUint16(20, 1, true);

    // Mono
    view.setUint16(22, 1, true);

    view.setUint32(24, sampleRate, true);

    // Byte rate = sample rate × channels × bytes per sample
    view.setUint32(28, sampleRate * 2, true);

    // Block align
    view.setUint16(32, 2, true);

    // Bits per sample
    view.setUint16(34, 16, true);

    writeString(36, 'data');
    view.setUint32(40, dataSize, true);

    let offset = 44;

    for (let i = 0; i < mono.length; i += 1) {
      let sample = (mono[i] - mean) * gain;

      sample = Math.max(-1, Math.min(1, sample));

      const pcm =
        sample < 0
          ? sample * 0x8000
          : sample * 0x7fff;

      view.setInt16(offset, pcm, true);

      offset += 2;
    }

    return new Blob([buffer], {
      type: 'audio/wav',
    });
  };

  /*
   * Stop microphone capture.
   *
   * A tiny delay is intentionally used before disconnecting the
   * ScriptProcessor so that the final microphone buffers can be
   * delivered to onaudioprocess.
   */
  const stopCapture = async () => {
    const stream = mediaStreamRef.current;
    const source = sourceNodeRef.current;
    const processor = processorNodeRef.current;
    const muteGain = muteGainRef.current;
    const audioContext = audioContextRef.current;

    /*
     * Give Web Audio one final moment to flush pending buffers.
     */
    await new Promise((resolve) => {
      window.setTimeout(resolve, 120);
    });

    try {
      processor?.disconnect();
      source?.disconnect();
      muteGain?.disconnect();

      if (stream) {
        stream.getTracks().forEach((track) => {
          try {
            track.stop();
          } catch {
            // Ignore already-stopped tracks.
          }
        });
      }

      if (
        audioContext &&
        audioContext.state !== 'closed'
      ) {
        await audioContext.close();
      }
    } finally {
      processorNodeRef.current = null;
      sourceNodeRef.current = null;
      muteGainRef.current = null;
      mediaStreamRef.current = null;
      audioContextRef.current = null;
    }

    return encodeWav(
      audioChunksRef.current,
      sampleRateRef.current
    );
  };

  const startRecording = async () => {
    if (
      disabled ||
      isRecording ||
      isTranscribing
    ) {
      return;
    }

    setVoiceError('');

    if (!navigator.mediaDevices?.getUserMedia) {
      setVoiceError(
        'Microphone recording is not supported by this browser.'
      );

      return;
    }

    try {
      const stream =
        await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            sampleRate: 48000,
            sampleSize: 16,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });

      const AudioContextClass =
        window.AudioContext ||
        window.webkitAudioContext;

      if (!AudioContextClass) {
        stream
          .getTracks()
          .forEach((track) => track.stop());

        throw new Error(
          'Web Audio is not supported by this browser.'
        );
      }

      const audioContext =
        new AudioContextClass();

      if (audioContext.state === 'suspended') {
        await audioContext.resume();
      }

      const source =
        audioContext.createMediaStreamSource(
          stream
        );

      /*
       * ScriptProcessor is retained because it works reliably
       * with the existing browser compatibility target.
       */
      const processor =
        audioContext.createScriptProcessor(
          4096,
          1,
          1
        );

      /*
       * IMPORTANT:
       *
       * Do not use absolute zero here.
       *
       * Some Chromium/browser audio paths can stop processing
       * a completely silent destination chain.
       *
       * An extremely tiny gain keeps the processing graph alive
       * while remaining effectively inaudible.
       */
      const muteGain =
        audioContext.createGain();

      muteGain.gain.value = 0.00001;

      const chunks = [];

      sampleRateRef.current =
        audioContext.sampleRate;

      audioChunksRef.current = chunks;

      mediaStreamRef.current = stream;
      audioContextRef.current = audioContext;
      sourceNodeRef.current = source;
      processorNodeRef.current = processor;
      muteGainRef.current = muteGain;

      recordingStartedAtRef.current =
        performance.now();

      processor.onaudioprocess = (event) => {
        const input =
          event.inputBuffer.getChannelData(0);

        if (!input || input.length === 0) {
          return;
        }

        /*
         * Copy the Float32Array because the browser reuses
         * the underlying AudioBuffer.
         */
        chunks.push(
          new Float32Array(input)
        );
      };

      /*
       * Start the processing graph.
       */
      source.connect(processor);

      processor.connect(muteGain);

      muteGain.connect(
        audioContext.destination
      );

      setIsRecording(true);
    } catch (error) {
      try {
        mediaStreamRef.current
          ?.getTracks()
          .forEach((track) => track.stop());
      } catch {
        // Ignore cleanup errors.
      }

      try {
        if (
          audioContextRef.current &&
          audioContextRef.current.state !==
            'closed'
        ) {
          await audioContextRef.current.close();
        }
      } catch {
        // Ignore cleanup errors.
      }

      audioContextRef.current = null;
      sourceNodeRef.current = null;
      processorNodeRef.current = null;
      muteGainRef.current = null;
      mediaStreamRef.current = null;

      if (error?.name === 'NotAllowedError') {
        setVoiceError(
          'Microphone permission was denied. Allow microphone access and try again.'
        );
      } else if (
        error?.name === 'NotFoundError'
      ) {
        setVoiceError(
          'No microphone was found. Connect or enable a microphone and try again.'
        );
      } else {
        setVoiceError(
          error?.message ||
            'Unable to access the microphone.'
        );
      }
    }
  };

  const finishRecording = async () => {
    if (!isRecording) {
      return;
    }

    setIsRecording(false);
    setIsTranscribing(true);
    setVoiceError('');

    try {
      /*
       * Make sure the user actually had the microphone active
       * for a reasonable amount of time.
       */
      const recordingDuration =
        (performance.now() -
          recordingStartedAtRef.current) /
        1000;

      if (recordingDuration < 0.35) {
        /*
         * Still clean up the microphone.
         */
        try {
          await stopCapture();
        } catch {
          // Ignore cleanup error.
        }

        audioChunksRef.current = [];

        throw new Error(
          'The recording is too short. Please speak for at least a moment.'
        );
      }

      /*
       * Wait for final audio buffers and create WAV.
       */
      const audioBlob =
        await stopCapture();

      const capturedChunks =
        audioChunksRef.current.length;

      audioChunksRef.current = [];

      if (capturedChunks === 0) {
        throw new Error(
          'No microphone audio was captured. Check your microphone and browser permission.'
        );
      }

      if (audioBlob.size < 1000) {
        throw new Error(
          'The microphone recording is empty. Please speak clearly and try again.'
        );
      }

      /*
       * Send the actual WAV to the local FastAPI server.
       */
      const result =
        await transcribeAudio(audioBlob);

      const text = String(
        result?.text || ''
      ).trim();

      if (!text) {
        throw new Error(
          'No speech could be recognized. Please speak clearly and try again.'
        );
      }

      /*
       * Put the transcription into the composer.
       * Do NOT automatically execute the prompt.
       */
      setPrompt(text);

      requestAnimationFrame(() => {
        const element =
          textareaRef.current;

        if (!element) {
          return;
        }

        element.style.height = 'auto';

        element.style.height =
          `${Math.min(
            element.scrollHeight,
            200
          )}px`;

        element.focus();
      });
    } catch (error) {
      setVoiceError(
        error?.message ||
          'Speech transcription failed.'
      );
    } finally {
      audioChunksRef.current = [];
      setIsTranscribing(false);
    }
  };

  return (
    <div className="composer">
      <FileUploader
        files={files}
        onChange={setFiles}
        disabled={
          disabled ||
          isRecording ||
          isTranscribing
        }
        inputRef={fileInputRef}
      />

      {voiceError ? (
        <div
          className="composer__voice-error"
          role="status"
        >
          {voiceError}
        </div>
      ) : null}

      <form
        className="composer__row"
        onSubmit={handleSubmit}
      >
        <button
          type="button"
          className={`composer__attach ${
            isRecording
              ? 'composer__voice-button--recording'
              : ''
          }`}
          onClick={
            isRecording
              ? finishRecording
              : startRecording
          }
          disabled={
            disabled ||
            isTranscribing
          }
          title={
            isTranscribing
              ? 'Transcribing locally'
              : isRecording
                ? 'Stop recording'
                : 'Voice input'
          }
          aria-label={
            isTranscribing
              ? 'Transcribing locally'
              : isRecording
                ? 'Stop recording'
                : 'Voice input'
          }
        >
          {isTranscribing ? (
            <LoaderCircle
              size={18}
              className="composer__voice-spinner"
            />
          ) : isRecording ? (
            <Square size={15} />
          ) : (
            <Mic size={18} />
          )}
        </button>

        <button
          type="button"
          className="composer__attach"
          onClick={() =>
            fileInputRef.current?.click()
          }
          disabled={
            disabled ||
            isRecording ||
            isTranscribing
          }
          title="Attach file"
          aria-label="Attach file"
        >
          <Paperclip size={18} />
        </button>

        <textarea
          ref={textareaRef}
          value={prompt}
          onChange={autoGrow}
          onKeyDown={handleKeyDown}
          placeholder={
            isRecording
              ? 'Listening...'
              : isTranscribing
                ? 'Transcribing locally...'
                : 'Ask anything...'
          }
          rows={1}
          disabled={
            disabled ||
            isRecording ||
            isTranscribing
          }
        />

        <button
          type="submit"
          className="composer__send"
          disabled={!canSend}
          aria-label="Send message"
        >
          <ArrowUp size={18} />
        </button>
      </form>
    </div>
  );
}