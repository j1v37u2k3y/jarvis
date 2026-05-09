/**
 * Continuous audio capture at 16kHz mono — maintains a ring buffer of
 * the last N seconds of Float32 samples and exposes two snapshot modes:
 *
 * 1. `snapshot(seconds)` — grab the last N seconds for piggybacking onto
 *    a final transcript so the server can verify the speaker.
 * 2. `recordSample(seconds)` — record N fresh seconds for enrollment.
 *    Resolves with the bytes when the duration is up.
 *
 * Both return raw int16 PCM (base64-encoded for `snapshot`, as a Blob
 * for `recordSample`). The backend's voice_id.wav.decode() accepts both
 * headerless int16 PCM and WAV.
 */

import pcmProcessorUrl from "./pcm-processor.js?url";

// 16kHz is resemblyzer's native rate. Creating the AudioContext at
// this rate makes the browser do the resampling for us — no DSP on
// the main thread.
const SAMPLE_RATE = 16_000;

// Ring buffer holds ~5s of audio at 16kHz. Web Speech API's onresult
// fires ~1-2s after the user stops speaking (endpoint detection), so a
// 4s snapshot at that moment still captures the utterance with margin.
const RING_BUFFER_SECONDS = 5;
const RING_BUFFER_SAMPLES = SAMPLE_RATE * RING_BUFFER_SECONDS;

export interface AudioCapture {
  start(): Promise<void>;
  stop(): void;
  /** Pause the worklet without releasing the mic — cheap and reversible. */
  suspend(): Promise<void>;
  /** Resume after suspend(). No-op if never started or already running. */
  resume(): Promise<void>;
  /** Last `seconds` of audio as base64 int16 PCM. null if no audio yet. */
  snapshot(seconds: number): string | null;
  /** Record `seconds` of fresh audio starting now, return as a WAV Blob. */
  recordSample(seconds: number): Promise<Blob>;
  /** Voice-activity-detected recording — waits for speech onset, stops on
   *  trailing silence. Streams live RMS to `onLevel` for UI metering. */
  recordSampleVAD(opts?: VADOptions): Promise<VADResult>;
  isRunning(): boolean;
}

export interface VADOptions {
  /** Audio captured before speech onset, backfilled into the result. */
  prerollSeconds?: number;
  /** Stop recording after this much trailing silence. */
  silenceMs?: number;
  /** Hard cap on total recording length (preroll + speech + tail silence). */
  maxSeconds?: number;
  /** Reject as too-short if voiced samples fall below this. */
  minSpeechMs?: number;
  /** Reject as no-speech if onset doesn't happen within this window. */
  onsetTimeoutMs?: number;
  /** RMS above this counts as speech. */
  rmsThreshold?: number;
  /** Live RMS callback — fires every worklet frame (~125Hz). */
  onLevel?: (rms: number) => void;
}

export type VADReason = "ok" | "no-speech" | "too-short";

export interface VADResult {
  blob: Blob | null;
  reason: VADReason;
}

export function createAudioCapture(): AudioCapture {
  let audioCtx: AudioContext | null = null;
  let workletNode: AudioWorkletNode | null = null;
  let source: MediaStreamAudioSourceNode | null = null;
  let stream: MediaStream | null = null;

  // Ring buffer — Float32Array sized to hold RING_BUFFER_SAMPLES.
  // `writePos` wraps; `filled` tracks how much real data exists (for
  // the first few seconds after start()).
  const ring = new Float32Array(RING_BUFFER_SAMPLES);
  let writePos = 0;
  let filled = 0;

  // When recordSample() is active, we collect frames into this array
  // until the target duration is reached.
  let activeRecording: {
    chunks: Float32Array[];
    targetSamples: number;
    gathered: number;
    resolve: (blob: Blob) => void;
  } | null = null;

  // Voice-activity-detected recording state. Lives parallel to activeRecording.
  let activeVAD: {
    state: "waiting" | "recording";
    rmsThreshold: number;
    silenceMsTarget: number;
    maxSamples: number;
    minSpeechSamples: number;
    prerollSamples: number;
    onLevel?: (rms: number) => void;
    // Rolling preroll buffer maintained while waiting for onset.
    prerollChunks: Float32Array[];
    prerollLength: number;
    // Captured audio after onset.
    chunks: Float32Array[];
    recordedSamples: number;
    voicedSamples: number;
    trailingSilenceSamples: number;
    onsetTimer: ReturnType<typeof setTimeout> | null;
    resolve: (result: VADResult) => void;
  } | null = null;

  function handleFrame(frame: Float32Array) {
    // Append to ring
    for (let i = 0; i < frame.length; i++) {
      ring[writePos] = frame[i];
      writePos = (writePos + 1) % RING_BUFFER_SAMPLES;
    }
    filled = Math.min(filled + frame.length, RING_BUFFER_SAMPLES);

    // Feed active recording
    if (activeRecording) {
      activeRecording.chunks.push(frame);
      activeRecording.gathered += frame.length;
      if (activeRecording.gathered >= activeRecording.targetSamples) {
        const flat = concatFloat32(activeRecording.chunks, activeRecording.targetSamples);
        const wav = encodeWav(flat, SAMPLE_RATE);
        activeRecording.resolve(new Blob([wav], { type: "audio/wav" }));
        activeRecording = null;
      }
    }

    // Feed VAD recording
    if (activeVAD) {
      const rms = frameRms(frame);
      activeVAD.onLevel?.(rms);
      const isVoiced = rms > activeVAD.rmsThreshold;

      if (activeVAD.state === "waiting") {
        // Maintain rolling preroll buffer — drop oldest chunks once over budget.
        activeVAD.prerollChunks.push(frame);
        activeVAD.prerollLength += frame.length;
        while (
          activeVAD.prerollLength > activeVAD.prerollSamples &&
          activeVAD.prerollChunks.length > 1
        ) {
          const dropped = activeVAD.prerollChunks.shift()!;
          activeVAD.prerollLength -= dropped.length;
        }

        if (isVoiced) {
          // Onset! Promote preroll into the recording chunk list.
          activeVAD.state = "recording";
          activeVAD.chunks = activeVAD.prerollChunks.slice();
          activeVAD.recordedSamples = activeVAD.prerollLength;
          activeVAD.voicedSamples = frame.length;
          activeVAD.trailingSilenceSamples = 0;
          if (activeVAD.onsetTimer) {
            clearTimeout(activeVAD.onsetTimer);
            activeVAD.onsetTimer = null;
          }
        }
      } else {
        // recording
        activeVAD.chunks.push(frame);
        activeVAD.recordedSamples += frame.length;
        if (isVoiced) {
          activeVAD.voicedSamples += frame.length;
          activeVAD.trailingSilenceSamples = 0;
        } else {
          activeVAD.trailingSilenceSamples += frame.length;
        }

        const trailingSilenceMs = (activeVAD.trailingSilenceSamples / SAMPLE_RATE) * 1000;
        const overCap = activeVAD.recordedSamples >= activeVAD.maxSamples;
        const silenceDone = trailingSilenceMs >= activeVAD.silenceMsTarget;

        if (silenceDone || overCap) {
          const av = activeVAD;
          activeVAD = null;
          if (av.voicedSamples < av.minSpeechSamples) {
            av.resolve({ blob: null, reason: "too-short" });
          } else {
            const flat = concatFloat32(av.chunks, av.recordedSamples);
            const wav = encodeWav(flat, SAMPLE_RATE);
            av.resolve({ blob: new Blob([wav], { type: "audio/wav" }), reason: "ok" });
          }
        }
      }
    }
  }

  return {
    async start() {
      if (audioCtx) return;
      // EC/NS/AGC disabled so this stream doesn't fight Web Speech API's own
      // mic capture in Chrome. Resemblyzer (server-side speaker-ID) prefers
      // raw audio and does its own normalization.
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
      audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
      await audioCtx.audioWorklet.addModule(pcmProcessorUrl);
      source = audioCtx.createMediaStreamSource(stream);
      workletNode = new AudioWorkletNode(audioCtx, "pcm-processor");
      workletNode.port.onmessage = (e: MessageEvent<Float32Array>) => handleFrame(e.data);
      source.connect(workletNode);
      // Note: we do NOT connect workletNode to destination — we don't
      // want to play the mic back through speakers. It processes fine
      // without being routed to output.
    },

    async suspend() {
      if (audioCtx && audioCtx.state === "running") {
        await audioCtx.suspend();
        console.log("[audio-capture] suspended");
      }
    },

    async resume() {
      if (audioCtx && audioCtx.state === "suspended") {
        await audioCtx.resume();
        console.log("[audio-capture] resumed");
      }
    },

    stop() {
      if (workletNode) {
        workletNode.disconnect();
        workletNode.port.onmessage = null;
        workletNode = null;
      }
      if (source) {
        source.disconnect();
        source = null;
      }
      if (stream) {
        for (const track of stream.getTracks()) track.stop();
        stream = null;
      }
      if (audioCtx) {
        audioCtx.close();
        audioCtx = null;
      }
      writePos = 0;
      filled = 0;
      activeRecording = null;
      if (activeVAD?.onsetTimer) clearTimeout(activeVAD.onsetTimer);
      activeVAD = null;
    },

    snapshot(seconds: number): string | null {
      const want = Math.min(Math.floor(seconds * SAMPLE_RATE), filled);
      if (want <= 0) return null;
      // Read `want` samples ending at writePos
      const out = new Float32Array(want);
      const start = (writePos - want + RING_BUFFER_SAMPLES) % RING_BUFFER_SAMPLES;
      if (start + want <= RING_BUFFER_SAMPLES) {
        out.set(ring.subarray(start, start + want));
      } else {
        const firstChunk = RING_BUFFER_SAMPLES - start;
        out.set(ring.subarray(start), 0);
        out.set(ring.subarray(0, want - firstChunk), firstChunk);
      }
      // Float32 → Int16 → base64
      const pcm16 = floatToInt16(out);
      return int16ToBase64(pcm16);
    },

    recordSample(seconds: number): Promise<Blob> {
      if (!audioCtx) return Promise.reject(new Error("audio capture not started"));
      if (activeRecording) return Promise.reject(new Error("already recording"));
      return new Promise((resolve) => {
        activeRecording = {
          chunks: [],
          targetSamples: Math.floor(seconds * SAMPLE_RATE),
          gathered: 0,
          resolve,
        };
      });
    },

    recordSampleVAD(opts: VADOptions = {}): Promise<VADResult> {
      if (!audioCtx) return Promise.reject(new Error("audio capture not started"));
      if (activeRecording || activeVAD) return Promise.reject(new Error("already recording"));

      const prerollSeconds = opts.prerollSeconds ?? 0.3;
      const silenceMs = opts.silenceMs ?? 800;
      const maxSeconds = opts.maxSeconds ?? 5.0;
      const minSpeechMs = opts.minSpeechMs ?? 400;
      const onsetTimeoutMs = opts.onsetTimeoutMs ?? 4000;
      const rmsThreshold = opts.rmsThreshold ?? 0.02;

      return new Promise((resolve) => {
        activeVAD = {
          state: "waiting",
          rmsThreshold,
          silenceMsTarget: silenceMs,
          maxSamples: Math.floor(maxSeconds * SAMPLE_RATE),
          minSpeechSamples: Math.floor((minSpeechMs / 1000) * SAMPLE_RATE),
          prerollSamples: Math.floor(prerollSeconds * SAMPLE_RATE),
          onLevel: opts.onLevel,
          prerollChunks: [],
          prerollLength: 0,
          chunks: [],
          recordedSamples: 0,
          voicedSamples: 0,
          trailingSilenceSamples: 0,
          onsetTimer: null,
          resolve,
        };
        // Onset timeout — fires only if we never leave the `waiting` state.
        activeVAD.onsetTimer = setTimeout(() => {
          if (activeVAD && activeVAD.state === "waiting") {
            const av = activeVAD;
            activeVAD = null;
            av.resolve({ blob: null, reason: "no-speech" });
          }
        }, onsetTimeoutMs);
      });
    },

    isRunning() {
      return audioCtx !== null;
    },
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function frameRms(frame: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
  return Math.sqrt(sum / frame.length);
}

function concatFloat32(chunks: Float32Array[], maxLength: number): Float32Array {
  const out = new Float32Array(maxLength);
  let offset = 0;
  for (const chunk of chunks) {
    const room = maxLength - offset;
    if (room <= 0) break;
    if (chunk.length <= room) {
      out.set(chunk, offset);
      offset += chunk.length;
    } else {
      out.set(chunk.subarray(0, room), offset);
      offset = maxLength;
      break;
    }
  }
  return out;
}

function floatToInt16(floats: Float32Array): Int16Array {
  const out = new Int16Array(floats.length);
  for (let i = 0; i < floats.length; i++) {
    const s = Math.max(-1, Math.min(1, floats[i]));
    out[i] = s < 0 ? Math.floor(s * 32768) : Math.floor(s * 32767);
  }
  return out;
}

function int16ToBase64(pcm: Int16Array): string {
  const bytes = new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength);
  // btoa chokes on large strings, but 2s @ 16kHz @ 2 bytes = 64KB — fine.
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

/** Minimal WAV (RIFF/PCM16) encoder for enrollment uploads. */
function encodeWav(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const pcm = floatToInt16(samples);
  const buffer = new ArrayBuffer(44 + pcm.byteLength);
  const view = new DataView(buffer);
  // RIFF chunk
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + pcm.byteLength, true);
  writeString(view, 8, "WAVE");
  // fmt chunk
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true); // PCM fmt chunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  // data chunk
  writeString(view, 36, "data");
  view.setUint32(40, pcm.byteLength, true);
  // samples
  new Int16Array(buffer, 44).set(pcm);
  return buffer;
}

function writeString(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}
