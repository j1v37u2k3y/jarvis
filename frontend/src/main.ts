/**
 * JARVIS — Main entry point.
 *
 * Wires together the orb visualization, WebSocket communication,
 * speech recognition, and audio playback into a single experience.
 */

import { createOrb, type OrbState } from "./orb";
import { createVoiceInput, createAudioPlayer } from "./voice";
import { createAudioCapture } from "./audio-capture";
import { createSocket } from "./ws";
import { openSettings, checkFirstTimeSetup, setAudioCapture } from "./settings";
import "./style.css";

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

type State = "idle" | "listening" | "thinking" | "speaking";
let currentState: State = "idle";
let isMuted = false;

const statusEl = document.getElementById("status-text")!;
const errorEl = document.getElementById("error-text")!;

function showError(msg: string) {
  errorEl.textContent = msg;
  errorEl.style.opacity = "1";
  setTimeout(() => {
    errorEl.style.opacity = "0";
  }, 5000);
}

function updateStatus(state: State) {
  const labels: Record<State, string> = {
    idle: "",
    listening: "listening...",
    thinking: "thinking...",
    speaking: "",
  };
  statusEl.textContent = labels[state];
}

// ---------------------------------------------------------------------------
// Init components
// ---------------------------------------------------------------------------

const canvas = document.getElementById("orb-canvas") as HTMLCanvasElement;
const orb = createOrb(canvas);

const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
const WS_URL = `${wsProto}//${window.location.host}/ws/voice`;
const socket = createSocket(WS_URL);

const audioPlayer = createAudioPlayer();
orb.setAnalyser(audioPlayer.getAnalyser());

// Continuous mic capture (for speaker-ID) — shares the mic with Web
// Speech API but is its own stream. Started lazily on first user
// interaction to respect browser autoplay policy.
const audioCapture = createAudioCapture();
setAudioCapture(audioCapture); // settings panel uses it for enrollment

async function ensureAudioCapture() {
  if (!audioCapture.isRunning()) {
    try {
      await audioCapture.start();
      console.log("[audio-capture] started");
    } catch (err) {
      console.warn("[audio-capture] failed to start:", err);
    }
  }
}

function transition(newState: State) {
  if (newState === currentState) return;
  currentState = newState;
  orb.setState(newState as OrbState);
  updateStatus(newState);

  switch (newState) {
    case "idle":
      if (!isMuted) voiceInput.resume();
      break;
    case "listening":
      if (!isMuted) voiceInput.resume();
      break;
    case "thinking":
      voiceInput.pause();
      break;
    case "speaking":
      voiceInput.pause();
      break;
  }
}

// ---------------------------------------------------------------------------
// Voice input
// ---------------------------------------------------------------------------

const voiceInput = createVoiceInput(
  (text: string) => {
    // Cancel any current JARVIS response before sending new input
    audioPlayer.stop();
    // Snapshot the last 2s of mic audio so the backend can verify the
    // speaker. null if audio-capture hasn't started yet (first utterance
    // before user interaction) — server treats missing audio as soft mode.
    const audioB64 = audioCapture.snapshot(2);
    socket.send({ type: "transcript", text, isFinal: true, audio_b64: audioB64 });
    transition("thinking");
  },
  (msg: string) => {
    showError(msg);
  }
);

// ---------------------------------------------------------------------------
// Audio playback finished
// ---------------------------------------------------------------------------

audioPlayer.onFinished(() => {
  transition("idle");
});

// ---------------------------------------------------------------------------
// WebSocket messages
// ---------------------------------------------------------------------------

socket.onMessage((msg) => {
  const type = msg.type as string;

  if (type === "audio") {
    const audioData = msg.data as string;
    console.log("[audio] received", audioData ? `${audioData.length} chars` : "EMPTY", "state:", currentState);
    if (audioData) {
      if (currentState !== "speaking") {
        transition("speaking");
      }
      audioPlayer.enqueue(audioData);
    } else {
      // TTS failed — no audio but still need to return to idle
      console.warn("[audio] no data received, returning to idle");
      transition("idle");
    }
    // Log text for debugging
    if (msg.text) console.log("[JARVIS]", msg.text);
  } else if (type === "status") {
    const state = msg.state as string;
    if (state === "thinking" && currentState !== "thinking") {
      transition("thinking");
    } else if (state === "working") {
      // Task spawned — show thinking with a different label
      transition("thinking");
      statusEl.textContent = "working...";
    } else if (state === "idle") {
      transition("idle");
    }
  } else if (type === "text") {
    // Text fallback when TTS fails
    console.log("[JARVIS]", msg.text);
  } else if (type === "task_spawned") {
    console.log("[task]", "spawned:", msg.task_id, msg.prompt);
  } else if (type === "task_complete") {
    console.log("[task]", "complete:", msg.task_id, msg.status, msg.summary);
  }
});

// ---------------------------------------------------------------------------
// Kick off
// ---------------------------------------------------------------------------

// Start listening after a brief delay for the orb to render
setTimeout(() => {
  voiceInput.start();
  transition("listening");
}, 1000);

// Resume AudioContext on ANY user interaction (browser autoplay policy)
function ensureAudioContext() {
  const ctx = audioPlayer.getAnalyser().context as AudioContext;
  if (ctx.state === "suspended") {
    ctx.resume().then(() => console.log("[audio] context resumed"));
  }
}
document.addEventListener("click", ensureAudioContext);
document.addEventListener("touchstart", ensureAudioContext);
document.addEventListener("keydown", ensureAudioContext, { once: true });

// Start mic capture on first interaction (browser requires a gesture to
// grant mic permission for AudioContext-based capture).
document.addEventListener("click", ensureAudioCapture, { once: true });
document.addEventListener("keydown", ensureAudioCapture, { once: true });

// Try to resume audio context on load
ensureAudioContext();

// ---------------------------------------------------------------------------
// UI Controls
// ---------------------------------------------------------------------------

const btnMute = document.getElementById("btn-mute")!;
const btnMenu = document.getElementById("btn-menu")!;
const menuDropdown = document.getElementById("menu-dropdown")!;
const btnRestart = document.getElementById("btn-restart")!;
const btnFixSelf = document.getElementById("btn-fix-self")!;
const textInput = document.getElementById("text-input") as HTMLInputElement;
const textSend = document.getElementById("text-send")!;

// Text input — sends through same path as voice, but marked `source: "text"`
// so the backend speaker-ID gate skips it (typed-into-authenticated-browser
// is a different trust surface than "voice within mic range").
function sendTextMessage() {
  const text = textInput.value.trim();
  if (!text) return;
  audioPlayer.stop();
  socket.send({ type: "transcript", text, isFinal: true, source: "text" });
  transition("thinking");
  textInput.value = "";
}

textSend.addEventListener("click", sendTextMessage);
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendTextMessage();
});

btnMute.addEventListener("click", (e) => {
  e.stopPropagation();
  isMuted = !isMuted;
  btnMute.classList.toggle("muted", isMuted);
  if (isMuted) {
    voiceInput.pause();
    transition("idle");
  } else {
    voiceInput.resume();
    transition("listening");
  }
});

btnMenu.addEventListener("click", (e) => {
  e.stopPropagation();
  menuDropdown.style.display = menuDropdown.style.display === "none" ? "block" : "none";
});

document.addEventListener("click", () => {
  menuDropdown.style.display = "none";
});

btnRestart.addEventListener("click", async (e) => {
  e.stopPropagation();
  menuDropdown.style.display = "none";
  statusEl.textContent = "restarting...";
  try {
    const tokenRes = await fetch("/auth/token");
    const { token } = await tokenRes.json();
    await fetch("/api/restart", {
      method: "POST",
      headers: token ? { "Authorization": `Bearer ${token}` } : {},
    });
    // Wait a few seconds then reload
    setTimeout(() => window.location.reload(), 4000);
  } catch {
    statusEl.textContent = "restart failed";
  }
});

btnFixSelf.addEventListener("click", (e) => {
  e.stopPropagation();
  menuDropdown.style.display = "none";
  // Activate work mode on the WebSocket session (JARVIS becomes Claude Code's voice)
  socket.send({ type: "fix_self" });
  statusEl.textContent = "entering work mode...";
});

// Settings button
const btnSettings = document.getElementById("btn-settings")!;
btnSettings.addEventListener("click", (e) => {
  e.stopPropagation();
  menuDropdown.style.display = "none";
  openSettings();
});

// First-time setup detection — check after a short delay for server readiness
setTimeout(() => {
  checkFirstTimeSetup();
}, 2000);
