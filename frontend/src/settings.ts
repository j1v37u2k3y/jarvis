/**
 * JARVIS — Settings Panel
 *
 * Overlay panel for API keys, connection status, preferences, and system info.
 * Slides in from the right with glass-morphism styling.
 */

import type { AudioCapture } from "./audio-capture";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface StatusResponse {
  claude_code_installed: boolean;
  calendar_accessible: boolean;
  mail_accessible: boolean;
  notes_accessible: boolean;
  memory_count: number;
  task_count: number;
  server_port: number;
  uptime_seconds: number;
  env_keys_set: {
    anthropic: boolean;
    fish_audio: boolean;
    fish_voice_id: boolean;
    user_name: string;
  };
}

interface PreferencesResponse {
  user_name: string;
  honorific: string;
  calendar_accounts: string;
}

interface VoiceStatus {
  enrolled: boolean;
  name: string | null;
  sample_count: number;
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let panelEl: HTMLElement | null = null;
let isOpen = false;
let isFirstTimeSetup = false;
let setupStep = 0; // 0=anthropic, 1=fish, 2=name, 3=done

let _audioCapture: AudioCapture | null = null;

/** main.ts calls this once during boot so the enrollment wizard can reach the mic. */
export function setAudioCapture(capture: AudioCapture): void {
  _audioCapture = capture;
}

let _onEnrollmentChange: ((active: boolean) => void) | null = null;

/** main.ts wires this to pause the Web Speech API loop and any in-flight TTS
 *  while the wizard is recording — otherwise the main listener transcribes
 *  the enrollment phrase as a voice command. */
export function setEnrollmentStateHandler(fn: (active: boolean) => void): void {
  _onEnrollmentChange = fn;
}

/** Used by main.ts on boot to gate the voice loop: returns true once a
 *  speaker is enrolled, false otherwise. Reuses the existing auth helper. */
export async function isVoiceEnrolled(): Promise<boolean> {
  try {
    const status = await apiGet<{ enrolled: boolean }>("/api/voice/status");
    return !!status.enrolled;
  } catch {
    // If the endpoint is unreachable, fail open — let the user try to use
    // voice. The server-side speaker-ID gate will still reject if needed.
    return true;
  }
}

/** Fetch the canned "please enroll" TTS line as base64. null on failure. */
export async function fetchEnrollPromptAudio(): Promise<string | null> {
  try {
    const res = await apiGet<{ audio: string | null }>("/api/voice/enroll-prompt");
    return res.audio ?? null;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

let _authToken = "";

async function ensureAuthToken(): Promise<string> {
  if (!_authToken) {
    try {
      const res = await fetch("/auth/token");
      const data = await res.json();
      _authToken = data.token || "";
    } catch {
      console.warn("[settings] failed to fetch auth token");
    }
  }
  return _authToken;
}

async function apiGet<T>(url: string): Promise<T> {
  const token = await ensureAuthToken();
  const res = await fetch(url, {
    headers: token ? { "Authorization": `Bearer ${token}` } : {},
  });
  return res.json();
}

async function apiPost<T>(url: string, body: unknown): Promise<T> {
  const token = await ensureAuthToken();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  return res.json();
}

// ---------------------------------------------------------------------------
// Panel HTML
// ---------------------------------------------------------------------------

function buildPanelHTML(): string {
  return `
    <div class="settings-backdrop" id="settings-backdrop"></div>
    <div class="settings-panel" id="settings-panel-inner">
      <div class="settings-header">
        <h2>Settings</h2>
        <button class="settings-close" id="settings-close">&times;</button>
      </div>

      <div class="settings-welcome" id="settings-welcome" style="display:none">
        <p>Welcome to JARVIS. Let's get you set up.</p>
      </div>

      <div class="settings-body">

        <!-- API Keys -->
        <section class="settings-section" id="section-api-keys">
          <h3>API Keys</h3>

          <div class="settings-field">
            <label>Anthropic API Key</label>
            <div class="settings-input-row">
              <input type="password" id="input-anthropic-key" placeholder="sk-ant-..." />
              <button class="settings-btn" id="btn-test-anthropic">Test</button>
              <span class="status-dot" id="status-anthropic"></span>
            </div>
          </div>

          <div class="settings-field">
            <label>Fish Audio API Key</label>
            <div class="settings-input-row">
              <input type="password" id="input-fish-key" placeholder="Fish Audio key..." />
              <button class="settings-btn" id="btn-test-fish">Test</button>
              <span class="status-dot" id="status-fish"></span>
            </div>
          </div>

          <div class="settings-field">
            <label>Fish Voice ID</label>
            <div class="settings-input-row">
              <input type="text" id="input-fish-voice-id" placeholder="612b878b113047d9a770c069c8b4fdfe" />
              <button class="settings-btn" id="btn-save-voice-id">Save</button>
            </div>
          </div>

          <div class="settings-actions">
            <button class="settings-btn primary" id="btn-save-keys">Save Keys</button>
          </div>
        </section>

        <!-- Connection Status -->
        <section class="settings-section" id="section-status">
          <h3>Connection Status</h3>
          <div class="status-grid">
            <div class="status-row"><span class="status-dot" id="status-claude-cli"></span><span>Claude Code CLI</span></div>
            <div class="status-row"><span class="status-dot" id="status-calendar"></span><span>Apple Calendar</span></div>
            <div class="status-row"><span class="status-dot" id="status-mail"></span><span>Apple Mail</span></div>
            <div class="status-row"><span class="status-dot" id="status-notes"></span><span>Apple Notes</span></div>
            <div class="status-row"><span class="status-dot" id="status-server"></span><span>Server</span><span class="status-detail" id="status-server-detail"></span></div>
          </div>
        </section>

        <!-- User Preferences -->
        <section class="settings-section" id="section-preferences">
          <h3>User Preferences</h3>

          <div class="settings-field">
            <label>Your Name</label>
            <input type="text" id="input-user-name" placeholder="Your name" />
          </div>

          <div class="settings-field">
            <label>Honorific</label>
            <select id="input-honorific">
              <option value="sir">Sir</option>
              <option value="ma'am">Ma'am</option>
              <option value="none">None</option>
            </select>
          </div>

          <div class="settings-field">
            <label>Calendar Accounts</label>
            <textarea id="input-calendar-accounts" rows="2" placeholder="auto (or comma-separated emails)"></textarea>
          </div>

          <div class="settings-actions">
            <button class="settings-btn primary" id="btn-save-prefs">Save Preferences</button>
          </div>
        </section>

        <!-- Voice Recognition -->
        <section class="settings-section" id="section-voice-id">
          <h3>Voice Recognition</h3>
          <p class="voice-id-help" id="voice-id-help">
            JARVIS can recognize your voice and ignore commands from anyone else.
            Record 3 short samples to enroll.
          </p>
          <div class="voice-id-status" id="voice-id-status"></div>
          <div class="voice-id-wizard" id="voice-id-wizard" style="display:none"></div>
          <div class="settings-actions">
            <button class="settings-btn primary" id="btn-voice-enroll">Enroll Your Voice</button>
            <button class="settings-btn" id="btn-voice-clear" style="display:none">Clear &amp; Re-enroll</button>
          </div>
        </section>

        <!-- System Info -->
        <section class="settings-section" id="section-sysinfo">
          <h3>System Info</h3>
          <div class="sysinfo-grid">
            <div class="sysinfo-row"><span class="sysinfo-label">Memory entries</span><span id="sysinfo-memory">--</span></div>
            <div class="sysinfo-row"><span class="sysinfo-label">Tasks</span><span id="sysinfo-tasks">--</span></div>
            <div class="sysinfo-row"><span class="sysinfo-label">Server port</span><span id="sysinfo-port">--</span></div>
            <div class="sysinfo-row"><span class="sysinfo-label">Uptime</span><span id="sysinfo-uptime">--</span></div>
          </div>
        </section>

        <!-- Setup Navigation (first-time only) -->
        <div class="setup-nav" id="setup-nav" style="display:none">
          <button class="settings-btn primary" id="btn-setup-next">Next</button>
        </div>

      </div>
    </div>
  `;
}

// ---------------------------------------------------------------------------
// Panel lifecycle
// ---------------------------------------------------------------------------

function createPanel(): HTMLElement {
  const container = document.createElement("div");
  container.id = "settings-container";
  container.innerHTML = buildPanelHTML();
  document.body.appendChild(container);
  return container;
}

function setDotStatus(id: string, status: "green" | "red" | "yellow" | "off") {
  const dot = document.getElementById(id);
  if (!dot) return;
  dot.className = "status-dot";
  if (status !== "off") dot.classList.add(`status-${status}`);
}

function formatUptime(seconds: number): string {
  if (seconds < 60) return `${Math.floor(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

async function loadStatus() {
  try {
    const status = await apiGet<StatusResponse>("/api/settings/status");

    setDotStatus("status-claude-cli", status.claude_code_installed ? "green" : "red");
    setDotStatus("status-calendar", status.calendar_accessible ? "green" : "red");
    setDotStatus("status-mail", status.mail_accessible ? "green" : "red");
    setDotStatus("status-notes", status.notes_accessible ? "green" : "red");
    setDotStatus("status-server", "green");

    const serverDetail = document.getElementById("status-server-detail");
    if (serverDetail) serverDetail.textContent = `port ${status.server_port} | up ${formatUptime(status.uptime_seconds)}`;

    // API key status dots
    setDotStatus("status-anthropic", status.env_keys_set.anthropic ? "green" : "red");
    setDotStatus("status-fish", status.env_keys_set.fish_audio ? "green" : "red");

    // System info
    const memEl = document.getElementById("sysinfo-memory");
    if (memEl) memEl.textContent = String(status.memory_count);
    const taskEl = document.getElementById("sysinfo-tasks");
    if (taskEl) taskEl.textContent = String(status.task_count);
    const portEl = document.getElementById("sysinfo-port");
    if (portEl) portEl.textContent = String(status.server_port);
    const upEl = document.getElementById("sysinfo-uptime");
    if (upEl) upEl.textContent = formatUptime(status.uptime_seconds);

    return status;
  } catch (e) {
    console.error("[settings] failed to load status:", e);
    setDotStatus("status-server", "red");
    return null;
  }
}

async function loadPreferences() {
  try {
    const prefs = await apiGet<PreferencesResponse>("/api/settings/preferences");
    const nameEl = document.getElementById("input-user-name") as HTMLInputElement;
    const honEl = document.getElementById("input-honorific") as HTMLSelectElement;
    const calEl = document.getElementById("input-calendar-accounts") as HTMLTextAreaElement;
    if (nameEl) nameEl.value = prefs.user_name || "";
    if (honEl) honEl.value = prefs.honorific || "sir";
    if (calEl) calEl.value = prefs.calendar_accounts || "auto";
  } catch (e) {
    console.error("[settings] failed to load preferences:", e);
  }
}

function wireEvents() {
  // Close
  document.getElementById("settings-close")?.addEventListener("click", closeSettings);
  document.getElementById("settings-backdrop")?.addEventListener("click", closeSettings);

  // Save keys
  document.getElementById("btn-save-keys")?.addEventListener("click", async () => {
    const anthropicKey = (document.getElementById("input-anthropic-key") as HTMLInputElement).value.trim();
    const fishKey = (document.getElementById("input-fish-key") as HTMLInputElement).value.trim();

    if (anthropicKey) {
      await apiPost("/api/settings/keys", { key_name: "ANTHROPIC_API_KEY", key_value: anthropicKey });
    }
    if (fishKey) {
      await apiPost("/api/settings/keys", { key_name: "FISH_API_KEY", key_value: fishKey });
    }
    await loadStatus();
  });

  // Save voice ID
  document.getElementById("btn-save-voice-id")?.addEventListener("click", async () => {
    const voiceId = (document.getElementById("input-fish-voice-id") as HTMLInputElement).value.trim();
    if (voiceId) {
      await apiPost("/api/settings/keys", { key_name: "FISH_VOICE_ID", key_value: voiceId });
    }
  });

  // Test Anthropic
  document.getElementById("btn-test-anthropic")?.addEventListener("click", async () => {
    setDotStatus("status-anthropic", "yellow");
    const key = (document.getElementById("input-anthropic-key") as HTMLInputElement).value.trim();
    try {
      const result = await apiPost<{ valid: boolean; error?: string }>("/api/settings/test-anthropic", { key_value: key || undefined });
      setDotStatus("status-anthropic", result.valid ? "green" : "red");
    } catch {
      setDotStatus("status-anthropic", "red");
    }
  });

  // Test Fish
  document.getElementById("btn-test-fish")?.addEventListener("click", async () => {
    setDotStatus("status-fish", "yellow");
    const key = (document.getElementById("input-fish-key") as HTMLInputElement).value.trim();
    try {
      const result = await apiPost<{ valid: boolean; error?: string }>("/api/settings/test-fish", { key_value: key || undefined });
      setDotStatus("status-fish", result.valid ? "green" : "red");
    } catch {
      setDotStatus("status-fish", "red");
    }
  });

  // Save preferences
  document.getElementById("btn-save-prefs")?.addEventListener("click", async () => {
    const user_name = (document.getElementById("input-user-name") as HTMLInputElement).value.trim();
    const honorific = (document.getElementById("input-honorific") as HTMLSelectElement).value;
    const calendar_accounts = (document.getElementById("input-calendar-accounts") as HTMLTextAreaElement).value.trim();
    await apiPost("/api/settings/preferences", { user_name, honorific, calendar_accounts });
    await loadStatus();
  });

  // Voice enrollment
  document.getElementById("btn-voice-enroll")?.addEventListener("click", runVoiceEnrollment);
  document.getElementById("btn-voice-clear")?.addEventListener("click", async () => {
    await clearVoiceProfile();
    await loadVoiceStatus();
  });

  // Setup next button
  document.getElementById("btn-setup-next")?.addEventListener("click", advanceSetup);
}

// ---------------------------------------------------------------------------
// Voice enrollment
// ---------------------------------------------------------------------------

const ENROLLMENT_PROMPTS = [
  "JARVIS, this is me speaking.",
  "Good morning, JARVIS.",
  "Run a status check on all systems.",
];

async function loadVoiceStatus() {
  try {
    const status = await apiGet<VoiceStatus>("/api/voice/status");
    renderVoiceStatus(status);
  } catch (err) {
    console.warn("[voice-id] failed to load status:", err);
  }
}

function renderVoiceStatus(status: VoiceStatus) {
  const statusEl = document.getElementById("voice-id-status");
  const helpEl = document.getElementById("voice-id-help");
  const enrollBtn = document.getElementById("btn-voice-enroll") as HTMLButtonElement | null;
  const clearBtn = document.getElementById("btn-voice-clear") as HTMLButtonElement | null;
  if (!statusEl || !enrollBtn || !clearBtn) return;

  if (status.enrolled) {
    statusEl.innerHTML = `<span class="status-dot status-green"></span> Enrolled as <strong>${escapeHtml(status.name ?? "")}</strong> — ${status.sample_count} sample${status.sample_count === 1 ? "" : "s"}`;
    enrollBtn.textContent = "Add Another Sample";
    clearBtn.style.display = "inline-block";
    if (helpEl) helpEl.style.display = "none";
  } else {
    statusEl.innerHTML = `<span class="status-dot status-off"></span> Not enrolled`;
    enrollBtn.textContent = "Enroll Your Voice";
    clearBtn.style.display = "none";
    if (helpEl) helpEl.style.display = "block";
  }
}

async function runVoiceEnrollment() {
  if (!_audioCapture) {
    alert("Microphone isn't ready. Refresh the page and click anywhere to grant mic access, then try again.");
    return;
  }
  if (!_audioCapture.isRunning()) {
    try {
      await _audioCapture.start();
    } catch (err) {
      alert("Couldn't access the microphone. Grant mic permission and try again.");
      console.error(err);
      return;
    }
  }
  // Defensive: the main app's state machine suspends audio-capture when JARVIS
  // is speaking. Resume here so the wizard isn't recording into a paused ctx.
  await _audioCapture.resume();

  const wizardEl = document.getElementById("voice-id-wizard");
  const enrollBtn = document.getElementById("btn-voice-enroll") as HTMLButtonElement | null;
  const clearBtn = document.getElementById("btn-voice-clear") as HTMLButtonElement | null;
  if (!wizardEl || !enrollBtn) return;

  enrollBtn.disabled = true;
  if (clearBtn) clearBtn.disabled = true;
  wizardEl.style.display = "block";

  // Pause the main voice loop so it doesn't transcribe the enrollment phrase
  // as a command. main.ts handles the resume on the false call.
  _onEnrollmentChange?.(true);

  // Always use the user's configured name if present, else fall back to "me"
  let speakerName = "me";
  try {
    const prefs = await apiGet<PreferencesResponse>("/api/settings/preferences");
    if (prefs.user_name) speakerName = prefs.user_name.trim() || "me";
  } catch {
    // Fall back to "me"
  }

  let finalStatus: VoiceStatus | null = null;
  try {
    for (let i = 0; i < ENROLLMENT_PROMPTS.length; i++) {
      finalStatus = await captureOneSampleVAD(
        wizardEl,
        ENROLLMENT_PROMPTS[i],
        i + 1,
        ENROLLMENT_PROMPTS.length,
        speakerName,
      );
      if (finalStatus === null) break; // error
    }

    if (finalStatus) {
      wizardEl.innerHTML = `<div class="voice-id-step"><p><strong>Enrolled.</strong> I now recognize your voice, sir.</p></div>`;
      await loadVoiceStatus();
      setTimeout(() => {
        if (wizardEl) wizardEl.style.display = "none";
      }, 2500);
    }
  } finally {
    enrollBtn.disabled = false;
    if (clearBtn) clearBtn.disabled = false;
    _onEnrollmentChange?.(false);
  }
}

/** Record one sample using voice-activity detection — waits for speech onset,
 *  ends on trailing silence, shows a live RMS meter. Loops on no-speech /
 *  too-short outcomes via a Retry button. Returns the resulting VoiceStatus
 *  from the server, or null on hard error. */
async function captureOneSampleVAD(
  wizardEl: HTMLElement,
  prompt: string,
  step: number,
  totalSteps: number,
  speakerName: string,
): Promise<VoiceStatus | null> {
  if (!_audioCapture) return null;

  // Retry loop — only the upload path falls through to the return.
  for (;;) {
    wizardEl.innerHTML = `
      <div class="voice-id-step">
        <div class="voice-id-progress">Sample ${step} / ${totalSteps}</div>
        <p>Say: <em>"${escapeHtml(prompt)}"</em></p>
        <div class="voice-id-listening" id="vad-label">Listening for your voice…</div>
        <div class="voice-id-meter"><div class="voice-id-meter-fill" id="vad-meter-fill"></div></div>
      </div>`;

    const meterFill = document.getElementById("vad-meter-fill") as HTMLDivElement | null;
    const labelEl = document.getElementById("vad-label");
    let sawAnySignal = false;
    let switchedToRecording = false;

    const result = await _audioCapture.recordSampleVAD({
      onLevel: (rms) => {
        if (rms > 0.005) sawAnySignal = true;
        if (meterFill) {
          const pct = Math.min(100, (rms / 0.3) * 100);
          meterFill.style.width = `${pct}%`;
        }
        if (rms > 0.02 && !switchedToRecording && labelEl) {
          switchedToRecording = true;
          labelEl.classList.remove("voice-id-listening");
          labelEl.classList.add("voice-id-recording");
          labelEl.textContent = "🎙️ Recording…";
        }
      },
    });

    if (result.reason === "no-speech" && !sawAnySignal) {
      const retry = await showRetry(
        wizardEl,
        `<p><strong>I'm not picking up any audio, sir.</strong></p>
         <p>Check System Settings → Privacy &amp; Security → Microphone, ensure Chrome has permission, and try again.</p>`,
      );
      if (!retry) return null;
      continue;
    }
    if (result.reason === "no-speech") {
      const retry = await showRetry(wizardEl, `<p>I didn't quite hear you, sir.</p>`);
      if (!retry) return null;
      continue;
    }
    if (result.reason === "too-short" || !result.blob) {
      const retry = await showRetry(wizardEl, `<p>Just a touch longer, sir.</p>`);
      if (!retry) return null;
      continue;
    }

    // Uploading
    wizardEl.innerHTML = `
      <div class="voice-id-step">
        <div class="voice-id-progress">Sample ${step} / ${totalSteps}</div>
        <p>Processing…</p>
      </div>`;

    const form = new FormData();
    form.append("name", speakerName);
    form.append("audio", result.blob, `enroll-${step}.wav`);
    const token = await ensureAuthToken();
    try {
      const res = await fetch("/api/voice/enroll", {
        method: "POST",
        headers: token ? { Authorization: `Bearer ${token}` } : {},
        body: form,
      });
      const data = await res.json();
      if (!res.ok || !data.success) {
        wizardEl.innerHTML = `<div class="voice-id-step voice-id-error">Upload failed: ${escapeHtml(data.error ?? "unknown error")}</div>`;
        return null;
      }
      return { enrolled: true, name: speakerName, sample_count: data.sample_count };
    } catch (err) {
      wizardEl.innerHTML = `<div class="voice-id-step voice-id-error">Network error: ${escapeHtml(String(err))}</div>`;
      return null;
    }
  }
}

/** Render an error message + Retry button, resolve true when clicked. The
 *  caller can also bail by closing the panel — that path resolves nothing
 *  and the outer Promise simply hangs until GC, matching prior behavior. */
function showRetry(wizardEl: HTMLElement, innerHtml: string): Promise<boolean> {
  return new Promise((resolve) => {
    wizardEl.innerHTML = `
      <div class="voice-id-step voice-id-error">
        ${innerHtml}
        <button class="settings-btn primary" id="vad-retry">Retry</button>
      </div>`;
    const btn = document.getElementById("vad-retry") as HTMLButtonElement | null;
    if (!btn) {
      resolve(false);
      return;
    }
    btn.addEventListener("click", () => resolve(true), { once: true });
  });
}

async function clearVoiceProfile() {
  const token = await ensureAuthToken();
  await fetch("/api/voice/enroll", {
    method: "DELETE",
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  const wizardEl = document.getElementById("voice-id-wizard");
  if (wizardEl) {
    wizardEl.style.display = "none";
    wizardEl.innerHTML = "";
  }
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}


// ---------------------------------------------------------------------------
// First-time setup wizard
// ---------------------------------------------------------------------------

function enterSetupMode() {
  isFirstTimeSetup = true;
  setupStep = 0;

  const welcome = document.getElementById("settings-welcome");
  if (welcome) welcome.style.display = "block";

  const nav = document.getElementById("setup-nav");
  if (nav) nav.style.display = "flex";

  // Hide sections except API keys
  showSetupStep(0);
}

function showSetupStep(step: number) {
  const sections = ["section-api-keys", "section-status", "section-preferences", "section-sysinfo"];
  sections.forEach((id, i) => {
    const el = document.getElementById(id);
    if (!el) return;
    if (step === 0 && i === 0) el.style.display = "";
    else if (step === 1 && i === 0) el.style.display = "";
    else if (step === 2 && i === 2) el.style.display = "";
    else if (step === 3) el.style.display = "";
    else el.style.display = "none";
  });

  const nextBtn = document.getElementById("btn-setup-next");
  if (nextBtn) {
    if (step === 0) nextBtn.textContent = "Next: Test Keys";
    else if (step === 1) nextBtn.textContent = "Next: Set Your Name";
    else if (step === 2) nextBtn.textContent = "Finish Setup";
    else nextBtn.style.display = "none";
  }
}

async function advanceSetup() {
  setupStep++;
  if (setupStep >= 3) {
    // Done — save everything and close
    isFirstTimeSetup = false;
    const welcome = document.getElementById("settings-welcome");
    if (welcome) welcome.style.display = "none";
    const nav = document.getElementById("setup-nav");
    if (nav) nav.style.display = "none";

    // Show all sections
    ["section-api-keys", "section-status", "section-preferences", "section-sysinfo"].forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.style.display = "";
    });

    closeSettings();
    return;
  }
  showSetupStep(setupStep);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export async function openSettings() {
  if (isOpen) return;
  isOpen = true;

  if (!panelEl) {
    panelEl = createPanel();
    wireEvents();
  }

  panelEl.style.display = "block";

  // Trigger animation
  requestAnimationFrame(() => {
    panelEl!.classList.add("open");
  });

  // Load data
  const status = await loadStatus();
  await loadPreferences();
  await loadVoiceStatus();

  // Check for first-time setup
  if (status && !status.env_keys_set.anthropic) {
    enterSetupMode();
  }
}

export function closeSettings() {
  if (!panelEl || !isOpen) return;
  isOpen = false;
  panelEl.classList.remove("open");
  setTimeout(() => {
    if (panelEl) panelEl.style.display = "none";
  }, 300);
}

export function isSettingsOpen(): boolean {
  return isOpen;
}

/**
 * Check if first-time setup is needed and auto-open.
 */
export async function checkFirstTimeSetup(): Promise<boolean> {
  try {
    const status = await apiGet<StatusResponse>("/api/settings/status");
    if (!status.env_keys_set.anthropic) {
      openSettings();
      return true;
    }
  } catch {
    // Server not ready yet, skip
  }
  return false;
}
