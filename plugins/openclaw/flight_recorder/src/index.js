import fs from "node:fs";
import path from "node:path";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

const SCHEMA_VERSION = "hfr.openclaw.event.v1";
const DEFAULT_OUTPUT_DIR = ".hfr-openclaw";
const DEFAULT_MAX_FIELD_CHARS = 12000;
const MAX_COLLECTION_ITEMS = 200;
const SECRET_KEY_RE =
  /(?:api[_-]?key|secret|password|authorization|bearer|accessToken|refreshToken|idToken|authToken|(?:^|[_-])token(?:$|[_-]))/i;

const HOOKS = [
  "session_start",
  "session_end",
  "before_model_resolve",
  "agent_turn_prepare",
  "before_prompt_build",
  "before_agent_run",
  "before_agent_reply",
  "before_agent_finalize",
  "agent_end",
  "model_call_started",
  "model_call_ended",
  "llm_input",
  "llm_output",
  "before_tool_call",
  "after_tool_call",
  "subagent_spawned",
  "subagent_ended",
];

export { HOOKS, SCHEMA_VERSION };

export default definePluginEntry({
  id: "flight-recorder",
  name: "Flight Recorder",
  register(api) {
    for (const hook of HOOKS) {
      api.on(
        hook,
        async (event) => {
          debug(`hook ${hook}`);
          writeEvent(hook, event);
          return undefined;
        },
        { priority: -100, timeoutMs: 1000 },
      );
    }
  },
});

function writeEvent(hook, event) {
  try {
    const config = pluginConfig(event);
    const outputDir = process.env.OPENCLAW_FLIGHT_RECORDER_OUTPUT_DIR || config.outputDir || DEFAULT_OUTPUT_DIR;
    const fieldLimit = resolveMaxFieldChars(config);
    fs.mkdirSync(outputDir, { recursive: true });
    const row = {
      schema_version: SCHEMA_VERSION,
      hook,
      captured_at: new Date().toISOString(),
      payload: bound(event, fieldLimit),
    };
    const fileName = `${safeSegment(sessionKey(event))}.openclaw.jsonl`;
    fs.appendFileSync(path.join(outputDir, fileName), `${JSON.stringify(row)}\n`, "utf8");
  } catch (error) {
    debug(`write failed for ${hook}: ${error?.stack || error}`);
    return undefined;
  }
  return undefined;
}

function debug(message) {
  if (process.env.OPENCLAW_FLIGHT_RECORDER_DEBUG === "1") {
    console.warn(`[flight-recorder] ${message}`);
  }
}

function pluginConfig(event) {
  const config = event?.context?.pluginConfig;
  return config && typeof config === "object" ? config : {};
}

function resolveMaxFieldChars(config) {
  const raw = process.env.OPENCLAW_FLIGHT_RECORDER_MAX_FIELD_CHARS || config.maxFieldChars || DEFAULT_MAX_FIELD_CHARS;
  const parsed = Number.parseInt(String(raw), 10);
  return Number.isFinite(parsed) ? Math.max(100, parsed) : DEFAULT_MAX_FIELD_CHARS;
}

function sessionKey(event) {
  const context = event?.context && typeof event.context === "object" ? event.context : {};
  return (
    context.sessionKey ||
    context.sessionId ||
    context.runId ||
    event?.sessionKey ||
    event?.sessionId ||
    event?.runId ||
    "session"
  );
}

function safeSegment(value) {
  return String(value)
    .replace(/[^A-Za-z0-9_.-]+/g, "_")
    .replace(/^[._-]+|[._-]+$/g, "")
    .slice(0, 140) || "session";
}

function bound(value, maxChars, seen = new WeakSet(), key = "") {
  if (SECRET_KEY_RE.test(key)) {
    return "[REDACTED]";
  }
  if (value === null || value === undefined) {
    return value;
  }
  if (typeof value === "string") {
    return value.length <= maxChars ? value : `${value.slice(0, maxChars)}...[truncated]`;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  if (typeof value === "bigint") {
    return value.toString();
  }
  if (typeof value === "function") {
    return `[function ${value.name || "anonymous"}]`;
  }
  if (typeof value !== "object") {
    return String(value);
  }
  if (seen.has(value)) {
    return "[Circular]";
  }
  seen.add(value);
  if (Array.isArray(value)) {
    return value.slice(0, MAX_COLLECTION_ITEMS).map((item, index) => bound(item, maxChars, seen, String(index)));
  }
  const output = {};
  for (const [entryKey, entryValue] of Object.entries(value).slice(0, MAX_COLLECTION_ITEMS)) {
    output[entryKey] = bound(entryValue, maxChars, seen, entryKey);
  }
  return output;
}
