import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

const SCHEMA_VERSION = "hfr.openclaw.event.v1";
const DEFAULT_OUTPUT_DIR = ".hfr-openclaw";
const DEFAULT_MAX_FIELD_CHARS = 12000;
const MAX_COLLECTION_ITEMS = 200;
const MAX_DEPTH = 32;
const MAX_PAYLOAD_BYTES = 1024 * 1024;
const BUDGET_RESERVE_BYTES = 256;
const REDACTED_VALUE = "[REDACTED]";
const CIRCULAR_SENTINEL = "[Circular]";
const MAX_DEPTH_SENTINEL = "[Truncated: max depth]";
const COLLECTION_SENTINEL = "[Truncated: collection items]";
const AGGREGATE_SENTINEL = "[Truncated: aggregate limit]";
const SAFE_SECRET_KEY_SUFFIXES = [
  "_budget",
  "_available",
  "_checked",
  "_count",
  "_counts",
  "_env",
  "_environment",
  "_limit",
  "_limits",
  "_name",
  "_names",
  "_present",
  "_recorded",
  "_required",
  "_status",
  "_type",
  "_usage",
];
const SECRET_KEY_MARKERS = new Set(["authorization", "bearer", "password", "secret", "token"]);
const SECRET_KEY_TERMS = new Set(["auth", "authentication", "credential", "credentials", "cookie"]);
const SECRET_KEY_QUALIFIERS = new Set(["access", "api", "client", "private", "secret", "signing"]);

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

export { HOOKS, SCHEMA_VERSION, bound, redactText, writeEvent };

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
      hook: redactedFieldText(hook, fieldLimit),
      captured_at: new Date().toISOString(),
      payload: bound(event, fieldLimit),
    };
    const fileName = sessionFileName(sessionKey(event));
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

function sessionFileName(value) {
  const raw = String(value);
  const stem = safeSegment(redactText(raw));
  const digest = createHash("sha256").update(raw, "utf8").digest("hex");
  return `${stem}-${digest}.openclaw.jsonl`;
}

function bound(
  value,
  maxChars,
  state = createBoundState(),
  key = "",
  depth = 0,
  commandSequence = false,
  pairContainer = false,
) {
  if (state.exhausted) {
    return AGGREGATE_SENTINEL;
  }
  if (isSecretKey(key)) {
    return boundText(REDACTED_VALUE, maxChars, state, false);
  }
  if (value === null || value === undefined) {
    if (!consumeBudget(state, 4)) {
      return AGGREGATE_SENTINEL;
    }
    return value;
  }
  if (typeof value === "string") {
    return boundText(value, maxChars, state);
  }
  if (typeof value === "number" || typeof value === "boolean") {
    if (!consumeBudget(state, serializedBytes(value))) {
      return AGGREGATE_SENTINEL;
    }
    return value;
  }
  if (typeof value === "bigint") {
    return boundText(value.toString(), maxChars, state);
  }
  if (typeof value === "function") {
    return boundText(`[function ${value.name || "anonymous"}]`, maxChars, state);
  }
  if (typeof value !== "object") {
    return boundText(String(value), maxChars, state);
  }
  if (depth >= MAX_DEPTH) {
    return boundText(MAX_DEPTH_SENTINEL, maxChars, state, false);
  }
  if (state.seen.has(value)) {
    return boundText(CIRCULAR_SENTINEL, maxChars, state, false);
  }
  if (!consumeBudget(state, 2)) {
    return AGGREGATE_SENTINEL;
  }
  state.seen.add(value);
  if (Array.isArray(value)) {
    try {
      const output = [];
      const count = Math.min(value.length, MAX_COLLECTION_ITEMS);
      const sequenceContext = commandSequence || isCommandSequenceKey(key);
      const pairContext = pairContainer || isPairContainerKey(key);
      const redactPairValue = pairContext && isSecretPairArray(value);
      let redactNext = false;
      for (let index = 0; index < count; index += 1) {
        if (output.length > 0 && !consumeBudget(state, 1)) {
          output.push(AGGREGATE_SENTINEL);
          break;
        }
        const item = value[index];
        if (redactPairValue && index === 1) {
          output.push(boundText(REDACTED_VALUE, maxChars, state, false));
          if (state.exhausted) {
            break;
          }
          continue;
        }
        if (sequenceContext && redactNext) {
          if (isCliFlagToken(item)) {
            redactNext = false;
          } else if (isSequenceScalar(item)) {
            output.push(boundText(REDACTED_VALUE, maxChars, state, false));
            redactNext = false;
            if (state.exhausted) {
              break;
            }
            continue;
          } else {
            redactNext = false;
          }
        }
        output.push(
          bound(
            item,
            maxChars,
            state,
            String(index),
            depth + 1,
            sequenceContext && Array.isArray(item),
            pairContext,
          ),
        );
        if (state.exhausted) {
          break;
        }
        if (sequenceContext) {
          redactNext = isSecretCliFlagToken(item);
        }
      }
      if (!state.exhausted && value.length > MAX_COLLECTION_ITEMS) {
        if (output.length > 0) {
          consumeBudget(state, 1);
        }
        output.push(boundText(COLLECTION_SENTINEL, maxChars, state, false));
      }
      return output;
    } finally {
      state.seen.delete(value);
    }
  }
  try {
    const output = {};
    const entries = Object.entries(value);
    const count = Math.min(entries.length, MAX_COLLECTION_ITEMS);
    const pairContext = pairContainer || isPairContainerKey(key);
    const redactPairValue = pairContext && isSecretPairObject(value);
    for (let index = 0; index < count; index += 1) {
      const [entryKey, entryValue] = entries[index];
      const safeKey = redactedFieldText(entryKey, maxChars);
      const separatorBytes = Object.keys(output).length > 0 ? 1 : 0;
      if (!consumeBudget(state, serializedBytes(safeKey) + separatorBytes + 1)) {
        appendObjectSentinel(output, AGGREGATE_SENTINEL);
        break;
      }
      output[safeKey] = redactPairValue && normalizeKey(entryKey) === "value"
        ? boundText(REDACTED_VALUE, maxChars, state, false)
        : bound(
          entryValue,
          maxChars,
          state,
          entryKey,
          depth + 1,
          isCommandSequenceKey(entryKey),
          pairContext || isPairContainerKey(entryKey),
        );
      if (state.exhausted) {
        break;
      }
    }
    if (!state.exhausted && entries.length > MAX_COLLECTION_ITEMS) {
      appendObjectSentinel(output, COLLECTION_SENTINEL);
    }
    return output;
  } finally {
    state.seen.delete(value);
  }
}

function createBoundState() {
  return {
    exhausted: false,
    remaining: MAX_PAYLOAD_BYTES - BUDGET_RESERVE_BYTES,
    seen: new WeakSet(),
  };
}

function consumeBudget(state, size) {
  if (size <= state.remaining) {
    state.remaining -= size;
    return true;
  }
  state.remaining = 0;
  state.exhausted = true;
  return false;
}

function boundText(value, maxChars, state, redact = true) {
  let text = redact ? redactText(value) : String(value);
  if (text.length > maxChars) {
    text = `${text.slice(0, maxChars)}...[truncated]`;
  }
  const size = serializedBytes(text);
  if (size <= state.remaining) {
    consumeBudget(state, size);
    return text;
  }

  const available = state.remaining;
  state.remaining = 0;
  state.exhausted = true;
  let low = 0;
  let high = text.length;
  while (low < high) {
    const midpoint = Math.floor((low + high + 1) / 2);
    if (serializedBytes(text.slice(0, midpoint)) <= available) {
      low = midpoint;
    } else {
      high = midpoint - 1;
    }
  }
  return `${text.slice(0, low)}${AGGREGATE_SENTINEL}`;
}

function redactedFieldText(value, maxChars) {
  const text = redactText(value);
  return text.length <= maxChars ? text : `${text.slice(0, maxChars)}...[truncated]`;
}

function serializedBytes(value) {
  const serialized = JSON.stringify(value);
  return Buffer.byteLength(serialized === undefined ? "null" : serialized, "utf8");
}

function appendObjectSentinel(output, sentinel) {
  let key = "__truncated__";
  while (Object.hasOwn(output, key)) {
    key = `_${key}`;
  }
  output[key] = sentinel;
}

function isSecretKey(key) {
  const normalized = normalizeKey(key);
  if (!normalized || SAFE_SECRET_KEY_SUFFIXES.some((suffix) => normalized.endsWith(suffix))) {
    return false;
  }
  const parts = normalized.split("_");
  const hasApiKey = normalized.replaceAll("_", "").includes("apikey");
  return (
    hasApiKey ||
    parts.some((part) => SECRET_KEY_MARKERS.has(part)) ||
    parts.some((part) => SECRET_KEY_TERMS.has(part)) ||
    (parts.includes("key") && parts.some((part) => SECRET_KEY_QUALIFIERS.has(part)))
  );
}

function normalizeKey(key) {
  return String(key)
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1_$2")
    .replace(/([a-z0-9])([A-Z])/g, "$1_$2")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function isCommandSequenceKey(key) {
  return ["args", "arguments", "argv", "command", "command_argv"].includes(normalizeKey(key));
}

function isPairContainerKey(key) {
  return [
    "header",
    "headers",
    "parameters",
    "params",
    "query",
    "query_parameters",
    "query_params",
  ].includes(normalizeKey(key));
}

function isSecretPairArray(value) {
  return value.length === 2 && typeof value[0] === "string" && isSecretKey(value[0]);
}

function isSecretPairObject(value) {
  return Object.entries(value).some(
    ([entryKey, entryValue]) =>
      ["key", "name"].includes(normalizeKey(entryKey)) &&
      typeof entryValue === "string" &&
      isSecretKey(entryValue),
  );
}

function isCliFlagToken(value) {
  if (typeof value !== "string") {
    return false;
  }
  if (value === "-" || value === "--") {
    return true;
  }
  if (value.startsWith("--") && value.length > 2) {
    return /[A-Za-z0-9_]/.test(value[2]);
  }
  return value.startsWith("-") && value.length > 1 && /[A-Za-z_]/.test(value[1]);
}

function isSecretCliFlagToken(value) {
  if (typeof value !== "string" || !value.startsWith("--") || value.length <= 2) {
    return false;
  }
  const key = value.slice(2);
  return !/[=:]/.test(key) && /^[A-Za-z0-9_.-]+$/.test(key) && isSecretKey(key);
}

function isSequenceScalar(value) {
  return value === null || typeof value !== "object";
}

function redactText(value) {
  const text = redactCliArguments(value === null || value === undefined ? "" : String(value));
  const assignmentPrefix = /(^|[^\w.:-])(["']?)([\w.-]+)\2[^\S\r\n]*([:=])[^\S\r\n]*/g;
  const parts = [];
  let cursor = 0;
  let match;
  while ((match = assignmentPrefix.exec(text)) !== null) {
    const valueStart = match.index + match[0].length;
    if (!isSecretKey(match[3])) {
      assignmentPrefix.lastIndex = Math.max(valueStart, match.index + 1);
      continue;
    }
    const valueEnd = assignmentValueEnd(text, valueStart);
    const rawValue = text.slice(valueStart, valueEnd);
    if (!isUnredactedSecretValue(rawValue)) {
      assignmentPrefix.lastIndex = Math.max(valueEnd, valueStart + 1);
      continue;
    }
    parts.push(text.slice(cursor, valueStart), redactedAssignmentValue(rawValue));
    cursor = valueEnd;
    assignmentPrefix.lastIndex = Math.max(valueEnd, valueStart + 1);
  }
  if (parts.length === 0) {
    return text;
  }
  parts.push(text.slice(cursor));
  return parts.join("");
}

function redactCliArguments(text) {
  const flagPattern = /(?<!\S)--([A-Za-z0-9][A-Za-z0-9_.-]*)/g;
  const parts = [];
  let cursor = 0;
  let match;
  while ((match = flagPattern.exec(text)) !== null) {
    const flagEnd = match.index + match[0].length;
    if (!isSecretKey(match[1]) || flagEnd >= text.length || !" \t".includes(text[flagEnd])) {
      flagPattern.lastIndex = Math.max(flagEnd, match.index + 1);
      continue;
    }
    let valueStart = flagEnd;
    while (valueStart < text.length && " \t".includes(text[valueStart])) {
      valueStart += 1;
    }
    if (valueStart >= text.length || startsFollowingFlag(text, valueStart)) {
      flagPattern.lastIndex = Math.max(valueStart, flagEnd);
      continue;
    }
    const valueEnd = cliValueEnd(text, valueStart);
    const rawValue = text.slice(valueStart, valueEnd);
    if (!isUnredactedSecretValue(rawValue)) {
      flagPattern.lastIndex = Math.max(valueEnd, valueStart + 1);
      continue;
    }
    parts.push(text.slice(cursor, valueStart), redactedAssignmentValue(rawValue));
    cursor = valueEnd;
    flagPattern.lastIndex = Math.max(valueEnd, valueStart + 1);
  }
  if (parts.length === 0) {
    return text;
  }
  parts.push(text.slice(cursor));
  return parts.join("");
}

function assignmentValueEnd(text, valueStart) {
  if (valueStart >= text.length) {
    return valueStart;
  }
  if (text[valueStart] === '"' || text[valueStart] === "'") {
    return quotedValueEnd(text, valueStart);
  }
  let cursor = valueStart;
  while (cursor < text.length) {
    const character = text[cursor];
    if (",;}\r\n".includes(character)) {
      break;
    }
    if (character === " " || character === "\t") {
      let nextField = cursor;
      while (nextField < text.length && (text[nextField] === " " || text[nextField] === "\t")) {
        nextField += 1;
      }
      if (startsAdjacentAssignment(text, nextField) || startsCliFlag(text, nextField)) {
        break;
      }
    }
    cursor += 1;
  }
  while (cursor > valueStart && (text[cursor - 1] === " " || text[cursor - 1] === "\t")) {
    cursor -= 1;
  }
  return cursor;
}

function startsAdjacentAssignment(text, start) {
  const match = /^(["']?)([\w.-]+)\1[^\S\r\n]*([:=])[^\S\r\n]*/.exec(text.slice(start));
  return Boolean(match && start + match[0].length < text.length && !"=,;}\r\n".includes(text[start + match[0].length]));
}

function startsCliFlag(text, start) {
  return /^--[A-Za-z0-9][A-Za-z0-9_.-]*/.test(text.slice(start));
}

function startsFollowingFlag(text, start) {
  return /^(?:--[A-Za-z0-9_][A-Za-z0-9_.-]*|-[A-Za-z_][A-Za-z0-9_.-]*|--?)(?=$|[\s=;&|])/.test(
    text.slice(start),
  );
}

function cliValueEnd(text, valueStart) {
  if (text[valueStart] === '"' || text[valueStart] === "'") {
    return quotedValueEnd(text, valueStart);
  }
  let cursor = valueStart;
  while (cursor < text.length) {
    const character = text[cursor];
    if (character === "\\" && cursor + 1 < text.length) {
      cursor += 2;
      continue;
    }
    if (" \t\r\n;&|\"'".includes(character)) {
      break;
    }
    cursor += 1;
  }
  return cursor;
}

function quotedValueEnd(text, valueStart) {
  const quote = text[valueStart];
  let cursor = valueStart + 1;
  let escaped = false;
  while (cursor < text.length) {
    const character = text[cursor];
    if (character === "\r" || character === "\n") {
      return cursor;
    }
    if (character === quote && !escaped) {
      return cursor + 1;
    }
    if (character === "\\" && !escaped) {
      escaped = true;
    } else {
      escaped = false;
    }
    cursor += 1;
  }
  return cursor;
}

function isUnredactedSecretValue(value) {
  const unquoted = value.startsWith('"') || value.startsWith("'")
    ? value.slice(1, value.length > 1 && value.at(-1) === value[0] ? -1 : undefined)
    : value;
  return unquoted.trim().length > 0 && !/^(?:\[REDACTED\]|<redacted:[^>\r\n]+>)$/i.test(unquoted.trim());
}

function redactedAssignmentValue(rawValue) {
  if (rawValue.startsWith('"') || rawValue.startsWith("'")) {
    const quote = rawValue[0];
    const closingQuote = rawValue.length > 1 && rawValue.endsWith(quote) ? quote : "";
    return `${quote}${REDACTED_VALUE}${closingQuote}`;
  }
  return REDACTED_VALUE;
}
