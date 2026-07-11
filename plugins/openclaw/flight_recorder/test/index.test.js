import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const pluginRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

async function loadPluginModule() {
  const source = fs
    .readFileSync(path.join(pluginRoot, "src", "index.js"), "utf8")
    .replace(
      'import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";',
      "const definePluginEntry = (entry) => entry;",
    );
  return import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
}

test("observer redacts structural and embedded command secrets before disk", async () => {
  const outputDir = fs.mkdtempSync(path.join(os.tmpdir(), "hfr-openclaw-redaction-"));
  const previous = process.env.OPENCLAW_FLIGHT_RECORDER_OUTPUT_DIR;
  process.env.OPENCLAW_FLIGHT_RECORDER_OUTPUT_DIR = outputDir;
  const structuralSecret = "openclaw-structural-private-value";
  const commandSecret = "openclaw-command-private-value";
  const flagSecret = "openclaw-flag-private-value";
  const quotedFlagSecret = "openclaw-quoted-flag-private-value";
  try {
    const { writeEvent } = await loadPluginModule();
    writeEvent("before_tool_call", {
      sessionId: "session-privacy",
      apiKey: structuralSecret,
      args: {
        command: (
          `curl https://example.invalid api_key=${commandSecret} safe=visible ` +
          `--api-key --token ${flagSecret} --client-secret "${quotedFlagSecret}" ` +
          "--token-budget 10 --api-key --verbose"
        ),
      },
    });

    const files = fs.readdirSync(outputDir);
    assert.equal(files.length, 1);
    const text = fs.readFileSync(path.join(outputDir, files[0]), "utf8");
    const row = JSON.parse(text);
    assert.doesNotMatch(text, new RegExp(structuralSecret));
    assert.doesNotMatch(text, new RegExp(commandSecret));
    assert.doesNotMatch(text, new RegExp(flagSecret));
    assert.doesNotMatch(text, new RegExp(quotedFlagSecret));
    assert.equal(row.payload.apiKey, "[REDACTED]");
    assert.match(row.payload.args.command, /api_key=\[REDACTED\]/);
    assert.match(row.payload.args.command, /safe=visible/);
    assert.match(row.payload.args.command, /--api-key --token \[REDACTED\]/);
    assert.match(row.payload.args.command, /--client-secret "\[REDACTED\]"/);
    assert.match(row.payload.args.command, /--token-budget 10 --api-key --verbose/);
  } finally {
    if (previous === undefined) {
      delete process.env.OPENCLAW_FLIGHT_RECORDER_OUTPUT_DIR;
    } else {
      process.env.OPENCLAW_FLIGHT_RECORDER_OUTPUT_DIR = previous;
    }
    fs.rmSync(outputDir, { recursive: true, force: true });
  }
});

test("observer redacts tokenized command sequences without changing ordinary arrays", async () => {
  const outputDir = fs.mkdtempSync(path.join(os.tmpdir(), "hfr-openclaw-argv-redaction-"));
  const previous = process.env.OPENCLAW_FLIGHT_RECORDER_OUTPUT_DIR;
  process.env.OPENCLAW_FLIGHT_RECORDER_OUTPUT_DIR = outputDir;
  const argvSecret = "openclaw-argv-private-value";
  const nestedSecret = "openclaw-nested-argv-private-value";
  const commandArgvSecret = "openclaw-command-argv-private-value";
  const argsSecret = "openclaw-args-private-value";
  const argumentsSecret = "openclaw-arguments-private-value";
  try {
    const { writeEvent } = await loadPluginModule();
    writeEvent("before_tool_call", {
      sessionId: "session-tokenized-privacy",
      argv: [
        "curl",
        "--api-key",
        argvSecret,
        "--token-budget",
        "10",
        "--password",
        12345,
        "--client-secret",
        "--verbose",
        "visible",
      ],
      nested: {
        command: ["runner", "--api-key", "--token", nestedSecret],
      },
      aliases: {
        commandArgv: ["runner", "--access-key", commandArgvSecret],
        args: ["runner", "--token", argsSecret],
        arguments: ["runner", "--password", argumentsSecret],
      },
      ordinaryValues: ["--api-key", "ordinary-array-value"],
    });

    const [fileName] = fs.readdirSync(outputDir);
    const text = fs.readFileSync(path.join(outputDir, fileName), "utf8");
    const row = JSON.parse(text);
    assert.doesNotMatch(text, new RegExp(argvSecret));
    assert.doesNotMatch(text, new RegExp(nestedSecret));
    assert.doesNotMatch(text, new RegExp(commandArgvSecret));
    assert.doesNotMatch(text, new RegExp(argsSecret));
    assert.doesNotMatch(text, new RegExp(argumentsSecret));
    assert.deepEqual(row.payload.argv, [
      "curl",
      "--api-key",
      "[REDACTED]",
      "--token-budget",
      "10",
      "--password",
      "[REDACTED]",
      "--client-secret",
      "--verbose",
      "visible",
    ]);
    assert.deepEqual(row.payload.nested.command, [
      "runner",
      "--api-key",
      "--token",
      "[REDACTED]",
    ]);
    assert.deepEqual(row.payload.aliases, {
      commandArgv: ["runner", "--access-key", "[REDACTED]"],
      args: ["runner", "--token", "[REDACTED]"],
      arguments: ["runner", "--password", "[REDACTED]"],
    });
    assert.deepEqual(row.payload.ordinaryValues, ["--api-key", "ordinary-array-value"]);
  } finally {
    if (previous === undefined) {
      delete process.env.OPENCLAW_FLIGHT_RECORDER_OUTPUT_DIR;
    } else {
      process.env.OPENCLAW_FLIGHT_RECORDER_OUTPUT_DIR = previous;
    }
    fs.rmSync(outputDir, { recursive: true, force: true });
  }
});

test("observer redacts structural header and parameter pairs", async () => {
  const outputDir = fs.mkdtempSync(path.join(os.tmpdir(), "hfr-openclaw-pair-redaction-"));
  const previous = process.env.OPENCLAW_FLIGHT_RECORDER_OUTPUT_DIR;
  process.env.OPENCLAW_FLIGHT_RECORDER_OUTPUT_DIR = outputDir;
  const headerSecret = "openclaw-header-private-value";
  const namedHeaderSecret = "openclaw-named-header-private-value";
  const parameterSecret = "openclaw-parameter-private-value";
  const querySecret = "openclaw-query-private-value";
  try {
    const { writeEvent } = await loadPluginModule();
    writeEvent("before_model_resolve", {
      sessionId: "session-pair-privacy",
      headers: [
        ["Authorization", `Bearer ${headerSecret}`],
        { name: "Authorization", value: `Bearer ${namedHeaderSecret}` },
        ["Content-Type", "application/json"],
        { name: "token_budget", value: "10" },
      ],
      params: [
        ["api_key", parameterSecret],
        ["token_budget", "10"],
        ["page", "1"],
      ],
      queryParameters: [
        { key: "access_token", value: querySecret },
        { key: "token_count", value: "2" },
      ],
      ordinaryPairs: [["Authorization", "ordinary-bearer-value"]],
    });

    const [fileName] = fs.readdirSync(outputDir);
    const text = fs.readFileSync(path.join(outputDir, fileName), "utf8");
    const row = JSON.parse(text);
    for (const secret of [headerSecret, namedHeaderSecret, parameterSecret, querySecret]) {
      assert.doesNotMatch(text, new RegExp(secret));
    }
    assert.deepEqual(row.payload.headers, [
      ["Authorization", "[REDACTED]"],
      { name: "Authorization", value: "[REDACTED]" },
      ["Content-Type", "application/json"],
      { name: "token_budget", value: "10" },
    ]);
    assert.deepEqual(row.payload.params, [
      ["api_key", "[REDACTED]"],
      ["token_budget", "10"],
      ["page", "1"],
    ]);
    assert.deepEqual(row.payload.queryParameters, [
      { key: "access_token", value: "[REDACTED]" },
      { key: "token_count", value: "2" },
    ]);
    assert.deepEqual(row.payload.ordinaryPairs, [
      ["Authorization", "ordinary-bearer-value"],
    ]);
  } finally {
    if (previous === undefined) {
      delete process.env.OPENCLAW_FLIGHT_RECORDER_OUTPUT_DIR;
    } else {
      process.env.OPENCLAW_FLIGHT_RECORDER_OUTPUT_DIR = previous;
    }
    fs.rmSync(outputDir, { recursive: true, force: true });
  }
});

test("observer session filenames redact stems without collapsing distinct raw IDs", async () => {
  const outputDir = fs.mkdtempSync(path.join(os.tmpdir(), "hfr-openclaw-session-name-"));
  const previous = process.env.OPENCLAW_FLIGHT_RECORDER_OUTPUT_DIR;
  process.env.OPENCLAW_FLIGHT_RECORDER_OUTPUT_DIR = outputDir;
  const firstSecret = "openclaw-first-session-private-value";
  const secondSecret = "openclaw-second-session-private-value";
  const firstSession = `token=${firstSecret}`;
  const secondSession = `token=${secondSecret}`;
  try {
    const { writeEvent } = await loadPluginModule();
    writeEvent("session_start", { sessionId: firstSession });
    writeEvent("session_start", { sessionId: secondSession });

    const files = fs.readdirSync(outputDir).sort();
    assert.equal(files.length, 2);
    assert.notEqual(files[0], files[1]);
    const firstDigest = createHash("sha256").update(firstSession, "utf8").digest("hex");
    const secondDigest = createHash("sha256").update(secondSession, "utf8").digest("hex");
    assert.ok(files.some((fileName) => fileName.endsWith(`-${firstDigest}.openclaw.jsonl`)));
    assert.ok(files.some((fileName) => fileName.endsWith(`-${secondDigest}.openclaw.jsonl`)));
    const persisted = files
      .map((fileName) => `${fileName}\n${fs.readFileSync(path.join(outputDir, fileName), "utf8")}`)
      .join("\n");
    assert.doesNotMatch(persisted, new RegExp(firstSecret));
    assert.doesNotMatch(persisted, new RegExp(secondSecret));
  } finally {
    if (previous === undefined) {
      delete process.env.OPENCLAW_FLIGHT_RECORDER_OUTPUT_DIR;
    } else {
      process.env.OPENCLAW_FLIGHT_RECORDER_OUTPUT_DIR = previous;
    }
    fs.rmSync(outputDir, { recursive: true, force: true });
  }
});

test("observer preserves a bounded partial payload for adversarial collections", async () => {
  const outputDir = fs.mkdtempSync(path.join(os.tmpdir(), "hfr-openclaw-bounds-"));
  const previous = process.env.OPENCLAW_FLIGHT_RECORDER_OUTPUT_DIR;
  process.env.OPENCLAW_FLIGHT_RECORDER_OUTPUT_DIR = outputDir;
  try {
    const { writeEvent } = await loadPluginModule();
    const event = { sessionId: "session-bounds" };
    event.cycle = event;
    const deep = {};
    let cursor = deep;
    for (let index = 0; index < 64; index += 1) {
      cursor.child = {};
      cursor = cursor.child;
    }
    event.deep = deep;
    event.items = Array.from({ length: 500 }, (_, index) => index);
    event.aggregate = Array.from({ length: 100 }, () => "x".repeat(12000));

    writeEvent("llm_output", event);

    const [fileName] = fs.readdirSync(outputDir);
    const filePath = path.join(outputDir, fileName);
    const row = JSON.parse(fs.readFileSync(filePath, "utf8"));
    assert.equal(row.payload.cycle, "[Circular]");
    assert.match(JSON.stringify(row.payload.deep), /max depth/i);
    assert.ok(row.payload.items.length <= 201);
    assert.match(String(row.payload.items.at(-1)), /collection/i);
    assert.ok(row.payload.aggregate.length < 100);
    assert.match(String(row.payload.aggregate.at(-1)), /aggregate/i);
    assert.ok(Buffer.byteLength(JSON.stringify(row.payload), "utf8") <= 1024 * 1024);
    assert.ok(fs.statSync(filePath).size < 1_100_000);
  } finally {
    if (previous === undefined) {
      delete process.env.OPENCLAW_FLIGHT_RECORDER_OUTPUT_DIR;
    } else {
      process.env.OPENCLAW_FLIGHT_RECORDER_OUTPUT_DIR = previous;
    }
    fs.rmSync(outputDir, { recursive: true, force: true });
  }
});
