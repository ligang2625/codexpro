import { spawn } from "node:child_process";

export interface ClaudeCodeTarget {
  name: string;
  tmuxTarget: string;
}

export interface ClaudeCodeSendResult {
  ok: true;
  target: string;
  tmuxTarget: string;
  submitted: boolean;
  bytes: number;
}

export interface ClaudeCodeSkillResult extends ClaudeCodeSendResult {
  skill: string;
  command: string;
}

export interface ClaudeCodeCaptureResult {
  ok: true;
  target: string;
  tmuxTarget: string;
  lines: number;
  output: string;
  bytes: number;
  totalBytes: number;
  truncated: boolean;
}

export type ClaudeCodeStatus =
  | "empty"
  | "waiting_permission"
  | "waiting_input"
  | "likely_running"
  | "likely_done"
  | "unknown";

export type ClaudeCodeStatusConfidence = "low" | "medium" | "high";

export interface ClaudeCodeStatusSignal {
  kind: string;
  description: string;
  evidence: string;
}

export interface ClaudeCodeInspectResult extends ClaudeCodeCaptureResult {
  status: ClaudeCodeStatus;
  confidence: ClaudeCodeStatusConfidence;
  signals: ClaudeCodeStatusSignal[];
  heuristic: true;
  analyzedAt: string;
}

const TARGET_NAME_RE = /^[A-Za-z0-9_.-]{1,80}$/;
const TMUX_TARGET_RE = /^[A-Za-z0-9_.:@/%+-]{1,160}$/;
const SKILL_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/;

function parsePositiveIntEnv(name: string, fallback: number, min: number, max: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;

  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed)) return fallback;

  return Math.max(min, Math.min(parsed, max));
}

function maxTextBytes(): number {
  return parsePositiveIntEnv("CODEXPRO_CLAUDE_MAX_TEXT_BYTES", 20_000, 1_000, 200_000);
}

function maxCaptureLines(): number {
  return parsePositiveIntEnv("CODEXPRO_CLAUDE_CAPTURE_MAX_LINES", 500, 1, 5_000);
}

function maxCaptureBytes(): number {
  return parsePositiveIntEnv("CODEXPRO_CLAUDE_CAPTURE_MAX_BYTES", 120_000, 1_000, 1_000_000);
}

function normalizeCaptureLines(lines: number | undefined): number {
  const fallback = 80;
  const limit = maxCaptureLines();

  if (lines === undefined) {
    return Math.min(fallback, limit);
  }

  if (!Number.isFinite(lines)) {
    return Math.min(fallback, limit);
  }

  return Math.max(1, Math.min(Math.floor(lines), limit));
}

function truncateUtf8(
  text: string,
  maxBytes: number
): { text: string; bytes: number; totalBytes: number; truncated: boolean } {
  const totalBytes = Buffer.byteLength(text, "utf8");

  if (totalBytes <= maxBytes) {
    return {
      text,
      bytes: totalBytes,
      totalBytes,
      truncated: false
    };
  }

  const marker = "\n...[capture truncated]";
  const markerBytes = Buffer.byteLength(marker, "utf8");
  const bodyLimit = Math.max(0, maxBytes - markerBytes);

  let used = 0;
  let out = "";

  for (const char of text) {
    const charBytes = Buffer.byteLength(char, "utf8");
    if (used + charBytes > bodyLimit) break;

    out += char;
    used += charBytes;
  }

  const truncatedText = `${out}${marker}`;

  return {
    text: truncatedText,
    bytes: Buffer.byteLength(truncatedText, "utf8"),
    totalBytes,
    truncated: true
  };
}

function signalEvidence(value: string): string {
  return value.replace(/\s+/g, " ").trim().slice(0, 240);
}

function recentNonEmptyLines(output: string, maxLines = 80): string[] {
  return output
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((line) => line.trimEnd())
    .filter((line) => line.trim().length > 0)
    .slice(-maxLines);
}

function findMatchingLine(lines: string[], patterns: RegExp[]): string | undefined {
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const line = lines[i] ?? "";

    if (patterns.some((pattern) => pattern.test(line))) {
      return line;
    }
  }

  return undefined;
}

function analyzeClaudeCodeOutput(output: string): {
  status: ClaudeCodeStatus;
  confidence: ClaudeCodeStatusConfidence;
  signals: ClaudeCodeStatusSignal[];
} {
  const trimmed = output.trim();
  const signals: ClaudeCodeStatusSignal[] = [];

  if (!trimmed) {
    return {
      status: "empty",
      confidence: "high",
      signals: [
        {
          kind: "empty_output",
          description: "No visible Claude Code output was captured from the tmux pane.",
          evidence: ""
        }
      ]
    };
  }

  const lines = recentNonEmptyLines(trimmed, 100);
  const recentLines = lines.slice(-30);
  const tail = recentLines.join("\n");

  const permissionEvidence = findMatchingLine(recentLines, [
    /\bdo you want to proceed\b/i,
    /\bdo you want to continue\b/i,
    /\ballow\b.*\?/i,
    /\bapprove\b.*\?/i,
    /\bpermission\b.*\?/i,
    /\bpermission required\b/i,
    /\bneeds permission\b/i,
    /\bcan i\b.*\?/i,
    /\bwould you like to\b.*\?/i,
    /\byes\b.*\bno\b/i,
    /\by\/n\b/i,
    /\bY\/n\b/,
    /\bN\/y\b/,
    /❯.*\b(yes|allow|approve|proceed)\b/i
  ]);

  if (permissionEvidence) {
    signals.push({
      kind: "permission_prompt",
      description:
        "Recent output looks like Claude Code is asking the user to approve or allow an action.",
      evidence: signalEvidence(permissionEvidence)
    });

    return {
      status: "waiting_permission",
      confidence: "high",
      signals
    };
  }

  const inputEvidence = findMatchingLine(recentLines, [
    /\bpress enter\b/i,
    /\bwaiting for input\b/i,
    /\benter your\b/i,
    /\btype your\b/i,
    /\bplease respond\b/i,
    /\bcontinue\?\b/i,
    /^\s*(?:>|❯|\?)\s*$/,
    /^\s*(?:Human|User):\s*$/i
  ]);

  if (inputEvidence) {
    signals.push({
      kind: "input_prompt",
      description: "Recent output looks like Claude Code is waiting for user input.",
      evidence: signalEvidence(inputEvidence)
    });

    return {
      status: "waiting_input",
      confidence: "medium",
      signals
    };
  }

  const runningEvidence = findMatchingLine(recentLines, [
    /\besc to interrupt\b/i,
    /\bctrl\+c\b/i,
    /\bctrl-c\b/i,
    /\bthinking\b/i,
    /\brunning\b/i,
    /\bexecuting\b/i,
    /\bsearching\b/i,
    /\breading\b/i,
    /\bwriting\b/i,
    /\banalyzing\b/i,
    /\binstalling\b/i,
    /\bbuilding\b/i,
    /\bcompiling\b/i,
    /…\s*$/
  ]);

  if (runningEvidence) {
    signals.push({
      kind: "activity_marker",
      description: "Recent output contains activity markers that may indicate Claude Code is still working.",
      evidence: signalEvidence(runningEvidence)
    });

    return {
      status: "likely_running",
      confidence: "medium",
      signals
    };
  }

  const doneEvidence = findMatchingLine(recentLines, [
    /\bdone\b/i,
    /\bcomplete\b/i,
    /\bcompleted\b/i,
    /\bfinished\b/i,
    /\ball set\b/i,
    /\bready for review\b/i,
    /\btests? passed\b/i,
    /\bbuild succeeded\b/i,
    /\bno issues found\b/i,
    /\bsuccessfully\b/i
  ]);

  if (doneEvidence) {
    signals.push({
      kind: "completion_marker",
      description:
        "Recent output contains completion-like language. This is only a heuristic and may be wrong.",
      evidence: signalEvidence(doneEvidence)
    });

    return {
      status: "likely_done",
      confidence: "medium",
      signals
    };
  }

  const errorLikeEvidence = findMatchingLine(recentLines, [
    /\berror\b/i,
    /\bfailed\b/i,
    /\bfatal\b/i,
    /\bexception\b/i,
    /\btraceback\b/i
  ]);

  if (errorLikeEvidence) {
    signals.push({
      kind: "error_like_output",
      description:
        "Recent output contains error-like language. Claude Code may have stopped or may need follow-up.",
      evidence: signalEvidence(errorLikeEvidence)
    });

    return {
      status: "likely_done",
      confidence: "low",
      signals
    };
  }

  signals.push({
    kind: "recent_output",
    description:
      "Output was captured, but no strong running, completion, permission, or input signal was detected.",
    evidence: signalEvidence(tail.slice(-500))
  });

  return {
    status: "unknown",
    confidence: "low",
    signals
  };
}

function assertSafeTargetName(name: string): void {
  if (!TARGET_NAME_RE.test(name)) {
    throw new Error(
      `Invalid Claude Code target name: ${name}. ` +
        "Use only letters, numbers, dots, underscores, and hyphens."
    );
  }
}

function assertSafeTmuxTarget(target: string): void {
  if (!TMUX_TARGET_RE.test(target)) {
    throw new Error(
      `Invalid tmux target for Claude Code bridge: ${target}. ` +
        "Refusing whitespace, shell metacharacters, or control characters."
    );
  }
}

function normalizeTargetSpec(spec: string): ClaudeCodeTarget | undefined {
  const trimmed = spec.trim();
  if (!trimmed) return undefined;

  const eq = trimmed.indexOf("=");

  const name = eq >= 0 ? trimmed.slice(0, eq).trim() : trimmed;
  const tmuxTarget = eq >= 0 ? trimmed.slice(eq + 1).trim() : trimmed;

  if (!name || !tmuxTarget) {
    throw new Error(
      `Invalid CODEXPRO_CLAUDE_TARGETS entry: ${spec}. ` +
        "Use either 'name' or 'name=tmux-target'."
    );
  }

  assertSafeTargetName(name);
  assertSafeTmuxTarget(tmuxTarget);

  return { name, tmuxTarget };
}

export function listClaudeCodeTargets(): ClaudeCodeTarget[] {
  const raw = process.env.CODEXPRO_CLAUDE_TARGETS ?? "";

  const targets = raw
    .split(",")
    .map(normalizeTargetSpec)
    .filter((target): target is ClaudeCodeTarget => Boolean(target));

  const seen = new Set<string>();
  const out: ClaudeCodeTarget[] = [];

  for (const target of targets) {
    if (seen.has(target.name)) continue;
    seen.add(target.name);
    out.push(target);
  }

  return out;
}

function findTarget(name: string): ClaudeCodeTarget {
  const targetName = name.trim();
  assertSafeTargetName(targetName);

  const targets = listClaudeCodeTargets();
  const target = targets.find((item) => item.name === targetName);

  if (!target) {
    const available = targets.map((item) => item.name).join(", ") || "none";
    throw new Error(
      `Claude Code target is not allowed: ${targetName}. ` +
        `Allowed targets from CODEXPRO_CLAUDE_TARGETS: ${available}`
    );
  }

  return target;
}

async function runTmux(args: string[], stdin?: string): Promise<string> {
  return await new Promise<string>((resolve, reject) => {
    const child = spawn("tmux", args, {
      stdio: ["pipe", "pipe", "pipe"]
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString("utf8");
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString("utf8");
    });

    child.on("error", (error) => {
      reject(
        new Error(
          `Failed to run tmux. Is tmux installed and is CodexPro running inside a user session? ${error.message}`
        )
      );
    });

    child.on("close", (code) => {
      if (code === 0) {
        resolve(stdout);
        return;
      }

      reject(
        new Error(
          `tmux ${args.join(" ")} failed with exit code ${code}. ${stderr.trim()}`
        )
      );
    });

    if (stdin !== undefined) {
      child.stdin.write(stdin);
    }

    child.stdin.end();
  });
}

async function assertTmuxTargetExists(target: ClaudeCodeTarget): Promise<void> {
  await runTmux([
    "display-message",
    "-p",
    "-t",
    target.tmuxTarget,
    "#{pane_id}"
  ]);
}

function validateText(text: string): number {
  if (!text) {
    throw new Error("Text is required.");
  }

  if (text.includes("\0")) {
    throw new Error("Text contains a NUL byte, refusing to send.");
  }

  const bytes = Buffer.byteLength(text, "utf8");
  const limit = maxTextBytes();

  if (bytes > limit) {
    throw new Error(
      `Text is too large for Claude Code bridge: ${bytes} bytes. Limit: ${limit} bytes.`
    );
  }

  return bytes;
}

export async function sendToClaudeCode(input: {
  target: string;
  text: string;
  submit?: boolean;
}): Promise<ClaudeCodeSendResult> {
  const target = findTarget(input.target);
  const text = input.text;
  const submitted = input.submit === true;
  const bytes = validateText(text);

  await assertTmuxTargetExists(target);

  // Use tmux buffer + stdin to avoid shell interpolation and argument-length issues.
  await runTmux(["load-buffer", "-"], text);
  await runTmux(["paste-buffer", "-t", target.tmuxTarget]);

  if (submitted) {
    await runTmux(["send-keys", "-t", target.tmuxTarget, "Enter"]);
  }

  return {
    ok: true,
    target: target.name,
    tmuxTarget: target.tmuxTarget,
    submitted,
    bytes
  };
}

export async function captureClaudeCodeOutput(input: {
  target: string;
  lines?: number;
}): Promise<ClaudeCodeCaptureResult> {
  const target = findTarget(input.target);
  const lines = normalizeCaptureLines(input.lines);

  await assertTmuxTargetExists(target);

  const rawOutput = await runTmux([
    "capture-pane",
    "-p",
    "-J",
    "-S",
    `-${lines}`,
    "-t",
    target.tmuxTarget
  ]);

  const output = rawOutput.replace(/\r\n/g, "\n").trimEnd();
  const bounded = truncateUtf8(output, maxCaptureBytes());

  return {
    ok: true,
    target: target.name,
    tmuxTarget: target.tmuxTarget,
    lines,
    output: bounded.text,
    bytes: bounded.bytes,
    totalBytes: bounded.totalBytes,
    truncated: bounded.truncated
  };
}

export async function inspectClaudeCodeStatus(input: {
  target: string;
  lines?: number;
}): Promise<ClaudeCodeInspectResult> {
  const captured = await captureClaudeCodeOutput(input);
  const analysis = analyzeClaudeCodeOutput(captured.output);

  return {
    ...captured,
    status: analysis.status,
    confidence: analysis.confidence,
    signals: analysis.signals,
    heuristic: true,
    analyzedAt: new Date().toISOString()
  };
}

export function buildClaudeCodeSkillCommand(skill: string, args?: string): string {
  const cleanSkill = skill.trim();

  if (!SKILL_NAME_RE.test(cleanSkill)) {
    throw new Error(
      `Invalid Claude Code skill name: ${skill}. ` +
        "Use only letters, numbers, underscores, and hyphens. Do not include a leading slash."
    );
  }

  const cleanArgs = args?.trim();

  return cleanArgs ? `/${cleanSkill} ${cleanArgs}` : `/${cleanSkill}`;
}

export async function invokeClaudeCodeSkill(input: {
  target: string;
  skill: string;
  arguments?: string;
  submit?: boolean;
}): Promise<ClaudeCodeSkillResult> {
  const command = buildClaudeCodeSkillCommand(input.skill, input.arguments);
  const result = await sendToClaudeCode({
    target: input.target,
    text: command,
    submit: input.submit
  });

  return {
    ...result,
    skill: input.skill.trim(),
    command
  };
}
