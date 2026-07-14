import fsp from "node:fs/promises";
import path from "node:path";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import type { CodexProConfig } from "./config.js";
import { WorkspaceManager, PathGuard, CodexProError, type Workspace } from "./guard.js";
import { repoTree, readTextFile, writeTextFile, editTextFile, ensureAiBridge } from "./fsOps.js";
import { searchWorkspace } from "./searchOps.js";
import { runBash } from "./bashOps.js";
import { gitDiff, gitLog, gitStatus } from "./gitOps.js";
import { readAiBridgeContext, readCodexContext, workspaceSummary } from "./workspaceOps.js";
import { buildProContext, exportProContext } from "./proContext.js";
import { codexproInventory, loadSkill } from "./capabilitiesOps.js";
import { listCodexSessions, readCodexSession } from "./codexSessions.js";
import { TOOL_CARD_MIME_TYPE, TOOL_CARD_URI, toolCardWidgetHtml } from "./toolCardWidget.js";
import { redactSensitiveText, redactStructured } from "./redact.js";
import {
  captureClaudeCodeOutput,
  inspectClaudeCodeStatus,
  invokeClaudeCodeSkill,
  listClaudeCodeTargets,
  sendToClaudeCode
} from "./claudeCodeBridge.js";
import { createReviewExecutionTask } from "./reviewExecutionTask.js";

function errorText(error: unknown): string {
  if (error instanceof Error) return redactSensitiveText(`${error.name}: ${error.message}`);
  return redactSensitiveText(String(error));
}

function textResult(text: string, structuredContent: Record<string, unknown> = {}, meta: Record<string, unknown> = {}): any {
  return {
    content: [{ type: "text", text: redactSensitiveText(text) }],
    structuredContent: redactStructured(structuredContent),
    _meta: meta
  };
}

function countTextLines(value: string | undefined): number {
  if (!value) return 0;
  return value.split(/\r?\n/).filter((line) => line.length > 0).length;
}

function bashTextResult(config: CodexProConfig, result: Awaited<ReturnType<typeof runBash>>): string {
  if (config.bashTranscript === "full") {
    return `# Bash\n\n\`\`\`bash\n$ ${result.command}\n\`\`\`\n\nCWD: ${result.cwd}\nExit: ${result.exitCode}${result.signal ? ` (${result.signal})` : ""}\nDuration: ${result.durationMs} ms\n\n## stdout\n\n\`\`\`text\n${result.stdout || ""}\n\`\`\`\n\n## stderr\n\n\`\`\`text\n${result.stderr || ""}\n\`\`\``;
  }

  const stdoutLines = countTextLines(result.stdout);
  const stderrLines = countTextLines(result.stderr);
  return [
    "# Bash",
    "",
    `\`${result.command}\``,
    "",
    `CWD: ${result.cwd}`,
    `Exit: ${result.exitCode}${result.signal ? ` (${result.signal})` : ""}`,
    `Duration: ${result.durationMs} ms`,
    `Output: stdout ${stdoutLines} line${stdoutLines === 1 ? "" : "s"}, stderr ${stderrLines} line${stderrLines === 1 ? "" : "s"}.`,
    "",
    "Raw stdout/stderr are in the structured CodexPro card. Start with `--bash-transcript full` to print raw output in chat."
  ].join("\n");
}

function errorResult(error: unknown): any {
  return {
    isError: true,
    content: [{ type: "text", text: errorText(error) }],
    structuredContent: { error: errorText(error) }
  };
}

function tagToolResult(result: any, name: string, options: Record<string, unknown>): any {
  if (!result || typeof result !== "object") return result;
  const structured = result.structuredContent;
  const base =
    structured && typeof structured === "object" && !Array.isArray(structured)
      ? structured
      : {};
  result.structuredContent = {
    codexpro_tool: name,
    codexpro_title: options.title ?? name,
    ...base
  };
  return result;
}

function toolCardMeta(): Record<string, unknown> {
  return {
    ui: { resourceUri: TOOL_CARD_URI },
    "openai/outputTemplate": TOOL_CARD_URI
  };
}

function toolCallLoggingEnabled(): boolean {
  return process.env.CODEXPRO_LOG_TOOL_CALLS === "1" || process.env.CODEXPRO_LOG_REQUESTS === "1";
}

function logToolCall(name: string, status: "ok" | "error", started: number): void {
  if (!toolCallLoggingEnabled()) return;
  console.error(`[CodexProTool] ${name} ${status} ${Date.now() - started}ms`);
}

function registerToolCardResource(server: McpServer, config: CodexProConfig): void {
  const s = server as any;
  if (typeof s.registerResource !== "function") return;
  s.registerResource(
    "codexpro-tool-card",
    TOOL_CARD_URI,
    {
      title: "CodexPro Tool Card",
      description: "Compact visual renderer for CodexPro workspace orientation, source changes, and handoffs.",
      mimeType: TOOL_CARD_MIME_TYPE
    },
    async () => ({
      contents: [
        {
          uri: TOOL_CARD_URI,
          mimeType: TOOL_CARD_MIME_TYPE,
          text: toolCardWidgetHtml,
          _meta: {
            ui: {
              prefersBorder: true,
              domain: config.widgetDomain,
              csp: {
                connectDomains: [],
                resourceDomains: []
              }
            },
            "openai/widgetDescription": "Renders CodexPro workspace orientation, diagnostics, file diffs, change reviews, terminal checks, Pro context exports, and handoff plans as compact developer cards with bounded previews.",
            "openai/widgetPrefersBorder": true,
            "openai/widgetDomain": config.widgetDomain,
            "openai/widgetCSP": {
              connect_domains: [],
              resource_domains: []
            }
          }
        }
      ]
    })
  );
}


function isContextPath(config: CodexProConfig, relPath: string): boolean {
  const normalized = relPath.split(path.sep).join("/").replace(/^\.\//, "");
  const contextDir = config.contextDir.replace(/^\.\//, "").replace(/\/$/, "");
  return normalized === contextDir || normalized.startsWith(`${contextDir}/`);
}

function assertWriteToolAllowed(config: CodexProConfig, relPath: string): void {
  if (config.writeMode === "workspace") return;
  if (config.writeMode === "handoff" && isContextPath(config, relPath)) return;
  if (config.writeMode === "handoff") {
    throw new CodexProError(
      `Source writes are disabled because CODEXPRO_WRITE_MODE=handoff. ` +
        `Use handoff_to_agent or handoff_to_codex, or write/edit only inside ${config.contextDir}/.`
    );
  }
  throw new CodexProError("write/edit tools are disabled because CODEXPRO_WRITE_MODE=off. handoff_to_agent and handoff_to_codex are still available for planning.");
}

function registerToolCompat(
  server: McpServer,
  name: string,
  options: Record<string, unknown>,
  handler: (args: any) => Promise<any> | any
): void {
  const wrapped = async (args: any) => {
    const started = Date.now();
    try {
      const result = tagToolResult(await handler(args ?? {}), name, options);
      logToolCall(name, result?.isError ? "error" : "ok", started);
      return result;
    } catch (error) {
      const result = tagToolResult(errorResult(error), name, options);
      logToolCall(name, "error", started);
      return result;
    }
  };

  const securitySchemes = [{ type: "noauth" }];
  const fullOptions: Record<string, unknown> = {
    securitySchemes,
    ...options,
    _meta: {
      securitySchemes,
      ...(options._meta as Record<string, unknown> | undefined)
    }
  };

  const s = server as any;
  if (typeof s.registerTool === "function") {
    s.registerTool(name, fullOptions, wrapped);
    return;
  }

  if (typeof s.tool === "function") {
    s.tool(name, (fullOptions.description as string | undefined) ?? name, fullOptions.inputSchema ?? {}, wrapped);
    return;
  }

  throw new Error("Unsupported MCP SDK: McpServer has neither registerTool nor tool.");
}

const MINIMAL_TOOL_NAMES = [
  "server_config",
  "codexpro_self_test",
  "open_current_workspace",
  "open_workspace",
  "read",
  "write",
  "edit",
  "bash",
  "show_changes"
] as const;

const STANDARD_TOOL_NAMES = [
  ...MINIMAL_TOOL_NAMES,
  "tree",
  "search",
  "load_skill",
  "codex_context",
  "read_handoff",
  "export_pro_context",
  "handoff_to_agent",
  "create_review_execution_task",
  "list_claude_code_targets",
  "send_to_claude_code",
  "invoke_claude_code_skill",
  "capture_claude_output",
  "inspect_claude_code_status"
] as const;

const FULL_TOOL_NAMES = [
  "server_config",
  "codexpro_self_test",
  "codexpro_inventory",
  "load_skill",
  "list_workspaces",
  "open_current_workspace",
  "open_workspace",
  "workspace_snapshot",
  "tree",
  "search",
  "read",
  "write",
  "edit",
  "bash",
  "git_status",
  "git_diff",
  "show_changes",
  "read_handoff",
  "codex_context",
  "export_pro_context",
  "handoff_to_agent",
  "create_review_execution_task",
  "handoff_to_codex",
  "list_claude_code_targets",
  "send_to_claude_code",
  "invoke_claude_code_skill",
  "capture_claude_output",
  "inspect_claude_code_status",
  "sync_claude_code_status"
] as const;

function codexSessionToolNames(config: CodexProConfig): string[] {
  if (config.codexSessions === "off") return [];
  return config.codexSessions === "read"
    ? ["codex_sessions", "read_codex_session"]
    : ["codex_sessions"];
}

function toolNamesForMode(config: CodexProConfig): string[] {
  const names: string[] =
    config.toolMode === "full"
      ? [...FULL_TOOL_NAMES]
      : config.toolMode === "minimal"
        ? [...MINIMAL_TOOL_NAMES]
        : [...STANDARD_TOOL_NAMES];
  for (const name of codexSessionToolNames(config)) {
    if (!names.includes(name)) names.push(name);
  }
  return names;
}

const MINIMAL_TOOLS = new Set<string>(MINIMAL_TOOL_NAMES);
const STANDARD_TOOLS = new Set<string>(STANDARD_TOOL_NAMES);

function shouldRegisterTool(config: CodexProConfig, name: string): boolean {
  if (name === "codex_sessions") return config.codexSessions !== "off";
  if (name === "read_codex_session") return config.codexSessions === "read";
  if (config.toolMode === "full") return true;
  if (config.toolMode === "minimal") return MINIMAL_TOOLS.has(name);
  return STANDARD_TOOLS.has(name);
}

function registerCodexTool(
  config: CodexProConfig,
  server: McpServer,
  name: string,
  options: Record<string, unknown>,
  handler: (args: any) => Promise<any> | any
): void {
  if (!shouldRegisterTool(config, name)) return;
  registerToolCompat(server, name, options, handler);
}

function serverInstructions(config: CodexProConfig): string {
  return [
    "CodexPro connects ChatGPT to one local development workspace.",
    "",
    "Preferred workflow:",
    "1. Start with open_current_workspace. Use open_workspace only when the user gives a different root or asks to switch folders.",
    "2. If .claude/WEBGPT.md is present, call codex_context and follow its WebGPT instructions before editing files. Do not read or rely on CLAUDE.md unless the user explicitly asks for it.",
    "3. Inspect with tree, search, and read. Do not use bash for git status, git diff, cat, sed, grep, rg, find, ls, or file reading.",
    "4. Edit with write/edit. After edits, call show_changes once for git status, diff stats, and review diff.",
    "5. Use bash only for meaningful verification commands such as npm test, npm run build, lint, typecheck, or an existing project script.",
    "6. Keep tool calls minimal. Prefer one targeted search plus show_changes instead of repeated broad bash/git calls.",
    "7. Skill rule: do not auto-select skills. Only call load_skill when the user explicitly names a skill, including user skills under ~/.claude/skills, or when .claude/WEBGPT.md names a specific required skill. Treat SKILL.md and references as guidance, not as permission to execute scripts or read secrets.",
    "8. Claude Code bridge rule: only send text to Claude Code when the user explicitly asks. Use list_claude_code_targets first if the target is unclear. Default to submit=false unless the user clearly asks to send/execute/submit. Use capture_claude_output when the user asks for raw recent Claude Code output. Use inspect_claude_code_status when the user asks whether Claude Code is done, running, waiting for input, waiting for permission, or when continuing from its visible response. Use sync_claude_code_status when the user asks to save, persist, synchronize, or share Claude Code status with the workspace. capture_claude_output and inspect_claude_code_status are read-only; sync_claude_code_status reads recent tmux pane text and writes bounded status snapshots under the workspace. inspect_claude_code_status is heuristic and must not be treated as full Claude state. Do not answer Claude Code permission prompts, do not auto-confirm permissions, do not run tmux through bash, and do not use the bridge as a shell.",
    "9. Review handoff rule: when WebGPT has reviewed code and the user wants local Claude Code to implement changes, prefer create_review_execution_task first. It writes a structured .ai-bridge review task and returns a prompt for Claude Code, but it does not send, submit, execute, wait, or confirm permissions.",
    config.codexSessions !== "off"
      ? `7. Codex session history access is enabled in ${config.codexSessions} mode. Use it only when the user asks for local Codex session history.`
      : "",
    config.requireBashSession && config.bashSessionId
      ? `8. Bash session guard is enabled. Every bash call must include session_id="${config.bashSessionId}".`
      : config.bashSessionId
        ? `8. Bash session label for this server is "${config.bashSessionId}".`
        : "",
    "",
    `Current modes: tool=${config.toolMode}, bash=${config.bashMode}, write=${config.writeMode}.`
  ].filter(Boolean).join("\n");
}

function limitInt(value: unknown, fallback: number, min: number, max: number): number {
  const n = Number(value ?? fallback);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, Math.floor(n)));
}

function parseBool(value: unknown, fallback = false): boolean {
  if (typeof value === "boolean") return value;
  if (value === undefined || value === null) return fallback;
  return ["1", "true", "yes", "y"].includes(String(value).toLowerCase());
}

function diffBlock(diff: string): string {
  return `\n\n\`\`\`diff\n${diff}\n\`\`\``;
}

function codeFenceFor(value: string): string {
  let fence = "```";

  while (value.includes(fence)) {
    fence += "`";
  }

  return fence;
}

function normalizeWorkspaceWritePath(config: CodexProConfig, value: unknown, fallbackName: string): string {
  const fallbackPath = `${config.contextDir.replace(/\/$/, "")}/${fallbackName}`;
  const raw = typeof value === "string" && value.trim() ? value.trim() : fallbackPath;

  if (path.isAbsolute(raw) || raw.startsWith("~") || /^[A-Za-z]:[\\/]/.test(raw)) {
    throw new CodexProError(`Status sync path must be workspace-relative: ${raw}`);
  }

  const normalized = raw.split(path.sep).join("/").replace(/^\.\//, "");

  if (
    normalized === "" ||
    normalized === "." ||
    normalized === ".." ||
    normalized.startsWith("../") ||
    normalized.includes("/../")
  ) {
    throw new CodexProError(`Status sync path escapes workspace root: ${raw}`);
  }

  return normalized;
}

function statusSyncStructured(result: Awaited<ReturnType<typeof inspectClaudeCodeStatus>>): Record<string, unknown> {
  return {
    target: result.target,
    tmux_target: result.tmuxTarget,
    status: result.status,
    confidence: result.confidence,
    heuristic: result.heuristic,
    analyzed_at: result.analyzedAt,
    lines: result.lines,
    bytes: result.bytes,
    total_bytes: result.totalBytes,
    truncated: result.truncated,
    signals: result.signals,
    output: result.output
  };
}

function buildClaudeStatusMarkdown(result: Awaited<ReturnType<typeof inspectClaudeCodeStatus>>): string {
  const output = result.output || "(no output captured)";
  const fence = codeFenceFor(output);

  const signals = result.signals.length
    ? result.signals
        .map((signal) =>
          [
            `- ${signal.kind}: ${signal.description}`,
            signal.evidence ? `  Evidence: ${signal.evidence}` : ""
          ]
            .filter(Boolean)
            .join("\n")
        )
        .join("\n")
    : "- No signals detected.";

  return [
    "# Claude Code Status",
    "",
    `Target: ${result.target}`,
    `tmux target: ${result.tmuxTarget}`,
    `Status: ${result.status}`,
    `Confidence: ${result.confidence}`,
    `Heuristic: ${result.heuristic ? "yes" : "no"}`,
    `Analyzed at: ${result.analyzedAt}`,
    `Lines inspected: ${result.lines}`,
    `Bytes returned: ${result.bytes}`,
    `Total bytes captured before truncation: ${result.totalBytes}`,
    `Truncated: ${result.truncated ? "yes" : "no"}`,
    "",
    "This file is a workspace status snapshot generated from recent tmux pane text.",
    "It is not Claude Code internal state and may be incomplete or stale.",
    "",
    "## Signals",
    "",
    signals,
    "",
    "## Recent Output",
    "",
    `${fence}text`,
    output,
    fence,
    ""
  ].join("\n");
}

function buildClaudeStatusHistoryLine(result: Awaited<ReturnType<typeof inspectClaudeCodeStatus>>): string {
  return (
    JSON.stringify({
      ts: new Date().toISOString(),
      target: result.target,
      tmux_target: result.tmuxTarget,
      status: result.status,
      confidence: result.confidence,
      heuristic: result.heuristic,
      analyzed_at: result.analyzedAt,
      lines: result.lines,
      bytes: result.bytes,
      total_bytes: result.totalBytes,
      truncated: result.truncated,
      signals: result.signals
    }) + "\n"
  );
}

function diffStats(diff: string): { additions: number; deletions: number; changed: boolean } {
  let additions = 0;
  let deletions = 0;
  for (const line of diff.split("\n")) {
    if (line.startsWith("+") && !line.startsWith("+++")) additions += 1;
    if (line.startsWith("-") && !line.startsWith("---")) deletions += 1;
  }
  return { additions, deletions, changed: Boolean(diff.trim()) };
}

function normalizeGitOutput(output: string): string {
  return output.trim() === "(no output)" ? "" : output;
}

function looksLikeGitError(output: string): boolean {
  const trimmed = output.trim();
  const lower = trimmed.toLowerCase();
  return (
    trimmed.startsWith("fatal:") ||
    trimmed.startsWith("error:") ||
    trimmed.startsWith("git unavailable or failed:") ||
    trimmed.startsWith("git exited with status") ||
    trimmed.startsWith("usage: git ") ||
    lower.includes("not a git repository")
  );
}

function previewText(value: string, maxLines = 40, maxChars = 12_000): string {
  const lines = value.replace(/\r\n/g, "\n").split("\n").slice(0, maxLines).join("\n");
  return lines.length > maxChars ? `${lines.slice(0, maxChars)}\n...[preview truncated]` : lines;
}

function changedStatusLines(status: string): string[] {
  return status
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("##"));
}

function jsonlEvent(event: string, data: Record<string, unknown>): string {
  return JSON.stringify({ ts: new Date().toISOString(), event, ...data }) + "\n";
}

function cleanOneLine(value: unknown, fallback: string, maxLength = 120): string {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  return (text || fallback).slice(0, maxLength);
}

function normalizeAgentId(value: unknown): string {
  const agent = cleanOneLine(value, "custom", 64).toLowerCase();
  if (!/^[a-z0-9][a-z0-9._-]{0,63}$/.test(agent)) {
    throw new CodexProError("agent must use only lowercase letters, numbers, dots, underscores, or hyphens.");
  }
  return agent;
}

function displayAgentName(agent: string, agentName?: unknown): string {
  const explicit = cleanOneLine(agentName, "", 80);
  if (explicit) return explicit;
  if (agent === "codex") return "Codex";
  if (agent === "opencode") return "OpenCode";
  if (agent === "pi") return "Pi";
  return agent;
}

function shellQuote(value: string): string {
  return `'${value.replace(/'/g, "'\\''")}'`;
}

function agentCommandHint(agent: string, planPath: string, model?: string): string {
  const modelArg = model ? ` --model ${shellQuote(model)}` : " --model '<provider/model>'";
  const quotedPlanPath = shellQuote(planPath);
  if (agent === "opencode") return `opencode run${modelArg} "$(cat ${quotedPlanPath})"`;
  if (agent === "pi") return `pi run${modelArg} "$(cat ${quotedPlanPath})"`;
  if (agent === "codex") return `Read ${planPath} and execute it in small, reviewable steps.`;
  return `Run your local implementation agent manually with ${planPath} as the task input.`;
}

async function readRawTextFileBounded(config: CodexProConfig, guard: PathGuard, workspace: Workspace, filePath: string): Promise<string> {
  const resolved = guard.resolve(workspace, filePath);
  await guard.assertTextFile(resolved.absPath, config.maxReadBytes);
  return fsp.readFile(resolved.absPath, "utf8");
}

function buildAgentPlanBody(options: {
  title: string;
  plan: string;
  workspace: Workspace;
  agent: string;
  agentName: string;
  model?: string;
  statusPath: string;
  diffPath: string;
  executionLogPath: string;
}): string {
  const modelLine = options.model ? `Model: ${options.model}\n` : "";
  return `# ${options.title}

Updated: ${new Date().toISOString()}
Workspace: ${options.workspace.root}
Target agent: ${options.agentName} (${options.agent})
${modelLine}
## Plan

${options.plan.trim()}

## Implementation contract

- Work from this plan in small, reviewable steps.
- Keep edits scoped to the requested task and existing project conventions.
- Run focused verification before handing work back.
- Update ${options.statusPath} with files touched, checks run, results, blockers, and review notes.
- Save the final review diff to ${options.diffPath} when practical.
- Append notable execution events to ${options.executionLogPath} when the implementation agent supports logging.
`;
}

async function writeAgentHandoff(
  config: CodexProConfig,
  guard: PathGuard,
  workspace: Workspace,
  options: {
    agent: string;
    agentName?: string;
    model?: string;
    title: string;
    plan: string;
    append: boolean;
    eventName: string;
  }
): Promise<{
  agent: string;
  agentName: string;
  model?: string;
  title: string;
  planPath: string;
  statusPath: string;
  diffPath: string;
  logPath: string;
  executionLogPath: string;
  prompt: string;
  writeResult: Awaited<ReturnType<typeof writeTextFile>>;
}> {
  await ensureAiBridge(config, guard, workspace);
  const agent = normalizeAgentId(options.agent);
  const agentName = displayAgentName(agent, options.agentName);
  const model = options.model ? cleanOneLine(options.model, "", 120) : undefined;
  const plan = String(options.plan ?? "").trim();
  if (!plan) throw new CodexProError("plan must not be empty.");
  const planPath = `${config.contextDir}/current-plan.md`;
  const statusPath = `${config.contextDir}/agent-status.md`;
  const legacyCodexStatusPath = `${config.contextDir}/codex-status.md`;
  const diffPath = `${config.contextDir}/implementation-diff.patch`;
  const logPath = `${config.contextDir}/session-log.jsonl`;
  const executionLogPath = `${config.contextDir}/execution-log.jsonl`;
  const body = buildAgentPlanBody({
    title: options.title,
    plan,
    workspace,
    agent,
    agentName,
    model,
    statusPath,
    diffPath,
    executionLogPath
  });

  let content = body;
  if (options.append) {
    const raw = await readRawTextFileBounded(config, guard, workspace, planPath);
    content = `${raw.trimEnd()}\n\n---\n\n${body}`;
  }

  const writeResult = await writeTextFile(config, guard, workspace, planPath, content, { createDirs: true, overwrite: true });
  const event = {
    agent,
    agent_name: agentName,
    model,
    title: options.title,
    plan_path: planPath,
    status_path: statusPath,
    diff_path: diffPath
  };
  const logResolved = guard.resolve(workspace, logPath, { forWrite: true });
  const executionLogResolved = guard.resolve(workspace, executionLogPath, { forWrite: true });
  await fsp.appendFile(logResolved.absPath, jsonlEvent(options.eventName, event), "utf8");
  await fsp.appendFile(executionLogResolved.absPath, jsonlEvent(options.eventName, event), "utf8");

  const promptLines = [
    `Read ${planPath} and execute it in small, reviewable steps.`,
    `After each meaningful change, update ${statusPath} with files touched, checks run, results, blockers, and the next review focus.`,
    `Before review, write the final diff to ${diffPath} when practical.`,
    agentCommandHint(agent, planPath, model)
  ];
  if (agent === "codex") {
    promptLines.splice(2, 0, `For legacy Codex handoffs, mirror key status notes to ${legacyCodexStatusPath} if your workflow expects that file.`);
  }
  const prompt = promptLines.join("\n");

  return {
    agent,
    agentName,
    model,
    title: options.title,
    planPath,
    statusPath,
    diffPath,
    logPath,
    executionLogPath,
    prompt,
    writeResult
  };
}

const READ_ONLY_ANNOTATIONS = { readOnlyHint: true, openWorldHint: false, destructiveHint: false };
const SESSION_READ_ANNOTATIONS = { readOnlyHint: true, openWorldHint: false, destructiveHint: false, idempotentHint: false };
const LOCAL_WRITE_ANNOTATIONS = { readOnlyHint: false, openWorldHint: false, destructiveHint: true, idempotentHint: false };
const BASH_ANNOTATIONS = { readOnlyHint: false, openWorldHint: true, destructiveHint: true, idempotentHint: false };
const HANDOFF_WRITE_ANNOTATIONS = { readOnlyHint: false, openWorldHint: false, destructiveHint: false, idempotentHint: false };

const workspaceManagers = new Map<string, WorkspaceManager>();

function workspaceManagerKey(config: CodexProConfig): string {
  return JSON.stringify({
    defaultRoot: config.defaultRoot,
    allowedRoots: [...config.allowedRoots].sort(),
    contextDir: config.contextDir
  });
}

function getSharedWorkspaceManager(config: CodexProConfig): WorkspaceManager {
  const key = workspaceManagerKey(config);
  const existing = workspaceManagers.get(key);
  if (existing) return existing;
  const manager = new WorkspaceManager(config);
  workspaceManagers.set(key, manager);
  return manager;
}

export function createCodexProServer(config: CodexProConfig): McpServer {
  const workspaces = getSharedWorkspaceManager(config);
  const guard = new PathGuard(config);
  const server = new McpServer({ name: "CodexPro", version: "0.28.5" }, { instructions: serverInstructions(config) });
  registerToolCardResource(server, config);

  registerCodexTool(
    config,
    server,
    "server_config",
    {
      title: "Server Config",
      description: "Show CodexPro server configuration, safety modes, limits, and blocked paths. Does not reveal auth tokens.",
      inputSchema: {},
      annotations: READ_ONLY_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Reading CodexPro server config...",
        "openai/toolInvocation/invoked": "CodexPro server config ready"
      }
    },
    async () => {
      const safeConfig = {
        defaultRoot: config.defaultRoot,
        allowedRoots: config.allowedRoots,
        host: config.host,
        port: config.port,
        widgetDomain: config.widgetDomain,
        authEnabled: Boolean(config.authToken),
        bashMode: config.bashMode,
        bashTranscript: config.bashTranscript,
        bashSessionId: config.bashSessionId ?? null,
        requireBashSession: config.requireBashSession,
        codexSessions: config.codexSessions,
        codexDir: config.codexDir,
        writeMode: config.writeMode,
        toolMode: config.toolMode,
        inheritEnv: config.inheritEnv,
        contextDir: config.contextDir,
        maxReadBytes: config.maxReadBytes,
        maxWriteBytes: config.maxWriteBytes,
        maxOutputBytes: config.maxOutputBytes,
        maxSearchResults: config.maxSearchResults,
        blockedGlobs: config.blockedGlobs
      };
      return textResult(`# CodexPro Server Config\n\n${JSON.stringify(safeConfig, null, 2)}`, safeConfig);
    }
  );

  registerCodexTool(
    config,
    server,
    "codexpro_self_test",
    {
      title: "CodexPro Self Test",
      description:
        "Run one controlled, local-only CodexPro diagnostic. It checks modes, expected tools, workspace access, skills, git, safe bash policy, selected-only Pro context, and optional .ai-bridge write/edit without touching source files.",
      inputSchema: {
        workspace_id: z.string().optional().describe("Workspace id from open_workspace. Omit to use default workspace."),
        write_probe: z.boolean().optional().describe("Create/edit only .ai-bridge/codexpro-self-test.md. Default: true."),
        bash_probe: z.boolean().optional().describe("Check bash policy with safe local commands only. Default: true."),
        pro_context_probe: z.boolean().optional().describe("Build a selected-only Pro context bundle in memory without writing pro-context.md. Default: true."),
        include_global_skills: z.boolean().optional().describe("Include user/plugin skill discovery in the inventory check. Default: true."),
        max_skills: z.number().int().min(1).max(120).optional().describe("Maximum skills to inspect during the inventory check. Default: 40.")
      },
      annotations: HANDOFF_WRITE_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Running CodexPro self-test...",
        "openai/toolInvocation/invoked": "CodexPro self-test complete"
      }
    },
    async (args) => {
      const workspace = workspaces.getWorkspace(args.workspace_id);
      const started = Date.now();
      const checks: Array<{ name: string; status: "pass" | "warn" | "fail"; detail: string }> = [];
      const filesTouched: string[] = [];
      const probePath = `${config.contextDir}/codexpro-self-test.md`;

      const check = (name: string, status: "pass" | "warn" | "fail", detail: string) => {
        checks.push({ name, status, detail: cleanOneLine(detail, detail, 260) });
      };

      check("workspace", "pass", workspace.root);
      check("tool mode", config.toolMode === "full" ? "pass" : "warn", `${config.toolMode}; expected tools: ${toolNamesForMode(config).length}`);
      check("write mode", config.writeMode === "off" ? "warn" : "pass", config.writeMode);
      check("bash mode", config.bashMode === "full" ? "warn" : "pass", config.bashMode);
      check(
        "http auth",
        config.requireHttpToken && !config.authToken ? "fail" : "pass",
        config.requireHttpToken ? "token required for public/non-loopback access" : "loopback token not required"
      );
      check("registered tool set", "pass", `${toolNamesForMode(config).length} tools for ${config.toolMode} mode`);

      try {
        const inventory = await codexproInventory(config, workspace, {
          includeGlobalSkills: parseBool(args.include_global_skills, true),
          includeMcpServers: true,
          maxSkills: limitInt(args.max_skills, 40, 1, 120)
        });
        check("inventory", "pass", `${inventory.skills.length} skills inspected, ${inventory.mcpServers.length} MCP server names visible`);
      } catch (error) {
        check("inventory", "fail", errorText(error));
      }

      try {
        const status = gitStatus(config, workspace);
        const gitFailed = looksLikeGitError(status);
        const changed = gitFailed ? 0 : changedStatusLines(status).length;
        check("git status", gitFailed ? "warn" : "pass", gitFailed ? status : `${changed} changed entries`);
      } catch (error) {
        check("git status", "fail", errorText(error));
      }

      if (parseBool(args.write_probe, true)) {
        if (config.writeMode === "off") {
          check("write/edit probe", "warn", "skipped because CODEXPRO_WRITE_MODE=off");
        } else {
          try {
            assertWriteToolAllowed(config, probePath);
            const content = [
              "# CodexPro Self Test",
              "",
              `Updated: ${new Date().toISOString()}`,
              `Workspace: ${workspace.root}`,
              "marker: before",
              ""
            ].join("\n");
            await writeTextFile(config, guard, workspace, probePath, content, { createDirs: true, overwrite: true });
            await editTextFile(config, guard, workspace, probePath, "marker: before", "marker: after", { expectedReplacements: 1 });
            const readBack = await readTextFile(config, guard, workspace, probePath, { maxBytes: 20_000 });
            if (!readBack.text.includes("marker: after")) throw new CodexProError("self-test edit marker was not found after edit.");
            const scopedStatus = gitStatus(config, workspace, guard, probePath);
            const scopedFiles = changedStatusLines(scopedStatus);
            filesTouched.push(probePath);
            check(
              "write/edit probe",
              scopedFiles.length && scopedFiles.every((line) => line.includes(probePath)) ? "pass" : "warn",
              scopedFiles.length ? `path-scoped status: ${scopedFiles.join(", ")}` : "path-scoped status clean after write/edit"
            );
          } catch (error) {
            check("write/edit probe", "fail", errorText(error));
          }
        }
      } else {
        check("write/edit probe", "warn", "skipped by request");
      }

      if (parseBool(args.pro_context_probe, true)) {
        try {
          if (!filesTouched.includes(probePath)) {
            check("selected-only pro context", "warn", "skipped because write probe did not create the selected file");
          } else {
            const context = await buildProContext(config, guard, workspace, {
              title: "CodexPro Self Test Context",
              selectedPaths: [probePath],
              includeImportantFiles: false,
              includeChangedFiles: false,
              includeDiff: false,
              includeAiBridge: false,
              maxFiles: 4,
              maxTotalBytes: 80_000
            });
            const exactOnly = context.filesIncluded.length === 1 && context.filesIncluded[0] === probePath;
            check(
              "selected-only pro context",
              exactOnly ? "pass" : "fail",
              exactOnly ? `included only ${probePath}` : `included ${context.filesIncluded.join(", ") || "no files"}`
            );
          }
        } catch (error) {
          check("selected-only pro context", "fail", errorText(error));
        }
      } else {
        check("selected-only pro context", "warn", "skipped by request");
      }

      if (parseBool(args.bash_probe, true)) {
        try {
          if (config.bashMode === "off") {
            check("bash policy", "warn", "bash disabled");
          } else {
            const bashProbeOptions = { timeoutMs: 10_000, sessionId: config.bashSessionId };
            const pwd = await runBash(config, guard, workspace, "pwd", bashProbeOptions);
            if (config.bashMode === "safe") {
              try {
                await runBash(config, guard, workspace, "ls $HOME", bashProbeOptions);
                check("bash policy", "fail", "safe bash allowed environment expansion unexpectedly");
              } catch {
                check("bash policy", pwd.exitCode === 0 ? "pass" : "warn", "safe bash allowed pwd and blocked environment expansion");
              }
            } else {
              check("bash policy", pwd.exitCode === 0 ? "warn" : "fail", "full bash is enabled; use only for trusted local repos");
            }
          }
        } catch (error) {
          check("bash policy", "fail", errorText(error));
        }
      } else {
        check("bash policy", "warn", "skipped by request");
      }

      check(
        "terms boundary",
        "pass",
        "local workspace bridge only; does not provide models, proxy model access, bypass quotas, or execute remote/local agents from MCP"
      );

      const failed = checks.filter((item) => item.status === "fail").length;
      const warned = checks.filter((item) => item.status === "warn").length;
      const passed = checks.filter((item) => item.status === "pass").length;
      const status = failed ? "fail" : warned ? "warn" : "pass";
      const text = [
        "# CodexPro Self Test",
        "",
        `Status: ${status}`,
        `Workspace: ${workspace.root}`,
        `Mode: tools=${config.toolMode}, write=${config.writeMode}, bash=${config.bashMode}${config.bashSessionId ? `, bash_session=${config.bashSessionId}${config.requireBashSession ? " required" : ""}` : ""}`,
        `Expected tools: ${toolNamesForMode(config).length}`,
        `Duration: ${Date.now() - started} ms`,
        "",
        "## Checks",
        "",
        ...checks.map((item) => `- ${item.status.toUpperCase()} ${item.name}: ${item.detail}`),
        "",
        "## Terms Boundary",
        "",
        "CodexPro exposes local repo tools to the ChatGPT session the user controls. It does not provide models, proxy model access, resell access, modify quotas, bypass limits, or run local implementation agents through remote MCP tools."
      ].join("\n");

      return textResult(text, {
        workspace_id: workspace.id,
        root: workspace.root,
        status,
        passed,
        warned,
        failed,
        duration_ms: Date.now() - started,
        expected_tools: toolNamesForMode(config),
        expected_tool_count: toolNamesForMode(config).length,
        bash_mode: config.bashMode,
        bash_session_id: config.bashSessionId ?? null,
        require_bash_session: config.requireBashSession,
        write_mode: config.writeMode,
        tool_mode: config.toolMode,
        files_touched: filesTouched,
        checks,
        terms_boundary: {
          local_workspace_bridge: true,
          provides_models: false,
          proxies_model_access: false,
          bypasses_quotas: false,
          remote_agent_execution: false
        }
      });
    }
  );

  registerCodexTool(
    config,
    server,
    "codexpro_inventory",
    {
      title: "CodexPro Inventory",
      description:
        "List CodexPro modes plus discovered skill names and configured MCP server names. Use this early when planning needs local agent capabilities.",
      inputSchema: {
        workspace_id: z.string().optional().describe("Workspace id from open_workspace. Omit to use default workspace."),
        include_global_skills: z
          .boolean()
          .optional()
          .describe(
            "Include user/global skills in inventory, including ~/.claude/skills. Default: false."
          ),
        include_mcp_servers: z.boolean().optional().describe("Include configured MCP server names from safe config files. Default: true."),
        max_skills: z.number().int().min(1).max(500).optional().describe("Maximum skills to list. Default: 120.")
      },
      annotations: READ_ONLY_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Reading CodexPro inventory...",
        "openai/toolInvocation/invoked": "CodexPro inventory ready"
      }
    },
    async (args) => {
      const workspace = workspaces.getWorkspace(args.workspace_id);
      const inventory = await codexproInventory(config, workspace, {
        includeGlobalSkills: parseBool(args.include_global_skills, true),
        includeMcpServers: parseBool(args.include_mcp_servers, true),
        maxSkills: limitInt(args.max_skills, 120, 1, 500)
      });
      return textResult(inventory.text, {
        workspace_id: workspace.id,
        root: workspace.root,
        bash_mode: config.bashMode,
        write_mode: config.writeMode,
        tool_mode: config.toolMode,
        skills: inventory.skills,
        skill_count: inventory.skills.length,
        mcp_servers: inventory.mcpServers,
        mcp_server_count: inventory.mcpServers.length,
        widget_uri: TOOL_CARD_URI
      });
    }
  );

  registerCodexTool(
    config,
    server,
    "load_skill",
    {
      title: "Load Skill",
      description:
        "Load the bounded SKILL.md body for an explicitly requested workspace, user, plugin, or Claude Code skill by name. Supports project skills under .claude/skills and user skills under ~/.claude/skills when global skills are enabled. Optionally include markdown files under references/. Do not auto-select skills; call this only when the user explicitly names a skill or WebGPT instructions require a specific skill.",
      inputSchema: {
        workspace_id: z.string().optional().describe("Workspace id from open_workspace. Omit to use default workspace."),
        name: z.string().describe("Exact skill name from skill_inventory or codexpro_inventory."),
        source: z.enum(["workspace", "user", "plugin", "other"]).optional().describe("Optional source when multiple skills share a name."),
        path: z.string().optional().describe("Exact sanitized path from skill_inventory when name/source are still ambiguous."),
        include_global_skills: z
          .boolean()
          .optional()
          .describe(
            "Allow load_skill to search user/global skill directories, including ~/.claude/skills, ~/.codex/skills, ~/.agents/skills, and plugin cache. Default: true."
          ),
        max_bytes: z.number().int().min(1000).max(100000).optional().describe("Maximum bytes to return from SKILL.md. Default: 40000."),
        include_references: z
          .boolean()
          .optional()
          .describe(
            "Include markdown files under the selected skill's references/ directory. Default: false."
          ),
        
        max_reference_bytes: z
          .number()
          .int()
          .min(1000)
          .max(100000)
          .optional()
          .describe("Maximum bytes to return from each references/*.md file. Default: 20000."),
        
        max_references: z
          .number()
          .int()
          .min(0)
          .max(100)
          .optional()
          .describe("Maximum number of markdown reference files to include. Default: 20.")
      },
      annotations: READ_ONLY_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Loading skill instructions...",
        "openai/toolInvocation/invoked": "Skill instructions loaded"
      }
    },
    async (args) => {
      const workspace = workspaces.getWorkspace(args.workspace_id);
      const loaded = await loadSkill(workspace, {
        name: String(args.name ?? ""),
        source: args.source,
        path: typeof args.path === "string" ? args.path : undefined,
        includeGlobal: parseBool(args.include_global_skills, true),
        maxBytes: limitInt(args.max_bytes, 40_000, 1_000, 100_000),
        includeReferences: parseBool(args.include_references, false),
        maxReferenceBytes: limitInt(args.max_reference_bytes, 20_000, 1_000, 100_000),
        maxReferences: limitInt(args.max_references, 20, 0, 100)
      });
      const truncated = loaded.truncated
        ? "\n\n[truncated: increase max_bytes if more context is required]"
        : "";
      
      const referencesText = loaded.references.length
        ? [
            "",
            "## References",
            "",
            ...loaded.references.map((reference) => {
              const referenceTruncated = reference.truncated
                ? "\n\n[truncated: increase max_reference_bytes if more context is required]"
                : "";
      
              return [
                `### ${reference.path}`,
                "",
                `Bytes: ${reference.bytes}/${reference.totalBytes}`,
                "",
                "```markdown",
                `${reference.text}${referenceTruncated}`,
                "```"
              ].join("\n");
            })
          ].join("\n")
        : "";
      
      const warningsText = loaded.warnings.length
        ? [
            "",
            "## Skill Load Warnings",
            "",
            ...loaded.warnings.map((warning) => `- ${warning}`)
          ].join("\n")
        : "";
      
      const text = `# Load Skill
      
      Name: ${loaded.skill.name}
      Source: ${loaded.skill.source}
      Path: ${loaded.skill.path}
      Bytes: ${loaded.bytes}/${loaded.totalBytes}
      
      ## SKILL.md
      
      \`\`\`markdown
      ${loaded.text}${truncated}
      \`\`\`${referencesText}${warningsText}`;
      return textResult(text, {
        workspace_id: workspace.id,
        root: workspace.root,
        skill: loaded.skill,
        bytes: loaded.bytes,
        total_bytes: loaded.totalBytes,
        truncated: loaded.truncated,
        text: loaded.text,
        references: loaded.references,
        reference_count: loaded.references.length,
        warnings: loaded.warnings
      });
    }
  );

  registerCodexTool(
    config,
    server,
    "list_workspaces",
    {
      title: "List Workspaces",
      description: "List currently opened CodexPro workspaces for this MCP session.",
      inputSchema: {},
      annotations: READ_ONLY_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Listing CodexPro workspaces...",
        "openai/toolInvocation/invoked": "CodexPro workspaces listed"
      }
    },
    async () => {
      const current = workspaces.listWorkspaces();
      const text = current.length
        ? current.map((workspace) => `- ${workspace.id} — ${workspace.root} (opened ${workspace.openedAt})`).join("\n")
        : "No workspaces opened yet. Call open_workspace first.";
      return textResult(text, { workspaces: current, count: current.length });
    }
  );

  registerCodexTool(
    config,
    server,
    "open_current_workspace",
    {
      title: "Open Current Workspace",
      description:
        "Use this once at the start to open the configured default workspace without accepting a path. Do not call open_workspace after this unless switching roots.",
      inputSchema: {
        include_tree: z.boolean().optional().describe("Include a compact file tree. Default: false for speed."),
        max_depth: z.number().int().min(1).max(8).optional().describe("Tree depth when include_tree=true. Default: 2."),
        include_skills: z.boolean().optional().describe("Discover workspace, user, and plugin skills by name/description. Default: true."),
        include_global_skills: z.boolean().optional().describe("Also scan installed user/plugin skills when include_skills=true. Default: true.")
      },
      annotations: SESSION_READ_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Opening current CodexPro workspace...",
        "openai/toolInvocation/invoked": "Current CodexPro workspace opened"
      }
    },
    async (args) => {
      const workspace = workspaces.defaultWorkspace();
      const summary = await workspaceSummary(config, guard, workspace, {
        includeTree: parseBool(args.include_tree, false),
        maxDepth: limitInt(args.max_depth, 2, 1, 8),
        includeSkills: parseBool(args.include_skills, true),
        includeGlobalSkills: parseBool(args.include_global_skills, true),
        bootstrapContext: false
      });
      return textResult(summary.text, {
        workspace_id: summary.workspaceId,
        root: summary.root,
        agents_loaded: summary.agentsLoaded,
        agents_path: summary.agentsPath,
        webgpt_loaded: summary.webgptLoaded,
        webgpt_path: summary.webgptPath,
        webgpt_files: summary.webgptFiles,
        webgpt_warnings: summary.webgptWarnings,
        skills: summary.skills,
        skill_inventory: summary.skillInventory,
        skill_counts: summary.skillCounts,
        tree: summary.tree,
        git_status: summary.gitStatus,
        bash_mode: config.bashMode,
        write_mode: config.writeMode,
        tool_mode: config.toolMode
      });
    }
  );

  registerCodexTool(
    config,
    server,
    "open_workspace",
    {
      title: "Open Workspace",
      description:
        "Open a local project directory as a CodexPro workspace. Returns a workspace_id plus git status, WebGPT instruction status, AGENTS.md, skills, and a compact file tree.",
      inputSchema: {
        root: z.string().optional().describe("Project directory to open. Omit to use CODEXPRO_ROOT/current working directory. Supports ~/ paths."),
        path: z.string().optional().describe("Alias for root. Useful for clients that naturally send path instead of root."),
        include_tree: z.boolean().optional().describe("Include a compact file tree. Default: true."),
        max_depth: z.number().int().min(1).max(8).optional().describe("Tree depth. Default: 3."),
        max_files: z.number().int().min(1).max(3000).optional().describe("Alias for maximum tree entries. Default: 500."),
        include_skills: z.boolean().optional().describe("Discover workspace, user, and plugin skills by name/description. Default: true."),
        include_global_skills: z.boolean().optional().describe("Also scan installed user/plugin skills when include_skills=true. Default: true."),
        bootstrap_context: z.boolean().optional().describe("Deprecated and ignored. Use handoff_to_agent to create .ai-bridge files.")
      },
      annotations: SESSION_READ_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Opening CodexPro workspace...",
        "openai/toolInvocation/invoked": "CodexPro workspace opened"
      }
    },
    async (args) => {
      if (args.root && args.path && args.root !== args.path) {
        throw new CodexProError("open_workspace accepts either root or path. If both are provided, they must match.");
      }
      const workspace = workspaces.openWorkspace(args.root ?? args.path);
      const summary = await workspaceSummary(config, guard, workspace, {
        includeTree: args.include_tree !== false,
        maxDepth: limitInt(args.max_depth, 3, 1, 8),
        maxEntries: limitInt(args.max_files, 500, 1, 3000),
        includeSkills: parseBool(args.include_skills, true),
        includeGlobalSkills: parseBool(args.include_global_skills, true),
        bootstrapContext: false
      });
      return textResult(summary.text, {
        workspace_id: summary.workspaceId,
        root: summary.root,
        agents_loaded: summary.agentsLoaded,
        agents_path: summary.agentsPath,
        webgpt_loaded: summary.webgptLoaded,
        webgpt_path: summary.webgptPath,
        webgpt_files: summary.webgptFiles,
        webgpt_warnings: summary.webgptWarnings,
        skills: summary.skills,
        skill_inventory: summary.skillInventory,
        skill_counts: summary.skillCounts,
        tree: summary.tree,
        git_status: summary.gitStatus,
        bash_mode: config.bashMode,
        write_mode: config.writeMode,
        tool_mode: config.toolMode
      });
    }
  );

  registerCodexTool(
    config,
    server,
    "workspace_snapshot",
    {
      title: "Workspace Snapshot",
      description: "Return git status, recent commits, .ai-bridge context, and a compact tree for an opened workspace.",
      inputSchema: {
        workspace_id: z.string().optional().describe("Workspace id from open_workspace. Omit to use default workspace."),
        max_depth: z.number().int().min(1).max(8).optional().describe("Tree depth. Default: 3."),
        max_files: z.number().int().min(1).max(3000).optional().describe("Alias for maximum tree entries. Default: 500."),
        include_skills: z.boolean().optional().describe("Discover repo-local skills. Default: false for speed."),
        include_global_skills: z.boolean().optional().describe("Also scan home-level skill folders when include_skills=true. Default: false.")
      },
      annotations: READ_ONLY_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Collecting workspace snapshot...",
        "openai/toolInvocation/invoked": "Workspace snapshot ready"
      }
    },
    async (args) => {
      const workspace = workspaces.getWorkspace(args.workspace_id);
      const summary = await workspaceSummary(config, guard, workspace, {
        includeTree: true,
        maxDepth: limitInt(args.max_depth, 3, 1, 8),
        maxEntries: limitInt(args.max_files, 500, 1, 3000),
        includeSkills: parseBool(args.include_skills, false),
        includeGlobalSkills: parseBool(args.include_global_skills, false)
      });
      const ai = await readAiBridgeContext(config, guard, workspace);
      const text = `${summary.text}\n\n## AI handoff context\n\n${ai.text}`;
      return textResult(text, {
        workspace_id: workspace.id,
        root: workspace.root,
        agents_loaded: summary.agentsLoaded,
        agents_path: summary.agentsPath,
        skills: summary.skills,
        skill_inventory: summary.skillInventory,
        skill_counts: summary.skillCounts,
        tree: summary.tree,
        git_status: summary.gitStatus,
        ai_context_files: ai.files,
        bash_mode: config.bashMode,
        write_mode: config.writeMode,
        tool_mode: config.toolMode
      });
    }
  );

  registerCodexTool(
    config,
    server,
    "tree",
    {
      title: "File Tree",
      description: "List files and directories inside the workspace, excluding blocked paths.",
      inputSchema: {
        workspace_id: z.string().optional().describe("Workspace id from open_workspace. Omit to use default workspace."),
        path: z.string().optional().describe("Directory relative to workspace root. Default: ."),
        max_depth: z.number().int().min(1).max(12).optional().describe("Maximum depth. Default: 4."),
        include_hidden: z.boolean().optional().describe("Include dotfiles/dotfolders that are not blocked. Default: false."),
        max_entries: z.number().int().min(1).max(3000).optional().describe("Maximum entries. Default: 800.")
      },
      annotations: READ_ONLY_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Listing workspace files...",
        "openai/toolInvocation/invoked": "Workspace files listed"
      }
    },
    async (args) => {
      const workspace = workspaces.getWorkspace(args.workspace_id);
      const result = await repoTree(config, guard, workspace, {
        path: args.path ?? ".",
        maxDepth: limitInt(args.max_depth, 4, 1, 12),
        includeHidden: parseBool(args.include_hidden, false),
        maxEntries: limitInt(args.max_entries, 800, 1, 3000)
      });
      return textResult(result.text, { workspace_id: workspace.id, root: workspace.root, ...result });
    }
  );

  registerCodexTool(
    config,
    server,
    "search",
    {
      title: "Search Files",
      description: "Use this for targeted verification or code lookup. Prefer one specific final search instead of repeated broad verification searches.",
      inputSchema: {
        workspace_id: z.string().optional().describe("Workspace id from open_workspace. Omit to use default workspace."),
        query: z.string().describe("Text or regex to search for."),
        regex: z.boolean().optional().describe("Treat query as a regular expression. Default: false."),
        path: z.string().optional().describe("Directory or file relative to workspace root. Default: ."),
        glob: z.string().optional().describe("Optional glob, for example src/**/*.ts."),
        include_hidden: z.boolean().optional().describe("Include hidden files that are not blocked. Default: false."),
        max_results: z.number().int().min(1).max(2000).optional().describe("Maximum results. Default from config.")
      },
      annotations: READ_ONLY_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Searching workspace...",
        "openai/toolInvocation/invoked": "Workspace search complete"
      }
    },
    async (args) => {
      const workspace = workspaces.getWorkspace(args.workspace_id);
      const result = await searchWorkspace(config, guard, workspace, {
        query: args.query,
        regex: parseBool(args.regex, false),
        root: args.path ?? ".",
        glob: args.glob,
        includeHidden: parseBool(args.include_hidden, false),
        maxResults: limitInt(args.max_results, config.maxSearchResults, 1, config.maxSearchResults)
      });
      return textResult(result.text, { workspace_id: workspace.id, root: workspace.root, ...result });
    }
  );

  registerCodexTool(
    config,
    server,
    "read",
    {
      title: "Read File",
      description: "Read a specific text file with line numbers. Avoid rereading files after write/edit unless exact final content is needed.",
      inputSchema: {
        workspace_id: z.string().optional().describe("Workspace id from open_workspace. Omit to use default workspace."),
        path: z.string().describe("File path relative to workspace root."),
        start_line: z.number().int().min(1).optional().describe("First line to read. Default: 1."),
        end_line: z.number().int().min(1).optional().describe("Last line to read. Default: end of file."),
        max_bytes: z.number().int().min(1000).max(2000000).optional().describe("Maximum file bytes. Capped by server config.")
      },
      annotations: READ_ONLY_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Reading file...",
        "openai/toolInvocation/invoked": "File read"
      }
    },
    async (args) => {
      const workspace = workspaces.getWorkspace(args.workspace_id);
      const result = await readTextFile(config, guard, workspace, args.path, {
        startLine: args.start_line,
        endLine: args.end_line,
        maxBytes: args.max_bytes
      });
      const text = `# Read File\n\nPath: ${result.path}\nLines: ${result.startLine}-${result.endLine} of ${result.totalLines}\nBytes: ${result.bytes}\nSHA-256: ${result.sha256}\n\n\`\`\`text\n${result.text}\n\`\`\``;
      return textResult(text, { workspace_id: workspace.id, root: workspace.root, ...result });
    }
  );

  registerCodexTool(
    config,
    server,
    "write",
    {
      title: "Write File",
      description: "Create or overwrite a meaningful text file inside the workspace. Returns a unified diff; do not create empty placeholder files.",
      inputSchema: {
        workspace_id: z.string().optional().describe("Workspace id from open_workspace. Omit to use default workspace."),
        path: z.string().describe("File path relative to workspace root."),
        content: z.string().describe("Complete file contents to write."),
        create_dirs: z.boolean().optional().describe("Create parent directories if missing. Default: true."),
        overwrite: z.boolean().optional().describe("Allow overwriting existing files. Default: true.")
      },
      annotations: LOCAL_WRITE_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Writing file...",
        "openai/toolInvocation/invoked": "File written"
      }
    },
    async (args) => {
      const workspace = workspaces.getWorkspace(args.workspace_id);
      const resolved = guard.resolve(workspace, args.path, { forWrite: true });
      assertWriteToolAllowed(config, resolved.relPath);
      const result = await writeTextFile(config, guard, workspace, args.path, String(args.content ?? ""), {
        createDirs: args.create_dirs !== false,
        overwrite: args.overwrite !== false
      });
      const text = `# Write File\n\nPath: ${result.path}\nExisted before: ${result.existed}\nBytes: ${result.bytes}\nSHA-256: ${result.sha256}\nDiff stats: +${result.diff.additions} -${result.diff.deletions}${diffBlock(result.diff.diff)}`;
      return textResult(text, {
        workspace_id: workspace.id,
        root: workspace.root,
        path: result.path,
        existed: result.existed,
        bytes: result.bytes,
        sha256: result.sha256,
        additions: result.diff.additions,
        deletions: result.diff.deletions,
        diff: result.diff.diff
      });
    }
  );

  registerCodexTool(
    config,
    server,
    "edit",
    {
      title: "Edit File",
      description: "Apply a targeted exact text replacement inside a workspace text file. Returns a unified diff.",
      inputSchema: {
        workspace_id: z.string().optional().describe("Workspace id from open_workspace. Omit to use default workspace."),
        path: z.string().describe("File path relative to workspace root."),
        old_text: z.string().describe("Exact text to replace. Must match once unless replace_all=true."),
        new_text: z.string().describe("Replacement text."),
        replace_all: z.boolean().optional().describe("Replace all occurrences. Default: false."),
        expected_replacements: z.number().int().min(1).optional().describe("Fail if actual replacement count differs.")
      },
      annotations: LOCAL_WRITE_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Editing file...",
        "openai/toolInvocation/invoked": "File edited"
      }
    },
    async (args) => {
      const workspace = workspaces.getWorkspace(args.workspace_id);
      const resolved = guard.resolve(workspace, args.path, { forWrite: true });
      assertWriteToolAllowed(config, resolved.relPath);
      const result = await editTextFile(config, guard, workspace, args.path, String(args.old_text ?? ""), String(args.new_text ?? ""), {
        replaceAll: parseBool(args.replace_all, false),
        expectedReplacements: args.expected_replacements
      });
      const text = `# Edit File\n\nPath: ${result.path}\nReplacements: ${result.replacements}\nBytes: ${result.bytes}\nSHA-256: ${result.sha256}\nDiff stats: +${result.diff.additions} -${result.diff.deletions}${diffBlock(result.diff.diff)}`;
      return textResult(text, {
        workspace_id: workspace.id,
        root: workspace.root,
        path: result.path,
        replacements: result.replacements,
        bytes: result.bytes,
        sha256: result.sha256,
        additions: result.diff.additions,
        deletions: result.diff.deletions,
        diff: result.diff.diff
      });
    }
  );

  registerCodexTool(
    config,
    server,
    "bash",
    {
      title: "Bash",
      description:
        "Run one allowlisted verification command in the workspace, such as tests, build, lint, typecheck, or a project script. Do not use for git status/diff or file inspection; use show_changes, tree, search, and read instead. Do not chain commands with &&, pipes, redirects, or shell file readers.",
      inputSchema: {
        workspace_id: z.string().optional().describe("Workspace id from open_workspace. Omit to use default workspace."),
        command: z.string().describe("Command to run."),
        session_id: z.string().optional().describe(config.requireBashSession && config.bashSessionId ? `Required bash session id for this server: ${config.bashSessionId}.` : "Optional bash session id. If configured on the server, a provided value must match it."),
        cwd: z.string().optional().describe("Working directory relative to workspace root. Default: ."),
        timeout_ms: z.number().int().min(1000).max(180000).optional().describe("Timeout in milliseconds. Default: 30000.")
      },
      annotations: BASH_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Running bash command...",
        "openai/toolInvocation/invoked": "Bash command finished"
      }
    },
    async (args) => {
      const workspace = workspaces.getWorkspace(args.workspace_id);
      const result = await runBash(config, guard, workspace, String(args.command ?? ""), {
        cwd: args.cwd,
        timeoutMs: args.timeout_ms,
        sessionId: args.session_id
      });
      const text = bashTextResult(config, result);
      return textResult(text, { workspace_id: workspace.id, root: workspace.root, ...result, bash_session_id: result.bashSessionId ?? null });
    }
  );

  registerCodexTool(
    config,
    server,
    "git_status",
    {
      title: "Git Status",
      description: "Show git branch and changed files for the workspace.",
      inputSchema: {
        workspace_id: z.string().optional().describe("Workspace id from open_workspace. Omit to use default workspace.")
      },
      annotations: READ_ONLY_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Reading git status...",
        "openai/toolInvocation/invoked": "Git status ready"
      }
    },
    async (args) => {
      const workspace = workspaces.getWorkspace(args.workspace_id);
      const status = gitStatus(config, workspace);
      const statusError = looksLikeGitError(status) ? status : "";
      const changedFiles = statusError ? [] : changedStatusLines(status);
      return textResult(status, {
        workspace_id: workspace.id,
        root: workspace.root,
        status,
        status_error: statusError || undefined,
        changed_files: changedFiles,
        changed: !statusError && changedFiles.length > 0
      });
    }
  );

  registerCodexTool(
    config,
    server,
    "git_diff",
    {
      title: "Git Diff",
      description: "Show current unstaged or staged git diff, optionally scoped to a file.",
      inputSchema: {
        workspace_id: z.string().optional().describe("Workspace id from open_workspace. Omit to use default workspace."),
        path: z.string().optional().describe("Optional file path relative to workspace root."),
        staged: z.boolean().optional().describe("Show staged diff. Default: false."),
        include_diff: z.boolean().optional().describe("Include the raw unified diff in the response. Default: true. Set false for stats-only checks.")
      },
      annotations: READ_ONLY_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Reading git diff...",
        "openai/toolInvocation/invoked": "Git diff ready"
      }
    },
    async (args) => {
      const workspace = workspaces.getWorkspace(args.workspace_id);
      const rawDiff = normalizeGitOutput(gitDiff(config, guard, workspace, args.path, parseBool(args.staged, false)));
      const diffError = rawDiff && looksLikeGitError(rawDiff) ? rawDiff : "";
      const stats = diffError ? { additions: 0, deletions: 0, changed: false } : diffStats(rawDiff);
      const includeDiff = parseBool(args.include_diff, true);
      const text = diffError
        ? diffError
        : includeDiff
        ? rawDiff
        : [
            "# Git Diff",
            "",
            `Workspace: ${workspace.root}`,
            `Path: ${args.path ?? "workspace diff"}`,
            `Staged: ${parseBool(args.staged, false)}`,
            `Diff stats: +${stats.additions} -${stats.deletions}`,
            "",
            "Raw diff omitted by include_diff=false."
          ].join("\n");
      return textResult(text, {
        workspace_id: workspace.id,
        root: workspace.root,
        path: args.path ?? "workspace diff",
        staged: parseBool(args.staged, false),
        include_diff: includeDiff,
        diff_error: diffError || undefined,
        additions: stats.additions,
        deletions: stats.deletions,
        changed: !diffError && stats.changed,
        diff: diffError || includeDiff ? rawDiff : ""
      });
    }
  );

  registerCodexTool(
    config,
    server,
    "show_changes",
    {
      title: "Show Changes",
      description: "Summarize the current workspace changes in one review-oriented result with git status, diff stats, and optional diff. Use this instead of bash git status, bash git diff, git_status, or git_diff when reviewing work.",
      inputSchema: {
        workspace_id: z.string().optional().describe("Workspace id from open_workspace. Omit to use default workspace."),
        path: z.string().optional().describe("Optional file path relative to workspace root."),
        staged: z.boolean().optional().describe("Show staged diff. Default: false."),
        include_diff: z.boolean().optional().describe("Include the unified diff. Default: true.")
      },
      annotations: READ_ONLY_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Summarizing workspace changes...",
        "openai/toolInvocation/invoked": "Workspace changes summarized"
      }
    },
    async (args) => {
      const workspace = workspaces.getWorkspace(args.workspace_id);
      const scopedPath = typeof args.path === "string" ? args.path : undefined;
      const status = gitStatus(config, workspace, guard, scopedPath);
      const includeDiff = parseBool(args.include_diff, true);
      const rawDiff = includeDiff ? normalizeGitOutput(gitDiff(config, guard, workspace, scopedPath, parseBool(args.staged, false))) : "";
      const statusError = looksLikeGitError(status) ? status : "";
      const diffError = rawDiff && looksLikeGitError(rawDiff) ? rawDiff : "";
      const diff = diffError ? "" : rawDiff;
      const stats = diffStats(diff);
      const changedFiles = statusError ? [] : changedStatusLines(status);
      const changedText = statusError
        ? `- Git status unavailable: ${statusError}`
        : changedFiles.length
          ? changedFiles.map((line) => `- ${line}`).join("\n")
          : "- No changed files.";
      const diffText = includeDiff
        ? diffError
          ? `\n\nGit diff unavailable: ${diffError}`
          : diff
          ? diffBlock(diff)
          : "\n\nNo diff output."
        : "\n\nDiff omitted by request.";
      const text = `# Show Changes\n\nWorkspace: ${workspace.root}\n\n## Changed\n\n${changedText}\n\n## Diff stats\n\n+${stats.additions} -${stats.deletions}${diffText}`;
      return textResult(text, {
        workspace_id: workspace.id,
        root: workspace.root,
        path: args.path ?? "workspace changes",
        status,
        status_error: statusError || undefined,
        diff_error: diffError || undefined,
        changed_files: changedFiles,
        staged: parseBool(args.staged, false),
        include_diff: includeDiff,
        additions: stats.additions,
        deletions: stats.deletions,
        changed: !statusError && (changedFiles.length > 0 || stats.changed),
        diff
      });
    }
  );

  registerCodexTool(
    config,
    server,
    "read_handoff",
    {
      title: "Read Handoff",
      description: "Read the shared .ai-bridge planning files used for ChatGPT-to-agent coordination.",
      inputSchema: {
        workspace_id: z.string().optional().describe("Workspace id from open_workspace. Omit to use default workspace.")
      },
      annotations: READ_ONLY_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Reading agent handoff context...",
        "openai/toolInvocation/invoked": "Agent handoff context ready"
      }
    },
    async (args) => {
      const workspace = workspaces.getWorkspace(args.workspace_id);
      const context = await readAiBridgeContext(config, guard, workspace);
      return textResult(context.text, {
        workspace_id: workspace.id,
        root: workspace.root,
        files: context.files,
        file_count: context.files.length,
        preview: previewText(context.text)
      });
    }
  );

  registerCodexTool(
    config,
    server,
    "codex_context",
    {
      title: "Codex Context",
      description:
        "Load workspace context in one call: WebGPT instructions from .claude/WEBGPT.md with simple @file.md imports, AGENTS instructions for a target path, .ai-bridge handoff files, and optional git status/diff. Does not load CLAUDE.md.",
      inputSchema: {
        workspace_id: z.string().optional().describe("Workspace id from open_workspace. Omit to use default workspace."),
        target_path: z.string().optional().describe("Workspace-relative file or directory whose AGENTS instruction chain should be loaded. WebGPT instructions always come from .claude/WEBGPT.md. Default: ."),
        include_ai_bridge: z.boolean().optional().describe("Include .ai-bridge plan, agent status, diff, decisions, questions, and execution log. Default: true."),
        include_git: z.boolean().optional().describe("Include git status. Default: true."),
        include_diff: z.boolean().optional().describe("Include full git diff. Default: false for speed/noise."),
        max_agent_bytes: z.number().int().min(1000).max(200000).optional().describe("Maximum bytes per AGENTS/WebGPT instruction file. Default: 60000.")
      },
      annotations: READ_ONLY_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Loading Codex context...",
        "openai/toolInvocation/invoked": "Codex context ready"
      }
    },
    async (args) => {
      const workspace = workspaces.getWorkspace(args.workspace_id);
      const context = await readCodexContext(config, guard, workspace, {
        targetPath: args.target_path,
        includeAiBridge: args.include_ai_bridge,
        includeGit: args.include_git,
        includeDiff: parseBool(args.include_diff, false),
        maxAgentBytes: args.max_agent_bytes
      });
      return textResult(context.text, {
        workspace_id: context.workspaceId,
        root: context.root,
        target_path: context.targetPath,
        webgpt_loaded: context.webgptLoaded,
        webgpt_files: context.webgptFiles,
        webgpt_warnings: context.webgptWarnings,
        agents_files: context.agentsFiles,
        ai_context_files: context.aiContextFiles,
        included_git_status: context.gitStatus !== undefined,
        included_git_diff: context.gitDiff !== undefined,
        preview: previewText(context.text)
      });
    }
  );

  registerCodexTool(
    config,
    server,
    "export_pro_context",
    {
      title: "Export Pro Context",
      description:
        "Create .ai-bridge/pro-context.md with repo tree, git state, selected files, and handoff context for high-context ChatGPT planning without live MCP tool calls.",
      inputSchema: {
        workspace_id: z.string().optional().describe("Workspace id from open_workspace. Omit to use default workspace."),
        title: z.string().optional().describe("Markdown title for the context bundle."),
        selected_paths: z.array(z.string()).optional().describe("Specific workspace-relative files to include."),
        extra_globs: z.array(z.string()).optional().describe("Additional workspace-relative glob patterns to include, for example src/**/*.ts."),
        include_important_files: z.boolean().optional().describe("Auto-include important root config/docs such as AGENTS.md, README.md, and package.json. Default: true."),
        include_changed_files: z.boolean().optional().describe("Auto-include currently changed files from git status. Default: true."),
        include_diff: z.boolean().optional().describe("Include the current git diff. Default: true."),
        include_ai_bridge: z.boolean().optional().describe("Include existing .ai-bridge planning files. Default: true."),
        max_depth: z.number().int().min(1).max(6).optional().describe("Repository tree depth. Default: 3."),
        max_files: z.number().int().min(1).max(80).optional().describe("Maximum file contents to include. Default: 24."),
        max_file_bytes: z.number().int().min(1000).max(250000).optional().describe("Maximum bytes per included file. Default: 60000."),
        max_total_bytes: z.number().int().min(20000).max(2000000).optional().describe("Maximum bytes in the generated bundle.")
      },
      annotations: HANDOFF_WRITE_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Exporting Pro context...",
        "openai/toolInvocation/invoked": "Pro context exported"
      }
    },
    async (args) => {
      const workspace = workspaces.getWorkspace(args.workspace_id);
      const result = await exportProContext(config, guard, workspace, {
        title: args.title,
        selectedPaths: args.selected_paths,
        extraGlobs: args.extra_globs,
        includeImportantFiles: args.include_important_files,
        includeChangedFiles: args.include_changed_files,
        includeDiff: args.include_diff,
        includeAiBridge: args.include_ai_bridge,
        maxDepth: args.max_depth,
        maxFiles: args.max_files,
        maxFileBytes: args.max_file_bytes,
        maxTotalBytes: args.max_total_bytes
      });
      const text = `# Export Pro Context\n\nWrote ${result.path}.\nBytes: ${result.bytes}\nFiles included: ${result.filesIncluded.length}\nFiles skipped: ${result.filesSkipped.length}\nTruncated: ${result.truncated}\n\nPaste ${result.path} into a high-context planning model when MCP tools are unavailable, then save the returned plan with codexpro pro-apply.`;
      return textResult(text, {
        workspace_id: workspace.id,
        root: workspace.root,
        path: result.path,
        bytes: result.bytes,
        files_included: result.filesIncluded,
        files_skipped: result.filesSkipped,
        truncated: result.truncated
      });
    }
  );

  
  registerCodexTool(
    config,
    server,
    "list_claude_code_targets",
    {
      title: "List Claude Code Targets",
      description:
        "List allowlisted tmux targets that CodexPro is allowed to send text to for Claude Code. Targets come from CODEXPRO_CLAUDE_TARGETS.",
      inputSchema: {}
    },
    async () => {
      const targets = listClaudeCodeTargets();
  
      const text = targets.length
        ? [
            "# Claude Code Targets",
            "",
            ...targets.map((target) => `- ${target.name} -> ${target.tmuxTarget}`)
          ].join("\n")
        : [
            "# Claude Code Targets",
            "",
            "No Claude Code targets configured.",
            "",
            "Set CODEXPRO_CLAUDE_TARGETS, for example:",
            "",
            "```bash",
            "export CODEXPRO_CLAUDE_TARGETS=cc-stock,cc-codexpro",
            "```",
            "",
            "Or use aliases:",
            "",
            "```bash",
            "export CODEXPRO_CLAUDE_TARGETS=stock=cc-stock,codexpro=cc-codexpro:0.0",
            "```"
          ].join("\n");
  
      return textResult(text, {
        enabled: targets.length > 0,
        targets: targets.map((target) => ({
          name: target.name,
          tmux_target: target.tmuxTarget
        }))
      });
    }
  );

  registerCodexTool(
    config,
    server,
    "send_to_claude_code",
    {
      title: "Send to Claude Code",
      description:
        "Paste text into an allowlisted tmux target running Claude Code. This does not read Claude Code output; use capture_claude_output separately for recent visible output. By default it only pastes text; set submit=true to press Enter.",
      inputSchema: {
        target: z
          .string()
          .min(1)
          .describe(
            "Allowlisted Claude Code target name from CODEXPRO_CLAUDE_TARGETS, not an arbitrary tmux target."
          ),
  
        text: z
          .string()
          .min(1)
          .max(200_000)
          .describe("Text to paste into the Claude Code terminal."),
  
        submit: z
          .boolean()
          .optional()
          .describe(
            "If true, press Enter after pasting. Default: false."
          )
      }
    },
    async (args) => {
      const result = await sendToClaudeCode({
        target: String(args.target ?? ""),
        text: String(args.text ?? ""),
        submit: parseBool(args.submit, false)
      });
  
      const text = [
        "# Sent to Claude Code",
        "",
        `Target: ${result.target}`,
        `tmux target: ${result.tmuxTarget}`,
        `Submitted: ${result.submitted ? "yes" : "no"}`,
        `Bytes: ${result.bytes}`,
        "",
        result.submitted
          ? "The text was pasted and Enter was sent."
          : "The text was pasted only. Review it in the Claude Code terminal and press Enter manually if appropriate."
      ].join("\n");
  
      return textResult(text, {
        ok: result.ok,
        target: result.target,
        tmux_target: result.tmuxTarget,
        submitted: result.submitted,
        bytes: result.bytes
      });
    }
  );
  
  registerCodexTool(
    config,
    server,
    "invoke_claude_code_skill",
    {
      title: "Invoke Claude Code Skill",
      description:
        "Paste a Claude Code slash skill command into an allowlisted tmux target. This only constructs '/skill-name arguments' and sends it to Claude Code; it does not execute scripts or bypass Claude Code permissions.",
      inputSchema: {
        target: z
          .string()
          .min(1)
          .describe(
            "Allowlisted Claude Code target name from CODEXPRO_CLAUDE_TARGETS."
          ),
  
        skill: z
          .string()
          .min(1)
          .max(128)
          .regex(/^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/)
          .describe(
            "Claude Code skill name without leading slash, for example review-to-plan."
          ),
  
        arguments: z
          .string()
          .optional()
          .describe("Arguments to pass after the slash skill command."),
  
        submit: z
          .boolean()
          .optional()
          .describe(
            "If true, press Enter after pasting the slash command. Default: false."
          )
      }
    },
    async (args) => {
      const result = await invokeClaudeCodeSkill({
        target: String(args.target ?? ""),
        skill: String(args.skill ?? ""),
        arguments: typeof args.arguments === "string" ? args.arguments : undefined,
        submit: parseBool(args.submit, false)
      });
  
      const text = [
        "# Claude Code Skill Command Sent",
        "",
        `Target: ${result.target}`,
        `tmux target: ${result.tmuxTarget}`,
        `Skill: ${result.skill}`,
        `Submitted: ${result.submitted ? "yes" : "no"}`,
        `Bytes: ${result.bytes}`,
        "",
        "Command:",
        "",
        "```text",
        result.command,
        "```",
        "",
        result.submitted
          ? "The command was pasted and Enter was sent."
          : "The command was pasted only. Review it in the Claude Code terminal and press Enter manually if appropriate."
      ].join("\n");
  
      return textResult(text, {
        ok: result.ok,
        target: result.target,
        tmux_target: result.tmuxTarget,
        skill: result.skill,
        command: result.command,
        submitted: result.submitted,
        bytes: result.bytes
      });
    }
  );

  registerCodexTool(
    config,
    server,
    "capture_claude_output",
    {
      title: "Capture Claude Code Output",
      description:
        "Read recent visible output from an allowlisted tmux target running Claude Code. This is read-only, bounded, and does not infer Claude Code state.",
      inputSchema: {
        target: z
          .string()
          .min(1)
          .describe(
            "Allowlisted Claude Code target name from CODEXPRO_CLAUDE_TARGETS, not an arbitrary tmux target."
          ),
        lines: z
          .number()
          .int()
          .min(1)
          .max(5000)
          .optional()
          .describe(
            "Number of recent tmux pane lines to capture. Default: 80. The server also clamps this with CODEXPRO_CLAUDE_CAPTURE_MAX_LINES."
          )
      },
      annotations: READ_ONLY_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Reading Claude Code output...",
        "openai/toolInvocation/invoked": "Claude Code output captured"
      }
    },
    async (args) => {
      const requestedLines = args.lines === undefined ? undefined : Number(args.lines);
  
      const result = await captureClaudeCodeOutput({
        target: String(args.target ?? ""),
        lines: Number.isFinite(requestedLines) ? requestedLines : undefined
      });
  
      const output = result.output || "(no output captured)";
      const fence = codeFenceFor(output);
  
      const text = [
        "# Claude Code Output",
        "",
        `Target: ${result.target}`,
        `tmux target: ${result.tmuxTarget}`,
        `Lines requested: ${result.lines}`,
        `Bytes returned: ${result.bytes}`,
        `Total bytes captured before truncation: ${result.totalBytes}`,
        `Truncated: ${result.truncated ? "yes" : "no"}`,
        "",
        result.truncated
          ? "Output was truncated by CODEXPRO_CLAUDE_CAPTURE_MAX_BYTES."
          : "Output is bounded to the requested recent pane lines.",
        "",
        "## Output",
        "",
        `${fence}text`,
        output,
        fence
      ].join("\n");
  
      return textResult(text, {
        ok: result.ok,
        target: result.target,
        tmux_target: result.tmuxTarget,
        lines: result.lines,
        output: result.output,
        bytes: result.bytes,
        total_bytes: result.totalBytes,
        truncated: result.truncated
      });
    }
  );


  registerCodexTool(
    config,
    server,
    "inspect_claude_code_status",
    {
      title: "Inspect Claude Code Status",
      description:
        "Read recent visible output from an allowlisted tmux target running Claude Code and return a heuristic status summary. This is read-only and must not be treated as full Claude state.",
      inputSchema: {
        target: z
          .string()
          .min(1)
          .describe(
            "Allowlisted Claude Code target name from CODEXPRO_CLAUDE_TARGETS, not an arbitrary tmux target."
          ),
        lines: z
          .number()
          .int()
          .min(1)
          .max(5000)
          .optional()
          .describe(
            "Number of recent tmux pane lines to inspect. Default: 80. The server also clamps this with CODEXPRO_CLAUDE_CAPTURE_MAX_LINES."
          )
      },
      annotations: READ_ONLY_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Inspecting Claude Code status...",
        "openai/toolInvocation/invoked": "Claude Code status inspected"
      }
    },
    async (args) => {
      const requestedLines = args.lines === undefined ? undefined : Number(args.lines);
  
      const result = await inspectClaudeCodeStatus({
        target: String(args.target ?? ""),
        lines: Number.isFinite(requestedLines) ? requestedLines : undefined
      });
  
      const output = result.output || "(no output captured)";
      const fence = codeFenceFor(output);
  
      const signals = result.signals.length
        ? result.signals
            .map((signal) =>
              [
                `- ${signal.kind}: ${signal.description}`,
                signal.evidence ? `  Evidence: ${signal.evidence}` : ""
              ]
                .filter(Boolean)
                .join("\n")
            )
            .join("\n")
        : "- No signals detected.";
  
      const text = [
        "# Claude Code Status",
        "",
        `Target: ${result.target}`,
        `tmux target: ${result.tmuxTarget}`,
        `Status: ${result.status}`,
        `Confidence: ${result.confidence}`,
        `Heuristic: ${result.heuristic ? "yes" : "no"}`,
        `Analyzed at: ${result.analyzedAt}`,
        `Lines inspected: ${result.lines}`,
        `Bytes returned: ${result.bytes}`,
        `Total bytes captured before truncation: ${result.totalBytes}`,
        `Truncated: ${result.truncated ? "yes" : "no"}`,
        "",
        "This status is a heuristic based only on recent tmux pane text. It is not Claude Code internal state.",
        "",
        "## Signals",
        "",
        signals,
        "",
        "## Recent Output",
        "",
        `${fence}text`,
        output,
        fence
      ].join("\n");
  
      return textResult(text, {
        ok: result.ok,
        target: result.target,
        tmux_target: result.tmuxTarget,
        status: result.status,
        confidence: result.confidence,
        heuristic: result.heuristic,
        analyzed_at: result.analyzedAt,
        signals: result.signals,
        lines: result.lines,
        output: result.output,
        bytes: result.bytes,
        total_bytes: result.totalBytes,
        truncated: result.truncated
      });
    }
  );
    
  registerCodexTool(
    config,
    server,
    "sync_claude_code_status",
    {
      title: "Sync Claude Code Status",
      description:
        "Inspect recent visible Claude Code output and write workspace status snapshots as Markdown, JSON, and JSONL history. This reads tmux output and writes only inside the workspace.",
      inputSchema: {
        workspace_id: z.string().optional().describe("Workspace id from open_workspace. Omit to use default workspace."),
        target: z
          .string()
          .min(1)
          .describe(
            "Allowlisted Claude Code target name from CODEXPRO_CLAUDE_TARGETS, not an arbitrary tmux target."
          ),
        lines: z
          .number()
          .int()
          .min(1)
          .max(5000)
          .optional()
          .describe(
            "Number of recent tmux pane lines to inspect. Default: 120. The server also clamps this with CODEXPRO_CLAUDE_CAPTURE_MAX_LINES."
          ),
        markdown_path: z
          .string()
          .optional()
          .describe("Workspace-relative Markdown snapshot path. Default: .ai-bridge/claude-status.md."),
        json_path: z
          .string()
          .optional()
          .describe("Workspace-relative JSON snapshot path. Default: .ai-bridge/claude-status.json."),
        history_path: z
          .string()
          .optional()
          .describe("Workspace-relative JSONL append-only history path. Default: .ai-bridge/claude-status-history.jsonl."),
        include_output: z
          .boolean()
          .optional()
          .describe("Include recent Claude Code output in the JSON snapshot. Default: true.")
      },
      annotations: HANDOFF_WRITE_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Syncing Claude Code status...",
        "openai/toolInvocation/invoked": "Claude Code status synced"
      }
    },
    async (args) => {
      const workspace = workspaces.getWorkspace(args.workspace_id);
  
      await ensureAiBridge(config, guard, workspace);
  
      const markdownPath = normalizeWorkspaceWritePath(config, args.markdown_path, "claude-status.md");
      const jsonPath = normalizeWorkspaceWritePath(config, args.json_path, "claude-status.json");
      const historyPath = normalizeWorkspaceWritePath(config, args.history_path, "claude-status-history.jsonl");
  
      const markdownResolved = guard.resolve(workspace, markdownPath, { forWrite: true });
      const jsonResolved = guard.resolve(workspace, jsonPath, { forWrite: true });
      const historyResolved = guard.resolve(workspace, historyPath, { forWrite: true });
  
      assertWriteToolAllowed(config, markdownResolved.relPath);
      assertWriteToolAllowed(config, jsonResolved.relPath);
      assertWriteToolAllowed(config, historyResolved.relPath);
  
      const requestedLines = args.lines === undefined ? 120 : Number(args.lines);
  
      const result = await inspectClaudeCodeStatus({
        target: String(args.target ?? ""),
        lines: Number.isFinite(requestedLines) ? requestedLines : 120
      });
  
      const includeOutput = parseBool(args.include_output, true);
  
      const markdown = buildClaudeStatusMarkdown(result);
      const structured = statusSyncStructured(result);
  
      if (!includeOutput) {
        delete structured.output;
      }
  
      const json = `${JSON.stringify(structured, null, 2)}\n`;
      const historyLine = buildClaudeStatusHistoryLine(result);
  
      const markdownWrite = await writeTextFile(config, guard, workspace, markdownResolved.relPath, markdown, {
        createDirs: true,
        overwrite: true
      });
  
      const jsonWrite = await writeTextFile(config, guard, workspace, jsonResolved.relPath, json, {
        createDirs: true,
        overwrite: true
      });
  
      await fsp.mkdir(path.dirname(historyResolved.absPath), { recursive: true });
      await fsp.appendFile(historyResolved.absPath, historyLine, "utf8");
  
      const text = [
        "# Claude Code Status Synced",
        "",
        `Workspace: ${workspace.root}`,
        `Target: ${result.target}`,
        `tmux target: ${result.tmuxTarget}`,
        `Status: ${result.status}`,
        `Confidence: ${result.confidence}`,
        `Heuristic: ${result.heuristic ? "yes" : "no"}`,
        `Analyzed at: ${result.analyzedAt}`,
        "",
        "## Files",
        "",
        `- Markdown snapshot: ${markdownWrite.path}`,
        `- JSON snapshot: ${jsonWrite.path}`,
        `- JSONL history: ${historyResolved.relPath}`,
        "",
        "## Signals",
        "",
        result.signals.length
          ? result.signals
              .map((signal) =>
                [
                  `- ${signal.kind}: ${signal.description}`,
                  signal.evidence ? `  Evidence: ${signal.evidence}` : ""
                ]
                  .filter(Boolean)
                  .join("\n")
              )
              .join("\n")
          : "- No signals detected.",
        "",
        "The synced status is based only on recent tmux pane text. It is not Claude Code internal state."
      ].join("\n");
  
      return textResult(text, {
        workspace_id: workspace.id,
        root: workspace.root,
        ok: result.ok,
        target: result.target,
        tmux_target: result.tmuxTarget,
        status: result.status,
        confidence: result.confidence,
        heuristic: result.heuristic,
        analyzed_at: result.analyzedAt,
        signals: result.signals,
        lines: result.lines,
        bytes: result.bytes,
        total_bytes: result.totalBytes,
        truncated: result.truncated,
        include_output: includeOutput,
        markdown_path: markdownWrite.path,
        markdown_bytes: markdownWrite.bytes,
        markdown_sha256: markdownWrite.sha256,
        json_path: jsonWrite.path,
        json_bytes: jsonWrite.bytes,
        json_sha256: jsonWrite.sha256,
        history_path: historyResolved.relPath
      });
    }
  );

  registerCodexTool(
    config,
    server,
    "create_review_execution_task",
    {
      title: "Create Review Execution Task",
      description:
        "Convert WebGPT code review findings into .ai-bridge handoff artifacts for Claude Code. This only writes task files and returns a prompt; it does not send text, submit Enter, execute code, wait, loop, or confirm permissions.",
      inputSchema: {
        workspace_id: z.string().optional().describe("Workspace id from open_workspace. Omit to use default workspace."),
        target: z
          .string()
          .optional()
          .describe("Optional allowlisted Claude Code target name from CODEXPRO_CLAUDE_TARGETS."),
        title: z.string().min(1).max(160).describe("Short title for this review-to-execution task."),
        review_goal: z
          .string()
          .min(1)
          .max(8000)
          .describe("What WebGPT reviewed and what this modification round should accomplish."),
        reviewed_files: z
          .array(z.string().min(1).max(300))
          .max(200)
          .optional()
          .describe("Files WebGPT reviewed before producing the findings."),
        findings: z
          .array(
            z.object({
              file: z.string().max(300).optional().describe("Relevant file path, if known."),
              issue: z.string().min(1).max(4000).describe("Problem found during WebGPT review."),
              recommendation: z.string().min(1).max(4000).describe("Concrete change WebGPT recommends."),
              priority: z.enum(["low", "medium", "high"]).optional().describe("Finding priority. Default: medium."),
              risk: z.string().max(2000).optional().describe("Risk or caution for Claude Code.")
            })
          )
          .min(1)
          .max(80)
          .describe("Structured WebGPT review findings that Claude Code should implement."),
        allowed_files: z
          .array(z.string().min(1).max(300))
          .max(200)
          .optional()
          .describe("Optional file paths Claude Code is allowed to modify."),
        forbidden_actions: z
          .array(z.string().min(1).max(500))
          .max(200)
          .optional()
          .describe("Actions Claude Code must not perform."),
        test_instructions: z
          .array(z.string().min(1).max(500))
          .max(80)
          .optional()
          .describe("Verification commands or checks Claude Code should run when practical."),
        extra_context: z
          .string()
          .max(10000)
          .optional()
          .describe("Optional extra context to include in the handoff prompt.")
      },
      annotations: HANDOFF_WRITE_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Creating review execution task...",
        "openai/toolInvocation/invoked": "Review execution task created"
      }
    },
    async (args) => {
      const workspace = workspaces.getWorkspace(args.workspace_id);
  
      const result = await createReviewExecutionTask(config, guard, workspace, {
        target: typeof args.target === "string" ? args.target : undefined,
        title: String(args.title ?? ""),
        reviewGoal: String(args.review_goal ?? ""),
        reviewedFiles: Array.isArray(args.reviewed_files) ? args.reviewed_files : undefined,
        findings: Array.isArray(args.findings) ? args.findings : [],
        allowedFiles: Array.isArray(args.allowed_files) ? args.allowed_files : undefined,
        forbiddenActions: Array.isArray(args.forbidden_actions) ? args.forbidden_actions : undefined,
        testInstructions: Array.isArray(args.test_instructions) ? args.test_instructions : undefined,
        extraContext: typeof args.extra_context === "string" ? args.extra_context : undefined
      });
  
      const fence = codeFenceFor(result.promptForClaude);
  
      const text = [
        "# Review Execution Task",
        "",
        `Task ID: ${result.taskId}`,
        `Title: ${result.title}`,
        `Target: ${result.target ?? "(not selected)"}`,
        `Status: ${result.status}`,
        "",
        "## Files Written",
        "",
        `- ${result.files.reviewPlanMarkdown}`,
        `- ${result.files.executionTaskMarkdown}`,
        `- ${result.files.executionTaskJson}`,
        `- ${result.files.historyJsonl}`,
        "",
        "## Safety",
        "",
        "- User confirmation required: yes",
        "- Default submit: false",
        "- Auto-send: no",
        "- Auto-submit: no",
        "- Auto permission confirmation: no",
        "- Auto loop: no",
        "",
        "## Prompt For Claude Code",
        "",
        `${fence}text`,
        result.promptForClaude,
        fence,
        "",
        "Next step: review the prompt. If it is correct, call send_to_claude_code with submit=false, or paste it manually into the Claude Code terminal."
      ].join("\n");
  
      return textResult(text, {
        workspace_id: workspace.id,
        root: workspace.root,
        task_id: result.taskId,
        target: result.target,
        title: result.title,
        status: result.status,
        files: result.files,
        prompt_for_claude: result.promptForClaude,
        review_plan_markdown: result.reviewPlanMarkdown,
        execution_task_markdown: result.executionTaskMarkdown,
        json_payload: result.jsonPayload,
        writes: result.writes,
        safety: result.safety
      });
    }
  );
  
  if (config.codexSessions !== "off") {
    registerCodexTool(
      config,
      server,
      "codex_sessions",
      {
        title: "Codex Sessions",
        description:
          "Opt-in, read-only local Codex session history browser. Lists metadata from the user's configured Codex session JSONL files without reading full transcripts.",
        inputSchema: {
          max_sessions: z.number().int().min(1).max(200).optional().describe("Maximum sessions to return. Default: 30."),
          query: z.string().optional().describe("Optional case-insensitive search over session id, title, summary, cwd, and source path.")
        },
        annotations: READ_ONLY_ANNOTATIONS,
        _meta: {
          ...toolCardMeta(),
          "openai/toolInvocation/invoking": "Listing local Codex sessions...",
          "openai/toolInvocation/invoked": "Codex sessions ready"
        }
      },
      async (args) => {
        const result = await listCodexSessions(config, {
          maxSessions: args.max_sessions,
          query: args.query
        });
        const rows = result.sessions.length
          ? result.sessions.map((session) => `- ${session.session_id}  ${session.title || "(untitled)"}${session.project_dir ? `  cwd=${session.project_dir}` : ""}`).join("\n")
          : "- No Codex sessions found.";
        const text = `# Codex Sessions\n\nCodex dir: ${result.codex_dir}\nMode: ${config.codexSessions}\nTotal matched: ${result.total_found}\n\n${rows}`;
        return textResult(text, {
          codex_dir: result.codex_dir,
          roots: result.roots,
          sessions: result.sessions,
          total_found: result.total_found,
          codex_sessions_mode: config.codexSessions
        });
      }
    );

    if (config.codexSessions === "read") {
      registerCodexTool(
        config,
        server,
        "read_codex_session",
        {
          title: "Read Codex Session",
          description:
            "Opt-in, read-only local Codex transcript reader. Requires --codex-sessions read and returns a bounded transcript from a local Codex session JSONL file.",
          inputSchema: {
            session_id: z.string().optional().describe("Codex session id from codex_sessions."),
            source_path: z.string().optional().describe("Source path from codex_sessions. Must be inside the configured Codex session roots."),
            max_messages: z.number().int().min(1).max(400).optional().describe("Maximum transcript messages. Default: 80."),
            max_total_bytes: z.number().int().min(4000).max(400000).optional().describe("Maximum transcript content bytes. Default: 80000.")
          },
          annotations: READ_ONLY_ANNOTATIONS,
          _meta: {
            ...toolCardMeta(),
            "openai/toolInvocation/invoking": "Reading local Codex session...",
            "openai/toolInvocation/invoked": "Codex session read"
          }
        },
        async (args) => {
          const result = await readCodexSession(config, {
            sessionId: args.session_id,
            sourcePath: args.source_path,
            maxMessages: args.max_messages,
            maxTotalBytes: args.max_total_bytes
          });
          return textResult(result.text, {
            session: result.session,
            messages: result.messages,
            message_count: result.messages.length,
            truncated: result.truncated,
            codex_sessions_mode: config.codexSessions
          });
        }
      );
    }
  }

  registerCodexTool(
    config,
    server,
    "handoff_to_agent",
    {
      title: "Handoff To Agent",
      description:
        "Write .ai-bridge/current-plan.md for Codex, OpenCode, Pi, or another local implementation agent. This only creates handoff files; it does not execute local agent commands.",
      inputSchema: {
        workspace_id: z.string().optional().describe("Workspace id from open_workspace. Omit to use default workspace."),
        agent: z.string().optional().describe("Target agent id, for example codex, opencode, pi, or custom. Default: custom."),
        agent_name: z.string().optional().describe("Human-readable agent name for custom agents."),
        model: z.string().optional().describe("Optional model identifier to include in the handoff plan."),
        title: z.string().optional().describe("Short task title."),
        plan: z.string().describe("Detailed implementation plan for the local agent."),
        append: z.boolean().optional().describe("Append to existing current-plan.md instead of overwriting. Default: false.")
      },
      annotations: HANDOFF_WRITE_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Writing agent handoff plan...",
        "openai/toolInvocation/invoked": "Agent handoff plan written"
      }
    },
    async (args) => {
      const workspace = workspaces.getWorkspace(args.workspace_id);
      const result = await writeAgentHandoff(config, guard, workspace, {
        agent: args.agent ?? "custom",
        agentName: args.agent_name,
        model: args.model,
        title: cleanOneLine(args.title, "Agent implementation plan"),
        plan: String(args.plan ?? ""),
        append: parseBool(args.append, false),
        eventName: "handoff_to_agent"
      });

      const text = `# Handoff To Agent

Agent: ${result.agentName} (${result.agent})
${result.model ? `Model: ${result.model}\n` : ""}Wrote ${result.planPath}.
Status path: ${result.statusPath}
Diff path: ${result.diffPath}
Execution log: ${result.executionLogPath}
Diff stats: +${result.writeResult.diff.additions} -${result.writeResult.diff.deletions}

Agent prompt:

\`\`\`text
${result.prompt}
\`\`\`${diffBlock(result.writeResult.diff.diff)}`;
      return textResult(text, {
        workspace_id: workspace.id,
        root: workspace.root,
        agent: result.agent,
        agent_name: result.agentName,
        model: result.model,
        plan_path: result.planPath,
        status_path: result.statusPath,
        diff_path: result.diffPath,
        log_path: result.logPath,
        execution_log_path: result.executionLogPath,
        additions: result.writeResult.diff.additions,
        deletions: result.writeResult.diff.deletions,
        diff: result.writeResult.diff.diff
      });
    }
  );

  registerCodexTool(
    config,
    server,
    "handoff_to_codex",
    {
      title: "Handoff To Codex",
      description: "Compatibility wrapper for handoff_to_agent with agent=codex.",
      inputSchema: {
        workspace_id: z.string().optional().describe("Workspace id from open_workspace. Omit to use default workspace."),
        title: z.string().optional().describe("Short task title."),
        plan: z.string().describe("Detailed implementation plan for Codex."),
        append: z.boolean().optional().describe("Append to existing current-plan.md instead of overwriting. Default: false.")
      },
      annotations: HANDOFF_WRITE_ANNOTATIONS,
      _meta: {
        ...toolCardMeta(),
        "openai/toolInvocation/invoking": "Writing Codex handoff plan...",
        "openai/toolInvocation/invoked": "Codex handoff plan written"
      }
    },
    async (args) => {
      const workspace = workspaces.getWorkspace(args.workspace_id);
      const result = await writeAgentHandoff(config, guard, workspace, {
        agent: "codex",
        title: cleanOneLine(args.title, "Codex implementation plan"),
        plan: String(args.plan ?? ""),
        append: parseBool(args.append, false),
        eventName: "handoff_to_codex"
      });
      const text = `# Handoff To Codex

Wrote ${result.planPath}.
Status path: ${result.statusPath}
Diff path: ${result.diffPath}
Diff stats: +${result.writeResult.diff.additions} -${result.writeResult.diff.deletions}

Codex prompt:

\`\`\`text
${result.prompt}
\`\`\`${diffBlock(result.writeResult.diff.diff)}`;
      return textResult(text, {
        workspace_id: workspace.id,
        root: workspace.root,
        agent: result.agent,
        agent_name: result.agentName,
        plan_path: result.planPath,
        status_path: result.statusPath,
        diff_path: result.diffPath,
        log_path: result.logPath,
        execution_log_path: result.executionLogPath,
        additions: result.writeResult.diff.additions,
        deletions: result.writeResult.diff.deletions,
        diff: result.writeResult.diff.diff
      });
    }
  );

  return server;
}
