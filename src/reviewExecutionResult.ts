import fsp from "node:fs/promises";
import path from "node:path";

import type { CodexProConfig } from "./config.js";

import {
  CodexProError,
  PathGuard,
  type Workspace
} from "./guard.js";

import {
  inspectClaudeCodeStatus,
  listClaudeCodeTargets,
  type ClaudeCodeStatus,
  type ClaudeCodeStatusConfidence,
  type ClaudeCodeStatusSignal
} from "./claudeCodeBridge.js";

import {
  readStoredReviewExecutionTask,
  type ReviewExecutionFindingInput,
  type ReviewExecutionTaskLifecycleStatus
} from "./reviewExecutionTask.js";

export type ClaudeExecutionResultStatus =
  | "pending"
  | "completed"
  | "partially_completed"
  | "blocked"
  | "failed"
  | "needs_input"
  | "missing"
  | "invalid";

export type ClaudeExecutionTestStatus =
  | "passed"
  | "failed"
  | "not_run"
  | "unknown";

export type ReviewExecutionSuggestedAction =
  | "wait"
  | "manual_permission"
  | "answer_claude"
  | "read_changed_files"
  | "review_partial_result"
  | "inspect_failure";

export interface ClaudeExecutionChangedFile {
  path: string;
  changeType:
    | "created"
    | "modified"
    | "deleted"
    | "renamed"
    | "unknown";
  summary: string;
}

export interface ClaudeExecutionTestResult {
  command: string;
  status: ClaudeExecutionTestStatus;
  summary: string;
}

export interface ClaudeExecutionNeedsInput {
  question: string;
  context: string;
}

export interface InspectReviewExecutionResultInput {
  taskId: string;
  target?: string;
  lines?: number;
}

export interface InspectReviewExecutionResultResult {
  taskId: string;
  title: string;

  dispatchStatus: ReviewExecutionTaskLifecycleStatus;

  target: string | null;

  result: {
    path: string;
    markdownPath: string;

    status: ClaudeExecutionResultStatus;
    summary: string;

    changedFiles: ClaudeExecutionChangedFile[];
    changeSummary: string[];

    tests: ClaudeExecutionTestResult[];

    remainingIssues: string[];
    risks: string[];

    needsInput: ClaudeExecutionNeedsInput | null;

    startedAt: string | null;
    completedAt: string | null;
    updatedAt: string | null;

    validationErrors: string[];
  };

  claude: {
    checked: boolean;
    target: string | null;

    status:
      | ClaudeCodeStatus
      | "not_checked"
      | "unavailable";

    confidence: ClaudeCodeStatusConfidence | null;
    signals: ClaudeCodeStatusSignal[];

    output: string;
    truncated: boolean;

    analyzedAt: string | null;
    error: string | null;
  };

  verification: {
    readyForWebGPTReview: boolean;
    filesToRead: string[];
    outOfScopeFiles: string[];

    requiresUserAction: boolean;
    suggestedAction: ReviewExecutionSuggestedAction;

    reasons: string[];
    reviewChecklist: string[];
  };

  reviewContext: {
    reviewGoal: string;
    reviewedFiles: string[];
    findings: ReviewExecutionFindingInput[];
    allowedFiles: string[];
    forbiddenActions: string[];
    testInstructions: string[];
  };

  audit: {
    updated: boolean;
    error: string | null;
  };
}

interface ParsedExecutionResult {
  status: ClaudeExecutionResultStatus;
  summary: string;

  changedFiles: ClaudeExecutionChangedFile[];
  changeSummary: string[];

  tests: ClaudeExecutionTestResult[];

  remainingIssues: string[];
  risks: string[];

  needsInput: ClaudeExecutionNeedsInput | null;

  startedAt: string | null;
  completedAt: string | null;
  updatedAt: string | null;

  validationErrors: string[];
}

function objectValue(
  value: unknown
): Record<string, unknown> {
  if (
    !value ||
    typeof value !== "object" ||
    Array.isArray(value)
  ) {
    return {};
  }

  return value as Record<string, unknown>;
}

function errorMessage(error: unknown): string {
  const value =
    error instanceof Error
      ? error.message
      : String(error);

  return value
    .replace(/\0/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 2_000);
}

function optionalString(
  value: unknown,
  maxLength = 100_000
): string | null {
  if (value === undefined || value === null) {
    return null;
  }

  const text = String(value).trim();
  if (!text) return null;

  if (text.includes("\0")) {
    return null;
  }

  return text.slice(0, maxLength);
}

function stringList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];

  return [
    ...new Set(
      value
        .map((item) => {
          if (
            item &&
            typeof item === "object" &&
            !Array.isArray(item)
          ) {
            const record = item as Record<string, unknown>;

            return String(
              record.summary ??
                record.message ??
                record.issue ??
                ""
            ).trim();
          }

          return String(item ?? "").trim();
        })
        .filter(Boolean)
    )
  ];
}

function normalizeResultStatus(
  value: unknown
): ClaudeExecutionResultStatus | null {
  if (value === "pending") return "pending";
  if (value === "completed") return "completed";

  if (value === "partially_completed") {
    return "partially_completed";
  }

  if (value === "blocked") return "blocked";
  if (value === "failed") return "failed";
  if (value === "needs_input") return "needs_input";

  return null;
}

function normalizeTestStatus(
  value: unknown
): ClaudeExecutionTestStatus {
  if (value === "passed") return "passed";
  if (value === "failed") return "failed";
  if (value === "not_run") return "not_run";

  return "unknown";
}

function normalizeChangeType(
  value: unknown
): ClaudeExecutionChangedFile["changeType"] {
  if (value === "created") return "created";
  if (value === "modified") return "modified";
  if (value === "deleted") return "deleted";
  if (value === "renamed") return "renamed";

  return "unknown";
}

function normalizeWorkspaceRelativePath(
  guard: PathGuard,
  workspace: Workspace,
  value: unknown
): string | null {
  const raw = optionalString(value, 2_000);
  if (!raw) return null;

  const normalized = raw
    .replace(/\\/g, "/")
    .replace(/^\.\/+/, "")
    .replace(/\/+/g, "/");

  if (
    normalized === ".." ||
    normalized.startsWith("../") ||
    normalized.includes("/../") ||
    normalized.startsWith("/")
  ) {
    return null;
  }

  try {
    guard.resolve(workspace, normalized);
  } catch {
    return null;
  }

  return normalized;
}

function normalizeChangedFiles(
  guard: PathGuard,
  workspace: Workspace,
  value: unknown,
  validationErrors: string[]
): ClaudeExecutionChangedFile[] {
  if (!Array.isArray(value)) return [];

  const files: ClaudeExecutionChangedFile[] = [];
  const seen = new Set<string>();

  for (const item of value) {
    let rawPath: unknown;
    let changeType: unknown;
    let summary: unknown;

    if (typeof item === "string") {
      rawPath = item;
      changeType = "unknown";
      summary = "";
    } else if (
      item &&
      typeof item === "object" &&
      !Array.isArray(item)
    ) {
      const record = item as Record<string, unknown>;

      rawPath =
        record.path ??
        record.file ??
        record.file_path;

      changeType =
        record.change_type ??
        record.changeType;

      summary =
        record.summary ??
        record.description;
    } else {
      continue;
    }

    const filePath = normalizeWorkspaceRelativePath(
      guard,
      workspace,
      rawPath
    );

    if (!filePath) {
      validationErrors.push(
        `Ignored unsafe or invalid changed file path: ${String(
          rawPath ?? ""
        )}`
      );
      continue;
    }

    if (seen.has(filePath)) continue;
    seen.add(filePath);

    files.push({
      path: filePath,
      changeType: normalizeChangeType(changeType),
      summary: optionalString(summary, 8_000) ?? ""
    });
  }

  return files;
}

function normalizeTests(
  value: unknown
): ClaudeExecutionTestResult[] {
  if (!Array.isArray(value)) return [];

  const tests: ClaudeExecutionTestResult[] = [];

  for (const item of value) {
    if (typeof item === "string") {
      const command = item.trim();

      if (command) {
        tests.push({
          command,
          status: "unknown",
          summary: ""
        });
      }

      continue;
    }

    if (
      !item ||
      typeof item !== "object" ||
      Array.isArray(item)
    ) {
      continue;
    }

    const record = item as Record<string, unknown>;

    const command = String(
      record.command ??
        record.name ??
        ""
    ).trim();

    if (!command) continue;

    tests.push({
      command,
      status: normalizeTestStatus(record.status),
      summary:
        optionalString(
          record.summary ??
            record.output ??
            record.result,
          12_000
        ) ?? ""
    });
  }

  return tests;
}

function normalizeNeedsInput(
  value: unknown
): ClaudeExecutionNeedsInput | null {
  if (typeof value === "string") {
    const question = value.trim();

    return question
      ? {
          question,
          context: ""
        }
      : null;
  }

  if (
    !value ||
    typeof value !== "object" ||
    Array.isArray(value)
  ) {
    return null;
  }

  const record = value as Record<string, unknown>;

  const question =
    optionalString(
      record.question ??
        record.message,
      12_000
    ) ?? "";

  const context =
    optionalString(
      record.context ??
        record.reason,
      12_000
    ) ?? "";

  if (!question && !context) return null;

  return {
    question:
      question ||
      "Claude Code requires additional input.",
    context
  };
}

async function resultFileExists(
  guard: PathGuard,
  workspace: Workspace,
  resultPath: string
): Promise<boolean> {
  const resolved = guard.resolve(workspace, resultPath);

  try {
    const stat = await fsp.stat(resolved.absPath);
    return stat.isFile();
  } catch (error) {
    const code =
      error &&
      typeof error === "object" &&
      "code" in error
        ? String(
            (error as { code?: unknown }).code
          )
        : "";

    if (code === "ENOENT") return false;

    throw error;
  }
}

async function readExecutionResult(
  config: CodexProConfig,
  guard: PathGuard,
  workspace: Workspace,
  resultPath: string,
  expectedTaskId: string
): Promise<ParsedExecutionResult> {
  const exists = await resultFileExists(
    guard,
    workspace,
    resultPath
  );

  if (!exists) {
    return {
      status: "missing",
      summary: "",

      changedFiles: [],
      changeSummary: [],

      tests: [],
      remainingIssues: [],
      risks: [],

      needsInput: null,

      startedAt: null,
      completedAt: null,
      updatedAt: null,

      validationErrors: [
        `Execution result file does not exist: ${resultPath}`
      ]
    };
  }

  const resolved = guard.resolve(
    workspace,
    resultPath
  );

  await guard.assertTextFile(
    resolved.absPath,
    Math.min(config.maxReadBytes, 1_000_000)
  );

  const raw = await fsp.readFile(
    resolved.absPath,
    "utf8"
  );

  let parsed: unknown;

  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    return {
      status: "invalid",
      summary: "",

      changedFiles: [],
      changeSummary: [],

      tests: [],
      remainingIssues: [],
      risks: [],

      needsInput: null,

      startedAt: null,
      completedAt: null,
      updatedAt: null,

      validationErrors: [
        `Invalid execution result JSON: ${errorMessage(
          error
        )}`
      ]
    };
  }

  const payload = objectValue(parsed);
  const validationErrors: string[] = [];

  if (payload.kind !== "claude_execution_result") {
    validationErrors.push(
      "Result kind must be claude_execution_result."
    );
  }

  const schemaVersion = Number(
    payload.schema_version ?? 1
  );

  if (
    !Number.isInteger(schemaVersion) ||
    schemaVersion !== 1
  ) {
    validationErrors.push(
      `Unsupported result schema_version: ${String(
        payload.schema_version
      )}`
    );
  }

  const resultTaskId = String(
    payload.task_id ?? ""
  ).trim();

  if (!resultTaskId) {
    validationErrors.push(
      "Result task_id is missing."
    );
  } else if (resultTaskId !== expectedTaskId) {
    throw new CodexProError(
      `Execution result task_id does not match the current task. ` +
        `Expected ${expectedTaskId}, received ${resultTaskId}.`
    );
  }

  const normalizedStatus =
    normalizeResultStatus(payload.status);

  if (!normalizedStatus) {
    validationErrors.push(
      `Unsupported execution result status: ${String(
        payload.status
      )}`
    );
  }

  const changedFiles = normalizeChangedFiles(
    guard,
    workspace,
    payload.changed_files,
    validationErrors
  );

  const parsedStatus =
    validationErrors.length > 0 &&
    !normalizedStatus
      ? "invalid"
      : normalizedStatus ?? "invalid";

  return {
    status: parsedStatus,

    summary:
      optionalString(payload.summary, 40_000) ?? "",

    changedFiles,

    changeSummary: stringList(
      payload.change_summary
    ),

    tests: normalizeTests(payload.tests),

    remainingIssues: stringList(
      payload.remaining_issues
    ),

    risks: stringList(payload.risks),

    needsInput: normalizeNeedsInput(
      payload.needs_input
    ),

    startedAt: optionalString(
      payload.started_at,
      200
    ),

    completedAt: optionalString(
      payload.completed_at,
      200
    ),

    updatedAt: optionalString(
      payload.updated_at,
      200
    ),

    validationErrors
  };
}

function resolveOptionalTarget(
  requestedTarget: string | undefined,
  storedTarget: string | null
): string | null {
  const candidate =
    requestedTarget?.trim() ||
    storedTarget?.trim() ||
    "";

  if (!candidate) return null;

  const allowed = listClaudeCodeTargets();
  const matched = allowed.find(
    (item) => item.name === candidate
  );

  if (!matched) {
    const available =
      allowed
        .map((item) => item.name)
        .join(", ") || "none";

    throw new CodexProError(
      `Claude Code target is not allowlisted: ${candidate}. ` +
        `Allowed targets: ${available}`
    );
  }

  return matched.name;
}

function normalizedScopePath(value: string): string {
  return value
    .replace(/\\/g, "/")
    .replace(/^\.\/+/, "")
    .replace(/\/+$/, "");
}

function pathAllowed(
  changedFile: string,
  allowedFiles: string[]
): boolean {
  if (!allowedFiles.length) return true;

  const changed = normalizedScopePath(changedFile);

  return allowedFiles.some((item) => {
    const allowed = normalizedScopePath(item);

    return (
      changed === allowed ||
      changed.startsWith(`${allowed}/`)
    );
  });
}

function buildVerificationDecision(input: {
  dispatchStatus: ReviewExecutionTaskLifecycleStatus;

  resultStatus: ClaudeExecutionResultStatus;
  changedFiles: ClaudeExecutionChangedFile[];
  validationErrors: string[];

  claudeStatus:
    | ClaudeCodeStatus
    | "not_checked"
    | "unavailable";
}): {
  readyForWebGPTReview: boolean;
  requiresUserAction: boolean;
  suggestedAction: ReviewExecutionSuggestedAction;
  reasons: string[];
} {
  const reasons: string[] = [];

  if (input.dispatchStatus === "draft") {
    reasons.push(
      "The task lifecycle is still draft; CodexPro has no recorded dispatch."
    );
  }

  if (input.claudeStatus === "waiting_permission") {
    return {
      readyForWebGPTReview: false,
      requiresUserAction: true,
      suggestedAction: "manual_permission",
      reasons: [
        ...reasons,
        "Claude Code appears to be waiting for a permission decision.",
        "Permission must be handled manually in the Claude Code terminal."
      ]
    };
  }

  if (
    input.claudeStatus === "waiting_input" ||
    input.resultStatus === "needs_input"
  ) {
    return {
      readyForWebGPTReview: false,
      requiresUserAction: true,
      suggestedAction: "answer_claude",
      reasons: [
        ...reasons,
        "Claude Code or the result report indicates that additional input is required."
      ]
    };
  }

  if (input.resultStatus === "completed") {
    if (!input.changedFiles.length) {
      return {
        readyForWebGPTReview: false,
        requiresUserAction: false,
        suggestedAction: "inspect_failure",
        reasons: [
          ...reasons,
          "The result says completed but does not list any changed files.",
          "Inspect the result and recent Claude Code output before accepting it."
        ]
      };
    }

    return {
      readyForWebGPTReview: true,
      requiresUserAction: false,
      suggestedAction: "read_changed_files",
      reasons: [
        ...reasons,
        "Claude Code reported completion and supplied changed files.",
        "WebGPT must now read the actual files and independently verify the implementation."
      ]
    };
  }

  if (
    input.resultStatus === "partially_completed"
  ) {
    return {
      readyForWebGPTReview:
        input.changedFiles.length > 0,

      requiresUserAction: false,

      suggestedAction: "review_partial_result",

      reasons: [
        ...reasons,
        "Claude Code reported partial completion.",
        input.changedFiles.length
          ? "WebGPT should inspect the changed files and compare them with the remaining issues."
          : "No changed files were reported, so the partial result cannot yet be independently reviewed."
      ]
    };
  }

  if (
    input.resultStatus === "blocked" ||
    input.resultStatus === "failed"
  ) {
    return {
      readyForWebGPTReview:
        input.changedFiles.length > 0,

      requiresUserAction: true,

      suggestedAction: "inspect_failure",

      reasons: [
        ...reasons,
        `Claude Code reported ${input.resultStatus}.`,
        input.changedFiles.length
          ? "Inspect the partial code changes before deciding whether to create a repair task."
          : "Review the failure or blocker before sending another task."
      ]
    };
  }

  if (
    input.resultStatus === "missing" ||
    input.resultStatus === "invalid"
  ) {
    return {
      readyForWebGPTReview: false,
      requiresUserAction: false,
      suggestedAction: "inspect_failure",
      reasons: [
        ...reasons,
        ...input.validationErrors,
        "The structured execution result cannot be trusted."
      ]
    };
  }

  if (input.resultStatus === "pending") {
    if (input.claudeStatus === "likely_running") {
      return {
        readyForWebGPTReview: false,
        requiresUserAction: false,
        suggestedAction: "wait",
        reasons: [
          ...reasons,
          "The result is still pending and Claude Code appears to be running."
        ]
      };
    }

    if (input.claudeStatus === "likely_done") {
      return {
        readyForWebGPTReview: false,
        requiresUserAction: false,
        suggestedAction: "inspect_failure",
        reasons: [
          ...reasons,
          "Claude Code appears to have stopped, but the structured result remains pending.",
          "Check recent output and confirm that the result-reporting Skill ran successfully."
        ]
      };
    }

    return {
      readyForWebGPTReview: false,
      requiresUserAction: false,
      suggestedAction: "wait",
      reasons: [
        ...reasons,
        "The structured result is still pending."
      ]
    };
  }

  return {
    readyForWebGPTReview: false,
    requiresUserAction: false,
    suggestedAction: "inspect_failure",
    reasons: [
      ...reasons,
      "The execution state could not be classified."
    ]
  };
}

function historyEventForStatus(
  status: ClaudeExecutionResultStatus,
  readyForWebGPTReview: boolean
): string {
  if (readyForWebGPTReview) {
    return "webgpt_review_ready";
  }

  if (status === "completed") {
    return "execution_result_completed";
  }

  if (status === "partially_completed") {
    return "execution_result_partial";
  }

  if (status === "blocked") {
    return "execution_result_blocked";
  }

  if (status === "failed") {
    return "execution_result_failed";
  }

  if (status === "invalid") {
    return "execution_result_invalid";
  }

  return "execution_result_pending";
}

async function appendInspectionHistory(
  config: CodexProConfig,
  guard: PathGuard,
  workspace: Workspace,
  historyPath: string,
  event: string,
  data: Record<string, unknown>
): Promise<{
  updated: boolean;
  error: string | null;
}> {
  if (config.writeMode === "off") {
    return {
      updated: false,
      error:
        "Audit history was not written because write mode is off."
    };
  }

  try {
    const resolved = guard.resolve(
      workspace,
      historyPath,
      {
        forWrite: true
      }
    );

    await fsp.mkdir(
      path.dirname(resolved.absPath),
      {
        recursive: true
      }
    );

    await fsp.appendFile(
      resolved.absPath,
      `${JSON.stringify({
        ts: new Date().toISOString(),
        event,
        ...data
      })}\n`,
      "utf8"
    );

    return {
      updated: true,
      error: null
    };
  } catch (error) {
    return {
      updated: false,
      error: errorMessage(error)
    };
  }
}

export async function inspectReviewExecutionResult(
  config: CodexProConfig,
  guard: PathGuard,
  workspace: Workspace,
  rawInput: InspectReviewExecutionResultInput
): Promise<InspectReviewExecutionResultResult> {
  const task = await readStoredReviewExecutionTask(
    config,
    guard,
    workspace
  );

  const requestedTaskId = String(
    rawInput.taskId ?? ""
  ).trim();

  if (!requestedTaskId) {
    throw new CodexProError(
      "taskId must not be empty."
    );
  }

  if (requestedTaskId !== task.taskId) {
    throw new CodexProError(
      `taskId does not match the current task. ` +
        `Expected ${task.taskId}, received ${requestedTaskId}.`
    );
  }

  const parsedResult = await readExecutionResult(
    config,
    guard,
    workspace,
    task.resultJsonPath,
    task.taskId
  );

  const target = resolveOptionalTarget(
    rawInput.target,
    task.target
  );

  let claudeChecked = false;

  let claudeStatus:
    | ClaudeCodeStatus
    | "not_checked"
    | "unavailable" = "not_checked";

  let claudeConfidence:
    | ClaudeCodeStatusConfidence
    | null = null;

  let claudeSignals:
    ClaudeCodeStatusSignal[] = [];

  let claudeOutput = "";
  let claudeTruncated = false;
  let claudeAnalyzedAt: string | null = null;
  let claudeError: string | null = null;

  if (target) {
    try {
      const inspected =
        await inspectClaudeCodeStatus({
          target,
          lines: rawInput.lines
        });

      claudeChecked = true;
      claudeStatus = inspected.status;
      claudeConfidence = inspected.confidence;
      claudeSignals = inspected.signals;
      claudeOutput = inspected.output;
      claudeTruncated = inspected.truncated;
      claudeAnalyzedAt = inspected.analyzedAt;
    } catch (error) {
      claudeChecked = true;
      claudeStatus = "unavailable";
      claudeError = errorMessage(error);
    }
  }

  const filesToRead =
    parsedResult.changedFiles.map(
      (item) => item.path
    );

  const outOfScopeFiles =
    parsedResult.changedFiles
      .filter(
        (item) =>
          !pathAllowed(
            item.path,
            task.allowedFiles
          )
      )
      .map((item) => item.path);

  const decision = buildVerificationDecision({
    dispatchStatus: task.status,

    resultStatus: parsedResult.status,
    changedFiles: parsedResult.changedFiles,
    validationErrors:
      parsedResult.validationErrors,

    claudeStatus
  });

  const reviewChecklist = [
    "Read every path in filesToRead using the workspace read tool.",
    "Compare the actual implementation with the original review findings.",
    "Check whether any changed file is outside allowedFiles.",
    "Do not treat Claude Code's result report as proof that the implementation is correct.",
    "Check build, test, lint, or typecheck evidence against the actual project scripts.",
    "Inspect remainingIssues and risks before declaring the task accepted.",
    "Do not automatically dispatch a follow-up task; present the review conclusion to the user first."
  ];

  const auditEvent = historyEventForStatus(
    parsedResult.status,
    decision.readyForWebGPTReview
  );

  const audit =
    await appendInspectionHistory(
      config,
      guard,
      workspace,
      task.historyPath,
      auditEvent,
      {
        task_id: task.taskId,

        dispatch_status: task.status,

        result_status: parsedResult.status,

        claude_status: claudeStatus,
        claude_confidence: claudeConfidence,

        ready_for_webgpt_review:
          decision.readyForWebGPTReview,

        changed_files: filesToRead,
        out_of_scope_files: outOfScopeFiles,

        suggested_action:
          decision.suggestedAction
      }
    );

  return {
    taskId: task.taskId,
    title: task.title,

    dispatchStatus: task.status,

    target,

    result: {
      path: task.resultJsonPath,
      markdownPath: task.resultMarkdownPath,

      status: parsedResult.status,
      summary: parsedResult.summary,

      changedFiles:
        parsedResult.changedFiles,

      changeSummary:
        parsedResult.changeSummary,

      tests: parsedResult.tests,

      remainingIssues:
        parsedResult.remainingIssues,

      risks: parsedResult.risks,

      needsInput:
        parsedResult.needsInput,

      startedAt: parsedResult.startedAt,
      completedAt: parsedResult.completedAt,
      updatedAt: parsedResult.updatedAt,

      validationErrors:
        parsedResult.validationErrors
    },

    claude: {
      checked: claudeChecked,
      target,

      status: claudeStatus,
      confidence: claudeConfidence,
      signals: claudeSignals,

      output: claudeOutput,
      truncated: claudeTruncated,

      analyzedAt: claudeAnalyzedAt,
      error: claudeError
    },

    verification: {
      readyForWebGPTReview:
        decision.readyForWebGPTReview,

      filesToRead,
      outOfScopeFiles,

      requiresUserAction:
        decision.requiresUserAction,

      suggestedAction:
        decision.suggestedAction,

      reasons: decision.reasons,
      reviewChecklist
    },

    reviewContext: {
      reviewGoal: task.reviewGoal,
      reviewedFiles: task.reviewedFiles,
      findings: task.findings,
      allowedFiles: task.allowedFiles,
      forbiddenActions:
        task.forbiddenActions,
      testInstructions:
        task.testInstructions
    },

    audit
  };
}
