import fsp from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";

import type { CodexProConfig } from "./config.js";
import {
  CodexProError,
  PathGuard,
  type Workspace
} from "./guard.js";
import { ensureAiBridge } from "./fsOps.js";
import { listClaudeCodeTargets } from "./claudeCodeBridge.js";
import {
  createReviewExecutionTask,
  readStoredReviewExecutionTask,
  type CreateReviewExecutionTaskResult,
  type ReviewExecutionFindingInput,
  type ReviewExecutionPromptMode,
  type ReviewExecutionTaskFileForClaude,
  type ReviewFindingPriority,
  type ReviewTaskLineage,
  type StoredReviewExecutionTask
} from "./reviewExecutionTask.js";
import { readExecutionResult } from "./reviewExecutionResult.js";
import {
  readStoredReviewVerification,
  type StoredReviewVerification
} from "./reviewVerification.js";

export type RevisionSourceType =
  | "finding"
  | "test"
  | "remaining_issue"
  | "risk"
  | "file_scope"
  | "forbidden_action"
  | "manual";

export interface RevisionSourceReferenceInput {
  sourceType: RevisionSourceType;
  sourceIndex?: number;
  sourcePath?: string;
  sourceCode?: string;
}

export interface RevisionFindingInput
  extends RevisionSourceReferenceInput,
    ReviewExecutionFindingInput {}

export interface DeferredRevisionItemInput
  extends RevisionSourceReferenceInput {
  reason: string;
}

export interface CreateReviewRevisionTaskInput {
  sourceTaskId: string;
  sourceVerificationSignature: string;

  title: string;
  revisionGoal: string;
  findings: RevisionFindingInput[];

  reviewedFiles?: string[];
  allowedFiles?: string[];
  forbiddenActions?: string[];
  testInstructions?: string[];
  deferredSourceItems?: DeferredRevisionItemInput[];
  extraContext?: string;

  target?: string;
  promptMode?: ReviewExecutionPromptMode;
  claudeSkill?: string;
  taskFileForClaude?: ReviewExecutionTaskFileForClaude;

  confirmed?: boolean;
}

export interface RevisionSourceItem {
  key: string;
  sourceType: RevisionSourceType;
  sourceIndex: number | null;
  sourcePath: string | null;
  sourceCode: string | null;
  label: string;
  details: string;
}

export interface ArchiveFileResult {
  name: string;
  sourcePath: string;
  archivePath: string;
  required: boolean;
  status:
    | "would_archive"
    | "archived"
    | "already_archived"
    | "missing_optional";
  bytes: number | null;
  sha256: string | null;
}

export interface CreateReviewRevisionTaskResult {
  sourceTaskId: string;
  sourceVerificationSignature: string;
  sourceExecutionResultDigest: string;
  requestSignature: string;

  newTaskId: string;
  rootTaskId: string;
  parentTaskId: string;
  revisionNumber: number;

  confirmed: boolean;
  created: boolean;
  idempotent: boolean;
  canCreate: boolean;

  coveredSourceItems: RevisionSourceItem[];
  deferredSourceItems: Array<RevisionSourceItem & { reason: string }>;
  unaccountedSourceItems: RevisionSourceItem[];

  archive: {
    directory: string;
    manifestPath: string;
    files: ArchiveFileResult[];
    warnings: string[];
  };

  task: CreateReviewExecutionTaskResult | null;

  audit: {
    updated: boolean;
    error: string | null;
  };

  safety: {
    requiredExplicitConfirmation: true;
    previewDefault: true;
    autoDispatchAllowed: false;
    autoSubmitAllowed: false;
    autoPermissionConfirmAllowed: false;
    automaticLoopAllowed: false;
    branchingAllowed: false;
  };
}

interface NormalizedSourceReference {
  sourceType: RevisionSourceType;
  sourceIndex: number | null;
  sourcePath: string | null;
  sourceCode: string | null;
  key: string;
}

interface NormalizedRevisionFinding {
  source: NormalizedSourceReference;
  file: string;
  issue: string;
  recommendation: string;
  priority: ReviewFindingPriority;
  risk: string;
}

interface NormalizedDeferredItem {
  source: NormalizedSourceReference;
  reason: string;
}

interface RevisionManifest {
  schema_version: 1;
  kind: "review_revision_manifest";
  request_signature: string;
  source_task_id: string;
  source_verification_signature: string;
  source_execution_result_digest: string;
  new_task_id: string;
  root_task_id: string;
  parent_task_id: string;
  revision_number: number;
  archive_directory: string;
  created_at: string;
}

const MAX_TEXT_BYTES = 20_000;
const MAX_LIST_ITEMS = 300;
const MAX_FINDINGS = 100;

function errorMessage(error: unknown): string {
  return (error instanceof Error ? error.message : String(error))
    .replace(/\0/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 2_000);
}

function cleanText(
  value: unknown,
  fieldName: string,
  maxBytes = MAX_TEXT_BYTES
): string {
  const text = String(value ?? "")
    .replace(/\r\n/g, "\n")
    .trim();

  if (!text) {
    throw new CodexProError(`${fieldName} must not be empty.`);
  }

  if (text.includes("\0")) {
    throw new CodexProError(`${fieldName} contains a NUL byte.`);
  }

  if (Buffer.byteLength(text, "utf8") > maxBytes) {
    throw new CodexProError(
      `${fieldName} is too large. Limit: ${maxBytes} bytes.`
    );
  }

  return text;
}

function cleanOptionalText(
  value: unknown,
  fieldName: string,
  maxBytes = MAX_TEXT_BYTES
): string | null {
  if (value === undefined || value === null) return null;
  const text = String(value).replace(/\r\n/g, "\n").trim();
  if (!text) return null;
  return cleanText(text, fieldName, maxBytes);
}

function cleanOneLine(
  value: unknown,
  fieldName: string,
  maxBytes = 500
): string {
  const text = cleanText(value, fieldName, maxBytes)
    .replace(/\s+/g, " ")
    .trim();

  return text;
}

function normalizeTaskId(value: unknown, fieldName: string): string {
  const taskId = cleanOneLine(value, fieldName, 200);

  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$/.test(taskId)) {
    throw new CodexProError(
      `${fieldName} must use only letters, numbers, dots, underscores, and hyphens.`
    );
  }

  return taskId;
}

function normalizeDigest(value: unknown, fieldName: string): string {
  const digest = cleanOneLine(value, fieldName, 64).toLowerCase();
  if (!/^[a-f0-9]{64}$/.test(digest)) {
    throw new CodexProError(`${fieldName} must be a 64-character SHA-256 digest.`);
  }
  return digest;
}

function cleanStringList(
  value: unknown,
  fieldName: string,
  maxItems = MAX_LIST_ITEMS
): string[] {
  if (value === undefined || value === null) return [];
  if (!Array.isArray(value)) {
    throw new CodexProError(`${fieldName} must be an array.`);
  }
  if (value.length > maxItems) {
    throw new CodexProError(`${fieldName} has too many items.`);
  }

  const result: string[] = [];
  for (let index = 0; index < value.length; index += 1) {
    const item = cleanOptionalText(
      value[index],
      `${fieldName}[${index}]`,
      4_000
    );
    if (item) result.push(item.replace(/\s+/g, " "));
  }
  return [...new Set(result)];
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }

  if (Array.isArray(value)) {
    return `[${value.map(canonicalJson).join(",")}]`;
  }

  const record = value as Record<string, unknown>;
  return `{${Object.keys(record)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`)
    .join(",")}}`;
}

function sha256(value: unknown): string {
  return createHash("sha256")
    .update(canonicalJson(value), "utf8")
    .digest("hex");
}

function normalizePriority(value: unknown): ReviewFindingPriority {
  if (value === "low" || value === "high") return value;
  return "medium";
}

function normalizePromptMode(value: unknown): ReviewExecutionPromptMode {
  if (value === "inline" || value === "skill_file_reference") return value;
  return "file_reference";
}

function normalizeTaskFileForClaude(
  value: unknown
): ReviewExecutionTaskFileForClaude {
  return value === "review_plan" ? "review_plan" : "execution_task";
}

function normalizeSkill(value: unknown): string | null {
  const raw = cleanOptionalText(value, "claudeSkill", 128);
  if (!raw) return null;
  const skill = raw.startsWith("/") ? raw.slice(1).trim() : raw;
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(skill)) {
    throw new CodexProError(
      "claudeSkill must use only letters, numbers, underscores, and hyphens."
    );
  }
  return skill;
}

function normalizeTarget(value: unknown): string | undefined {
  const target = cleanOptionalText(value, "target", 80);
  if (!target) return undefined;
  const matched = listClaudeCodeTargets().find((item) => item.name === target);
  if (!matched) {
    const allowed = listClaudeCodeTargets().map((item) => item.name).join(", ") || "none";
    throw new CodexProError(
      `Claude Code target is not allowlisted: ${target}. Allowed targets: ${allowed}`
    );
  }
  return matched.name;
}

function normalizeWorkspacePath(
  guard: PathGuard,
  workspace: Workspace,
  value: unknown,
  fieldName: string
): string {
  const raw = cleanOneLine(value, fieldName, 2_000)
    .replace(/\\/g, "/")
    .replace(/^\.\/+/, "")
    .replace(/\/+/g, "/");

  if (
    raw === "." ||
    raw === ".." ||
    raw.startsWith("../") ||
    raw.includes("/../") ||
    raw.startsWith("/") ||
    /^[A-Za-z]:\//.test(raw)
  ) {
    throw new CodexProError(`${fieldName} must be a workspace-relative path.`);
  }

  guard.resolve(workspace, raw);
  return raw;
}

function sourceKey(input: {
  sourceType: RevisionSourceType;
  sourceIndex: number | null;
  sourcePath: string | null;
  sourceCode: string | null;
}): string {
  if (input.sourceType === "manual") {
    return `manual:${input.sourceCode ?? input.sourcePath ?? "item"}`;
  }

  if (
    input.sourceType === "finding" ||
    input.sourceType === "remaining_issue" ||
    input.sourceType === "risk" ||
    input.sourceType === "forbidden_action"
  ) {
    return `${input.sourceType}:${input.sourceIndex}`;
  }

  if (input.sourceType === "test") {
    if (input.sourceIndex !== null) {
      return `test:instruction:${input.sourceIndex}`;
    }
    return `test:command:${input.sourceCode ?? ""}`;
  }

  return `file_scope:${input.sourceCode ?? "unspecified"}:${input.sourcePath ?? ""}`;
}

function normalizeSourceReference(
  guard: PathGuard,
  workspace: Workspace,
  raw: RevisionSourceReferenceInput,
  fieldName: string
): NormalizedSourceReference {
  const allowedTypes: RevisionSourceType[] = [
    "finding",
    "test",
    "remaining_issue",
    "risk",
    "file_scope",
    "forbidden_action",
    "manual"
  ];

  if (!allowedTypes.includes(raw.sourceType)) {
    throw new CodexProError(`${fieldName}.sourceType is invalid.`);
  }

  const sourceIndex =
    raw.sourceIndex === undefined || raw.sourceIndex === null
      ? null
      : Number(raw.sourceIndex);

  if (
    sourceIndex !== null &&
    (!Number.isInteger(sourceIndex) || sourceIndex < 1 || sourceIndex > 100_000)
  ) {
    throw new CodexProError(`${fieldName}.sourceIndex must be a positive integer.`);
  }

  const sourcePath = raw.sourcePath
    ? normalizeWorkspacePath(
        guard,
        workspace,
        raw.sourcePath,
        `${fieldName}.sourcePath`
      )
    : null;

  const sourceCode = cleanOptionalText(
    raw.sourceCode,
    `${fieldName}.sourceCode`,
    500
  );

  if (
    ["finding", "remaining_issue", "risk", "forbidden_action"].includes(
      raw.sourceType
    ) &&
    sourceIndex === null
  ) {
    throw new CodexProError(
      `${fieldName}.sourceIndex is required for sourceType=${raw.sourceType}.`
    );
  }

  if (
    raw.sourceType === "test" &&
    sourceIndex === null &&
    !sourceCode
  ) {
    throw new CodexProError(
      `${fieldName} requires sourceIndex or sourceCode for sourceType=test.`
    );
  }

  if (raw.sourceType === "file_scope" && !sourcePath) {
    throw new CodexProError(
      `${fieldName}.sourcePath is required for sourceType=file_scope.`
    );
  }

  return {
    sourceType: raw.sourceType,
    sourceIndex,
    sourcePath,
    sourceCode,
    key: sourceKey({
      sourceType: raw.sourceType,
      sourceIndex,
      sourcePath,
      sourceCode
    })
  };
}

function normalizeFindings(
  guard: PathGuard,
  workspace: Workspace,
  value: unknown
): NormalizedRevisionFinding[] {
  if (!Array.isArray(value) || value.length < 1) {
    throw new CodexProError("findings must contain at least one revision finding.");
  }
  if (value.length > MAX_FINDINGS) {
    throw new CodexProError(`findings has too many items. Limit: ${MAX_FINDINGS}.`);
  }

  return value.map((item, index) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      throw new CodexProError(`findings[${index}] must be an object.`);
    }
    const record = item as RevisionFindingInput;
    return {
      source: normalizeSourceReference(
        guard,
        workspace,
        record,
        `findings[${index}]`
      ),
      file: record.file
        ? normalizeWorkspacePath(
            guard,
            workspace,
            record.file,
            `findings[${index}].file`
          )
        : "",
      issue: cleanText(record.issue, `findings[${index}].issue`, 4_000),
      recommendation: cleanText(
        record.recommendation,
        `findings[${index}].recommendation`,
        4_000
      ),
      priority: normalizePriority(record.priority),
      risk: cleanOptionalText(
        record.risk,
        `findings[${index}].risk`,
        2_000
      ) ?? ""
    };
  });
}

function normalizeDeferredItems(
  guard: PathGuard,
  workspace: Workspace,
  value: unknown
): NormalizedDeferredItem[] {
  if (value === undefined || value === null) return [];
  if (!Array.isArray(value)) {
    throw new CodexProError("deferredSourceItems must be an array.");
  }
  if (value.length > MAX_LIST_ITEMS) {
    throw new CodexProError("deferredSourceItems has too many items.");
  }

  return value.map((item, index) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      throw new CodexProError(`deferredSourceItems[${index}] must be an object.`);
    }
    const record = item as DeferredRevisionItemInput;
    return {
      source: normalizeSourceReference(
        guard,
        workspace,
        record,
        `deferredSourceItems[${index}]`
      ),
      reason: cleanText(
        record.reason,
        `deferredSourceItems[${index}].reason`,
        4_000
      )
    };
  });
}

function recordValue(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  return value as Record<string, unknown>;
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item ?? "").trim()).filter(Boolean);
}

function verificationSourceItems(
  verification: StoredReviewVerification
): RevisionSourceItem[] {
  const items: RevisionSourceItem[] = [];

  for (const record of verification.findingResults) {
    const status = String(record.status ?? "");
    if (status === "verified") continue;
    const index = Number(record.finding_index);
    const original = recordValue(record.original_finding);
    items.push({
      key: `finding:${index}`,
      sourceType: "finding",
      sourceIndex: index,
      sourcePath: original.file ? String(original.file) : null,
      sourceCode: status,
      label: `Finding ${index} (${status})`,
      details: String(record.summary ?? original.issue ?? "")
    });
  }

  for (const record of verification.testEvidence) {
    const status = String(record.status ?? "");
    if (status === "passed") continue;
    const instructionIndex = Number(record.instruction_index);
    const hasIndex = Number.isInteger(instructionIndex) && instructionIndex > 0;
    const command = String(record.command ?? "").trim();
    items.push({
      key: hasIndex
        ? `test:instruction:${instructionIndex}`
        : `test:command:${command}`,
      sourceType: "test",
      sourceIndex: hasIndex ? instructionIndex : null,
      sourcePath: null,
      sourceCode: hasIndex ? status : command,
      label: hasIndex
        ? `Test instruction ${instructionIndex} (${status})`
        : `Test command ${command || "(unspecified)"} (${status})`,
      details: String(record.summary ?? "")
    });
  }

  for (const record of verification.remainingIssueResults) {
    const status = String(record.status ?? "");
    if (status === "resolved") continue;
    const index = Number(record.issue_index);
    items.push({
      key: `remaining_issue:${index}`,
      sourceType: "remaining_issue",
      sourceIndex: index,
      sourcePath: null,
      sourceCode: status,
      label: `Remaining issue ${index} (${status})`,
      details: String(record.summary ?? record.issue ?? "")
    });
  }

  for (const record of verification.riskResults) {
    const status = String(record.status ?? "");
    if (status === "accepted" || status === "resolved") continue;
    const index = Number(record.risk_index);
    items.push({
      key: `risk:${index}`,
      sourceType: "risk",
      sourceIndex: index,
      sourcePath: null,
      sourceCode: status,
      label: `Risk ${index} (${status})`,
      details: String(record.summary ?? record.risk ?? "")
    });
  }

  const fileReview = verification.fileReview;
  const fileCategories: Array<[string, string]> = [
    ["out_of_scope_files", "out_of_scope"],
    ["unreported_changed_files", "unreported_changed"],
    ["reported_but_not_observed_files", "reported_but_not_observed"],
    ["unreviewed_changed_files", "unreviewed_changed"]
  ];

  for (const [field, code] of fileCategories) {
    for (const filePath of stringArray(fileReview[field])) {
      items.push({
        key: `file_scope:${code}:${filePath}`,
        sourceType: "file_scope",
        sourceIndex: null,
        sourcePath: filePath,
        sourceCode: code,
        label: `File review ${code}: ${filePath}`,
        details: `Verification recorded ${filePath} in ${field}.`
      });
    }
  }

  const deletedResults = Array.isArray(fileReview.deleted_file_results)
    ? fileReview.deleted_file_results
    : [];
  for (const raw of deletedResults) {
    const record = recordValue(raw);
    const status = String(record.status ?? "");
    if (status === "expected") continue;
    const filePath = String(record.path ?? "").trim();
    if (!filePath) continue;
    const code = `deleted_${status || "unresolved"}`;
    items.push({
      key: `file_scope:${code}:${filePath}`,
      sourceType: "file_scope",
      sourceIndex: null,
      sourcePath: filePath,
      sourceCode: code,
      label: `Deleted file ${filePath} (${status})`,
      details: String(record.summary ?? "")
    });
  }

  verification.forbiddenActionViolations.forEach((violation, index) => {
    items.push({
      key: `forbidden_action:${index + 1}`,
      sourceType: "forbidden_action",
      sourceIndex: index + 1,
      sourcePath: null,
      sourceCode: null,
      label: `Forbidden action violation ${index + 1}`,
      details: violation
    });
  });

  const unique = new Map<string, RevisionSourceItem>();
  for (const item of items) unique.set(item.key, item);
  return [...unique.values()];
}

function assertSourceReferences(
  sourceItems: RevisionSourceItem[],
  findings: NormalizedRevisionFinding[],
  deferred: NormalizedDeferredItem[]
): {
  covered: RevisionSourceItem[];
  deferred: Array<RevisionSourceItem & { reason: string }>;
  unaccounted: RevisionSourceItem[];
} {
  const byKey = new Map(sourceItems.map((item) => [item.key, item]));
  const used = new Set<string>();
  const covered: RevisionSourceItem[] = [];

  for (const finding of findings) {
    if (finding.source.sourceType === "manual") continue;
    const item = byKey.get(finding.source.key);
    if (!item) {
      throw new CodexProError(
        `Revision finding references a source item that is not unresolved in the current Verification: ${finding.source.key}`
      );
    }
    if (used.has(item.key)) {
      throw new CodexProError(`Revision source item is referenced more than once: ${item.key}`);
    }
    used.add(item.key);
    covered.push(item);
  }

  const deferredItems: Array<RevisionSourceItem & { reason: string }> = [];
  for (const deferredItem of deferred) {
    if (deferredItem.source.sourceType === "manual") {
      throw new CodexProError("Manual source items cannot be deferred.");
    }
    const item = byKey.get(deferredItem.source.key);
    if (!item) {
      throw new CodexProError(
        `Deferred item references a source item that is not unresolved in the current Verification: ${deferredItem.source.key}`
      );
    }
    if (used.has(item.key)) {
      throw new CodexProError(
        `Revision source item cannot be both covered and deferred: ${item.key}`
      );
    }
    used.add(item.key);
    deferredItems.push({ ...item, reason: deferredItem.reason });
  }

  return {
    covered,
    deferred: deferredItems,
    unaccounted: sourceItems.filter((item) => !used.has(item.key))
  };
}

function sourceReferenceMarkdown(source: NormalizedSourceReference): string {
  const parts = [`type=${source.sourceType}`];
  if (source.sourceIndex !== null) parts.push(`index=${source.sourceIndex}`);
  if (source.sourcePath) parts.push(`path=${source.sourcePath}`);
  if (source.sourceCode) parts.push(`code=${source.sourceCode}`);
  return parts.join(", ");
}

function buildRevisionExtraContext(input: {
  sourceTaskId: string;
  sourceVerification: StoredReviewVerification;
  findings: NormalizedRevisionFinding[];
  deferred: Array<RevisionSourceItem & { reason: string }>;
  extraContext: string | null;
}): string {
  const findingLines = input.findings.map(
    (finding, index) =>
      `${index + 1}. ${sourceReferenceMarkdown(finding.source)} -> ${finding.issue}`
  );
  const deferredLines = input.deferred.map(
    (item) => `- ${item.key}: ${item.reason}`
  );

  return [
    "## Revision Source",
    "",
    `Source Task ID: ${input.sourceTaskId}`,
    `Source Verification Signature: ${input.sourceVerification.verificationSignature}`,
    `Source Execution Result Digest: ${input.sourceVerification.executionResultDigest}`,
    `Source Verdict: ${input.sourceVerification.verdict}`,
    "",
    "## Revision Finding Source Map",
    "",
    ...(findingLines.length ? findingLines : ["No source mappings."]),
    "",
    "## Explicitly Deferred Source Items",
    "",
    ...(deferredLines.length ? deferredLines : ["- None."]),
    input.extraContext ? "" : "",
    input.extraContext ? "## Additional Context" : "",
    input.extraContext ? "" : "",
    input.extraContext ?? ""
  ]
    .filter((item, index, all) => item !== "" || all[index - 1] !== "")
    .join("\n")
    .trim();
}


function normalizeAiBridgePath(
  config: CodexProConfig,
  value: unknown,
  fieldName: string
): string {
  const raw = cleanOneLine(value, fieldName, 2_000)
    .replace(/\\/g, "/")
    .replace(/^\.\/+/, "")
    .replace(/\/+/g, "/");
  const contextDir = config.contextDir
    .replace(/\\/g, "/")
    .replace(/^\.\/+/, "")
    .replace(/\/+$/, "");

  if (
    raw === ".." ||
    raw.startsWith("../") ||
    raw.includes("/../") ||
    raw.startsWith("/") ||
    /^[A-Za-z]:\//.test(raw) ||
    (raw !== contextDir && !raw.startsWith(`${contextDir}/`))
  ) {
    throw new CodexProError(
      `${fieldName} must remain inside ${contextDir}/.`
    );
  }

  return raw;
}

function aiBridgePath(
  config: CodexProConfig,
  suffix: string
): string {
  return `${config.contextDir.replace(/\/$/, "")}/${suffix}`;
}

function taskFilePaths(
  config: CodexProConfig,
  task: StoredReviewExecutionTask
): Array<{
  name: string;
  sourcePath: string;
  archiveName: string;
  required: boolean;
}> {
  const files = recordValue(task.payload.files);
  const valueOr = (
    value: unknown,
    fallback: string,
    fieldName: string
  ): string => {
    const text = String(value ?? "").trim() || fallback;
    return normalizeAiBridgePath(config, text, fieldName);
  };

  return [
    {
      name: "review plan Markdown",
      sourcePath: valueOr(
        files.review_plan_markdown,
        aiBridgePath(config, "webgpt-review-plan.md"),
        "files.review_plan_markdown"
      ),
      archiveName: "review-plan.md",
      required: false
    },
    {
      name: "execution task Markdown",
      sourcePath: valueOr(
        files.execution_task_markdown,
        aiBridgePath(config, "claude-execution-task.md"),
        "files.execution_task_markdown"
      ),
      archiveName: "claude-execution-task.md",
      required: false
    },
    {
      name: "execution task JSON",
      sourcePath: normalizeAiBridgePath(
        config,
        task.jsonPath,
        "files.execution_task_json"
      ),
      archiveName: "claude-execution-task.json",
      required: true
    },
    {
      name: "execution result Markdown",
      sourcePath: normalizeAiBridgePath(
        config,
        task.resultMarkdownPath,
        "files.execution_result_markdown"
      ),
      archiveName: "claude-execution-result.md",
      required: false
    },
    {
      name: "execution result JSON",
      sourcePath: normalizeAiBridgePath(
        config,
        task.resultJsonPath,
        "files.execution_result_json"
      ),
      archiveName: "claude-execution-result.json",
      required: true
    },
    {
      name: "WebGPT verification Markdown",
      sourcePath: normalizeAiBridgePath(
        config,
        task.verificationMarkdownPath,
        "files.verification_result_markdown"
      ),
      archiveName: "webgpt-verification-result.md",
      required: false
    },
    {
      name: "WebGPT verification JSON",
      sourcePath: normalizeAiBridgePath(
        config,
        task.verificationJsonPath,
        "files.verification_result_json"
      ),
      archiveName: "webgpt-verification-result.json",
      required: true
    }
  ];
}

async function readOptionalText(
  config: CodexProConfig,
  guard: PathGuard,
  workspace: Workspace,
  filePath: string
): Promise<{ exists: boolean; text: string }> {
  const resolved = guard.resolve(workspace, filePath);
  try {
    const stat = await fsp.stat(resolved.absPath);
    if (!stat.isFile()) return { exists: false, text: "" };
  } catch (error) {
    const code =
      error && typeof error === "object" && "code" in error
        ? String((error as { code?: unknown }).code)
        : "";
    if (code === "ENOENT") return { exists: false, text: "" };
    throw error;
  }

  await guard.assertTextFile(
    resolved.absPath,
    Math.min(config.maxReadBytes, 1_000_000)
  );
  return {
    exists: true,
    text: await fsp.readFile(resolved.absPath, "utf8")
  };
}

async function archiveFiles(
  config: CodexProConfig,
  guard: PathGuard,
  workspace: Workspace,
  task: StoredReviewExecutionTask,
  directory: string,
  confirmed: boolean
): Promise<{ files: ArchiveFileResult[]; warnings: string[] }> {
  const results: ArchiveFileResult[] = [];
  const warnings: string[] = [];

  for (const entry of taskFilePaths(config, task)) {
    const archivePath = `${directory}/${entry.archiveName}`;
    const source = await readOptionalText(
      config,
      guard,
      workspace,
      entry.sourcePath
    );

    if (!source.exists) {
      if (entry.required) {
        throw new CodexProError(
          `Required source artifact is missing: ${entry.sourcePath}`
        );
      }
      warnings.push(`Optional source artifact is missing: ${entry.sourcePath}`);
      results.push({
        name: entry.name,
        sourcePath: entry.sourcePath,
        archivePath,
        required: false,
        status: "missing_optional",
        bytes: null,
        sha256: null
      });
      continue;
    }

    const digest = createHash("sha256").update(source.text, "utf8").digest("hex");
    const bytes = Buffer.byteLength(source.text, "utf8");

    if (!confirmed) {
      results.push({
        name: entry.name,
        sourcePath: entry.sourcePath,
        archivePath,
        required: entry.required,
        status: "would_archive",
        bytes,
        sha256: digest
      });
      continue;
    }

    const target = guard.resolve(workspace, archivePath, { forWrite: true });
    await fsp.mkdir(path.dirname(target.absPath), { recursive: true });

    try {
      await fsp.writeFile(target.absPath, source.text, {
        encoding: "utf8",
        flag: "wx"
      });
      results.push({
        name: entry.name,
        sourcePath: entry.sourcePath,
        archivePath,
        required: entry.required,
        status: "archived",
        bytes,
        sha256: digest
      });
    } catch (error) {
      const code =
        error && typeof error === "object" && "code" in error
          ? String((error as { code?: unknown }).code)
          : "";
      if (code !== "EEXIST") throw error;

      const existing = await readOptionalText(
        config,
        guard,
        workspace,
        archivePath
      );
      const existingDigest = createHash("sha256")
        .update(existing.text, "utf8")
        .digest("hex");
      if (!existing.exists || existingDigest !== digest) {
        throw new CodexProError(
          `Archive conflict: ${archivePath} already exists with different content.`
        );
      }
      results.push({
        name: entry.name,
        sourcePath: entry.sourcePath,
        archivePath,
        required: entry.required,
        status: "already_archived",
        bytes,
        sha256: digest
      });
    }
  }

  return { files: results, warnings };
}

async function readManifest(
  config: CodexProConfig,
  guard: PathGuard,
  workspace: Workspace,
  manifestPath: string
): Promise<RevisionManifest | null> {
  const result = await readOptionalText(config, guard, workspace, manifestPath);
  if (!result.exists) return null;

  try {
    const value = JSON.parse(result.text) as RevisionManifest;
    if (
      value.kind !== "review_revision_manifest" ||
      value.schema_version !== 1
    ) {
      throw new Error("unsupported manifest kind or schema");
    }
    return value;
  } catch (error) {
    throw new CodexProError(
      `Invalid revision manifest ${manifestPath}: ${errorMessage(error)}`
    );
  }
}

async function writeManifest(
  guard: PathGuard,
  workspace: Workspace,
  manifestPath: string,
  manifest: RevisionManifest
): Promise<void> {
  const resolved = guard.resolve(workspace, manifestPath, { forWrite: true });
  await fsp.mkdir(path.dirname(resolved.absPath), { recursive: true });
  const content = `${JSON.stringify(manifest, null, 2)}\n`;

  try {
    await fsp.writeFile(resolved.absPath, content, {
      encoding: "utf8",
      flag: "wx"
    });
  } catch (error) {
    const code =
      error && typeof error === "object" && "code" in error
        ? String((error as { code?: unknown }).code)
        : "";
    if (code !== "EEXIST") throw error;

    const existing = await fsp.readFile(resolved.absPath, "utf8");
    if (existing !== content) {
      throw new CodexProError(
        `Revision manifest conflict: ${manifestPath} already exists with different content.`
      );
    }
  }
}

async function appendRevisionHistory(
  guard: PathGuard,
  workspace: Workspace,
  historyPath: string,
  data: Record<string, unknown>
): Promise<{ updated: boolean; error: string | null }> {
  try {
    const resolved = guard.resolve(workspace, historyPath, { forWrite: true });
    await fsp.mkdir(path.dirname(resolved.absPath), { recursive: true });
    await fsp.appendFile(
      resolved.absPath,
      `${JSON.stringify({
        ts: new Date().toISOString(),
        event: "review_revision_task_created",
        ...data
      })}\n`,
      "utf8"
    );
    return { updated: true, error: null };
  } catch (error) {
    return { updated: false, error: errorMessage(error) };
  }
}

function baseSafety(): CreateReviewRevisionTaskResult["safety"] {
  return {
    requiredExplicitConfirmation: true,
    previewDefault: true,
    autoDispatchAllowed: false,
    autoSubmitAllowed: false,
    autoPermissionConfirmAllowed: false,
    automaticLoopAllowed: false,
    branchingAllowed: false
  };
}

export async function createReviewRevisionTask(
  config: CodexProConfig,
  guard: PathGuard,
  workspace: Workspace,
  rawInput: CreateReviewRevisionTaskInput
): Promise<CreateReviewRevisionTaskResult> {
  const sourceTaskId = normalizeTaskId(rawInput.sourceTaskId, "sourceTaskId");
  const sourceVerificationSignature = normalizeDigest(
    rawInput.sourceVerificationSignature,
    "sourceVerificationSignature"
  );
  const title = cleanOneLine(rawInput.title, "title", 160);
  const revisionGoal = cleanText(rawInput.revisionGoal, "revisionGoal", 8_000);
  const findings = normalizeFindings(guard, workspace, rawInput.findings);
  const deferred = normalizeDeferredItems(
    guard,
    workspace,
    rawInput.deferredSourceItems
  );
  const reviewedFilesProvided = rawInput.reviewedFiles !== undefined;
  const allowedFilesProvided = rawInput.allowedFiles !== undefined;
  const forbiddenActionsProvided = rawInput.forbiddenActions !== undefined;
  const testInstructionsProvided = rawInput.testInstructions !== undefined;

  const reviewedFiles = cleanStringList(rawInput.reviewedFiles, "reviewedFiles");
  const allowedFiles = cleanStringList(rawInput.allowedFiles, "allowedFiles");
  const forbiddenActions = cleanStringList(
    rawInput.forbiddenActions,
    "forbiddenActions"
  );
  const testInstructions = cleanStringList(
    rawInput.testInstructions,
    "testInstructions",
    100
  );
  const extraContext = cleanOptionalText(
    rawInput.extraContext,
    "extraContext",
    10_000
  );
  const target = normalizeTarget(rawInput.target);
  const promptMode = normalizePromptMode(rawInput.promptMode);
  const claudeSkill = normalizeSkill(rawInput.claudeSkill);
  const taskFileForClaude = normalizeTaskFileForClaude(
    rawInput.taskFileForClaude
  );

  if (promptMode === "skill_file_reference" && !claudeSkill) {
    throw new CodexProError(
      "claudeSkill is required when promptMode=skill_file_reference."
    );
  }

  const requestPayload = {
    source_task_id: sourceTaskId,
    source_verification_signature: sourceVerificationSignature,
    title,
    revision_goal: revisionGoal,
    findings: findings.map((item) => ({
      source: item.source,
      file: item.file,
      issue: item.issue,
      recommendation: item.recommendation,
      priority: item.priority,
      risk: item.risk
    })),
    deferred_source_items: deferred,
    reviewed_files: reviewedFilesProvided ? reviewedFiles : "__inherit__",
    allowed_files: allowedFilesProvided ? allowedFiles : "__inherit__",
    forbidden_actions: forbiddenActionsProvided
      ? forbiddenActions
      : "__inherit__",
    test_instructions: testInstructionsProvided
      ? testInstructions
      : "__inherit__",
    extra_context: extraContext,
    target: target ?? "__inherit__",
    prompt_mode: promptMode,
    claude_skill: claudeSkill,
    task_file_for_claude: taskFileForClaude
  };
  const requestSignature = sha256(requestPayload);

  const archiveDirectory = aiBridgePath(
    config,
    `review-rounds/${sourceTaskId}`
  );
  const manifestPath = `${archiveDirectory}/revision-manifest.json`;
  const existingManifest = await readManifest(
    config,
    guard,
    workspace,
    manifestPath
  );

  if (existingManifest) {
    if (
      existingManifest.source_verification_signature !==
        sourceVerificationSignature ||
      existingManifest.request_signature !== requestSignature
    ) {
      throw new CodexProError(
        "A different Revision Task has already been created from this Verification. Branching is not allowed."
      );
    }

    return {
      sourceTaskId,
      sourceVerificationSignature,
      sourceExecutionResultDigest:
        existingManifest.source_execution_result_digest,
      requestSignature,
      newTaskId: existingManifest.new_task_id,
      rootTaskId: existingManifest.root_task_id,
      parentTaskId: existingManifest.parent_task_id,
      revisionNumber: existingManifest.revision_number,
      confirmed: rawInput.confirmed === true,
      created: true,
      idempotent: true,
      canCreate: true,
      coveredSourceItems: [],
      deferredSourceItems: [],
      unaccountedSourceItems: [],
      archive: {
        directory: existingManifest.archive_directory,
        manifestPath,
        files: [],
        warnings: []
      },
      task: null,
      audit: { updated: false, error: null },
      safety: baseSafety()
    };
  }

  const task = await readStoredReviewExecutionTask(config, guard, workspace);

  if (task.taskId !== sourceTaskId) {
    const existingChildId =
      `review_rev${task.lineage.revisionNumber}_${requestSignature.slice(0, 12)}`;

    if (
      task.lineage.parentTaskId === sourceTaskId &&
      task.lineage.sourceVerificationSignature === sourceVerificationSignature &&
      task.taskId === existingChildId
    ) {
      return {
        sourceTaskId,
        sourceVerificationSignature,
        sourceExecutionResultDigest:
          task.lineage.sourceExecutionResultDigest ?? "",
        requestSignature,
        newTaskId: task.taskId,
        rootTaskId: task.lineage.rootTaskId,
        parentTaskId: sourceTaskId,
        revisionNumber: task.lineage.revisionNumber,
        confirmed: rawInput.confirmed === true,
        created: true,
        idempotent: true,
        canCreate: true,
        coveredSourceItems: [],
        deferredSourceItems: [],
        unaccountedSourceItems: [],
        archive: {
          directory: archiveDirectory,
          manifestPath,
          files: [],
          warnings: [
            "The child task already exists, but the revision manifest is missing."
          ]
        },
        task: null,
        audit: { updated: false, error: null },
        safety: baseSafety()
      };
    }

    throw new CodexProError(
      `sourceTaskId does not match the current task. Expected ${task.taskId}, received ${sourceTaskId}.`
    );
  }

  if (task.status === "draft") {
    throw new CodexProError(
      "Cannot create a Revision Task from a source task whose dispatch lifecycle is still draft."
    );
  }

  const effectiveTarget = target ?? task.target ?? undefined;
  const effectiveReviewedFiles = reviewedFilesProvided
    ? reviewedFiles
    : task.reviewedFiles;
  const effectiveAllowedFiles = allowedFilesProvided
    ? allowedFiles
    : task.allowedFiles;
  const effectiveForbiddenActions = forbiddenActionsProvided
    ? forbiddenActions
    : task.forbiddenActions;
  const effectiveTestInstructions = testInstructionsProvided
    ? testInstructions
    : task.testInstructions;

  const verification = await readStoredReviewVerification(
    config,
    guard,
    workspace,
    task.verificationJsonPath,
    task.taskId
  );

  if (verification.verificationSignature !== sourceVerificationSignature) {
    throw new CodexProError(
      `sourceVerificationSignature does not match the current Verification. Expected ${verification.verificationSignature}.`
    );
  }

  if (verification.verdict !== "revision_required") {
    throw new CodexProError(
      `Only verdict=revision_required can create a Revision Task. Current verdict: ${verification.verdict}.`
    );
  }

  const result = await readExecutionResult(
    config,
    guard,
    workspace,
    task.resultJsonPath,
    task.taskId
  );

  if (!result.digest) {
    throw new CodexProError(
      "The current Execution Result is missing or invalid and cannot be used for Revision creation."
    );
  }

  if (verification.executionResultDigest !== result.digest) {
    throw new CodexProError(
      "The recorded Verification is stale because its Execution Result digest no longer matches the current result."
    );
  }

  const sourceItems = verificationSourceItems(verification);
  const accounting = assertSourceReferences(sourceItems, findings, deferred);
  const canCreate = accounting.unaccounted.length === 0;
  const confirmed = rawInput.confirmed === true;

  if (confirmed && !canCreate) {
    throw new CodexProError(
      `Cannot create Revision Task because ${accounting.unaccounted.length} unresolved Verification source item(s) are neither covered nor explicitly deferred: ${accounting.unaccounted
        .map((item) => item.key)
        .join(", ")}`
    );
  }

  const revisionNumber = task.lineage.revisionNumber + 1;
  const lineage: ReviewTaskLineage = {
    rootTaskId: task.lineage.rootTaskId,
    parentTaskId: task.taskId,
    revisionNumber,
    sourceVerificationSignature,
    sourceExecutionResultDigest: result.digest,
    sourceVerdict: "revision_required"
  };
  const newTaskId = `review_rev${revisionNumber}_${requestSignature.slice(0, 12)}`;

  const archivePreview = await archiveFiles(
    config,
    guard,
    workspace,
    task,
    archiveDirectory,
    false
  );

  const base: CreateReviewRevisionTaskResult = {
    sourceTaskId,
    sourceVerificationSignature,
    sourceExecutionResultDigest: result.digest,
    requestSignature,
    newTaskId,
    rootTaskId: lineage.rootTaskId,
    parentTaskId: task.taskId,
    revisionNumber,
    confirmed,
    created: false,
    idempotent: false,
    canCreate,
    coveredSourceItems: accounting.covered,
    deferredSourceItems: accounting.deferred,
    unaccountedSourceItems: accounting.unaccounted,
    archive: {
      directory: archiveDirectory,
      manifestPath,
      files: archivePreview.files,
      warnings: archivePreview.warnings
    },
    task: null,
    audit: { updated: false, error: null },
    safety: baseSafety()
  };

  if (!confirmed) return base;

  if (config.writeMode === "off") {
    throw new CodexProError(
      "create_review_revision_task requires write mode handoff or workspace when confirmed=true. Preview remains available with confirmed=false."
    );
  }

  await ensureAiBridge(config, guard, workspace);

  const archived = await archiveFiles(
    config,
    guard,
    workspace,
    task,
    archiveDirectory,
    true
  );

  const revisionContext = buildRevisionExtraContext({
    sourceTaskId,
    sourceVerification: verification,
    findings,
    deferred: accounting.deferred,
    extraContext
  });

  const createdTask = await createReviewExecutionTask(
    config,
    guard,
    workspace,
    {
      taskId: newTaskId,
      lineage,
      target: effectiveTarget,
      title,
      reviewGoal: revisionGoal,
      reviewedFiles: effectiveReviewedFiles,
      findings: findings.map((item) => ({
        file: item.file || undefined,
        issue: item.issue,
        recommendation: item.recommendation,
        priority: item.priority,
        risk: item.risk || undefined
      })),
      allowedFiles: effectiveAllowedFiles,
      forbiddenActions: effectiveForbiddenActions,
      testInstructions: effectiveTestInstructions,
      extraContext: revisionContext,
      promptMode,
      claudeSkill: claudeSkill ?? undefined,
      taskFileForClaude
    }
  );

  const createdAt = new Date().toISOString();
  const manifest: RevisionManifest = {
    schema_version: 1,
    kind: "review_revision_manifest",
    request_signature: requestSignature,
    source_task_id: sourceTaskId,
    source_verification_signature: sourceVerificationSignature,
    source_execution_result_digest: result.digest,
    new_task_id: newTaskId,
    root_task_id: lineage.rootTaskId,
    parent_task_id: task.taskId,
    revision_number: revisionNumber,
    archive_directory: archiveDirectory,
    created_at: createdAt
  };

  await writeManifest(guard, workspace, manifestPath, manifest);

  const audit = await appendRevisionHistory(
    guard,
    workspace,
    createdTask.files.historyJsonl,
    {
      source_task_id: sourceTaskId,
      new_task_id: newTaskId,
      root_task_id: lineage.rootTaskId,
      parent_task_id: task.taskId,
      revision_number: revisionNumber,
      source_verification_signature: sourceVerificationSignature,
      source_execution_result_digest: result.digest,
      source_verdict: "revision_required",
      request_signature: requestSignature,
      archive_path: archiveDirectory,
      revision_manifest: manifestPath,
      new_task_json: createdTask.files.executionTaskJson
    }
  );

  return {
    ...base,
    created: true,
    archive: {
      directory: archiveDirectory,
      manifestPath,
      files: archived.files,
      warnings: archived.warnings
    },
    task: createdTask,
    audit
  };
}
