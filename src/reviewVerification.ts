import fsp from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";

import type { CodexProConfig } from "./config.js";
import {
  CodexProError,
  PathGuard,
  type Workspace
} from "./guard.js";
import {
  ensureAiBridge,
  writeTextFile
} from "./fsOps.js";
import {
  readStoredReviewExecutionTask,
  type ReviewExecutionFindingInput
} from "./reviewExecutionTask.js";
import {
  readExecutionResult,
  type ClaudeExecutionChangedFile,
  type ClaudeExecutionTestResult,
  type ParsedExecutionResult
} from "./reviewExecutionResult.js";

export type WebGPTVerificationVerdict =
  | "accepted"
  | "revision_required"
  | "blocked";

export type FindingVerificationStatus =
  | "verified"
  | "unresolved"
  | "blocked";

export type VerificationTestStatus =
  | "passed"
  | "failed"
  | "not_run"
  | "insufficient_evidence";

export type RemainingIssueVerificationStatus =
  | "resolved"
  | "unresolved"
  | "blocked";

export type RiskVerificationStatus =
  | "accepted"
  | "resolved"
  | "unresolved"
  | "blocked";

export type DeletedFileVerificationStatus =
  | "expected"
  | "unexpected"
  | "blocked";

export interface FindingVerificationInput {
  findingIndex: number;
  status: FindingVerificationStatus;
  summary: string;
  evidence?: string[];
}

export interface VerificationTestEvidenceInput {
  instructionIndex?: number;
  command?: string;
  status: VerificationTestStatus;
  summary: string;
  evidence?: string[];
}

export interface RemainingIssueVerificationInput {
  issueIndex: number;
  status: RemainingIssueVerificationStatus;
  summary: string;
  evidence?: string[];
}

export interface RiskVerificationInput {
  riskIndex: number;
  status: RiskVerificationStatus;
  summary: string;
  evidence?: string[];
}

export interface ObservedChangedFileInput {
  path: string;
  changeType:
    | "created"
    | "modified"
    | "deleted"
    | "renamed"
    | "unknown";
  summary?: string;
}

export interface DeletedFileVerificationInput {
  path: string;
  status: DeletedFileVerificationStatus;
  summary: string;
  evidence?: string[];
}

export interface RecordReviewVerificationInput {
  taskId: string;
  executionResultDigest: string;
  verdict: WebGPTVerificationVerdict;
  summary: string;

  findingResults: FindingVerificationInput[];
  testEvidence?: VerificationTestEvidenceInput[];
  remainingIssueResults?: RemainingIssueVerificationInput[];
  riskResults?: RiskVerificationInput[];

  reviewedFiles?: string[];
  actualChangedFiles?: ObservedChangedFileInput[];
  deletedFileResults?: DeletedFileVerificationInput[];
  forbiddenActionViolations?: string[];

  confirmed?: boolean;
}

export interface VerificationAcceptanceCheck {
  code: string;
  passed: boolean;
  message: string;
}

export interface VerificationAcceptanceGate {
  passed: boolean;
  checks: VerificationAcceptanceCheck[];
}

export interface RecordReviewVerificationResult {
  taskId: string;
  title: string;
  verdict: WebGPTVerificationVerdict;

  confirmed: boolean;
  recorded: boolean;
  idempotent: boolean;
  canRecord: boolean;

  executionResultDigest: string;
  currentExecutionResultDigest: string;
  verificationSignature: string;

  files: {
    markdown: string;
    json: string;
    history: string;
  };

  fileReview: {
    reviewedFiles: string[];
    reportedChangedFiles: string[];
    actualChangedFiles: ObservedChangedFileInput[];
    deletedFiles: string[];
    outOfScopeFiles: string[];
    unreportedChangedFiles: string[];
    reportedButNotObservedFiles: string[];
    unreviewedChangedFiles: string[];
  };

  acceptanceGate: VerificationAcceptanceGate;

  jsonPayload: Record<string, unknown>;
  markdown: string;

  writes: {
    markdown: {
      path: string;
      bytes: number;
      additions: number;
      deletions: number;
    } | null;
    json: {
      path: string;
      bytes: number;
      additions: number;
      deletions: number;
    } | null;
  };

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
    autoRevisionLoopAllowed: false;
  };
}

interface NormalizedFindingVerification {
  findingIndex: number;
  status: FindingVerificationStatus;
  summary: string;
  evidence: string[];
  finding: ReviewExecutionFindingInput;
}

interface NormalizedTestEvidence {
  instructionIndex: number | null;
  instruction: string | null;
  command: string | null;
  status: VerificationTestStatus;
  summary: string;
  evidence: string[];
}

interface NormalizedRemainingIssueVerification {
  issueIndex: number;
  issue: string;
  status: RemainingIssueVerificationStatus;
  summary: string;
  evidence: string[];
}

interface NormalizedRiskVerification {
  riskIndex: number;
  risk: string;
  status: RiskVerificationStatus;
  summary: string;
  evidence: string[];
}

interface NormalizedDeletedFileVerification {
  path: string;
  status: DeletedFileVerificationStatus;
  summary: string;
  evidence: string[];
}

const MAX_TEXT_BYTES = 40_000;
const MAX_LIST_ITEMS = 500;

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

function cleanOptionalOneLine(
  value: unknown,
  fieldName: string,
  maxBytes = 2_000
): string | null {
  if (value === undefined || value === null) return null;

  const text = String(value).replace(/\s+/g, " ").trim();
  if (!text) return null;

  if (text.includes("\0")) {
    throw new CodexProError(`${fieldName} contains a NUL byte.`);
  }

  if (Buffer.byteLength(text, "utf8") > maxBytes) {
    throw new CodexProError(`${fieldName} is too large.`);
  }

  return text;
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
    throw new CodexProError(
      `${fieldName} has too many items. Limit: ${maxItems}.`
    );
  }

  const values = value
    .map((item, index) =>
      cleanOptionalOneLine(
        item,
        `${fieldName}[${index}]`,
        4_000
      )
    )
    .filter((item): item is string => Boolean(item));

  return [...new Set(values)];
}

function normalizePositiveIndex(
  value: unknown,
  fieldName: string,
  max: number
): number {
  const index = Number(value);

  if (
    !Number.isInteger(index) ||
    index < 1 ||
    index > max
  ) {
    throw new CodexProError(
      `${fieldName} must be an integer from 1 to ${max}.`
    );
  }

  return index;
}

function normalizeDigest(value: unknown): string {
  const digest = String(value ?? "").trim().toLowerCase();

  if (!/^[a-f0-9]{64}$/.test(digest)) {
    throw new CodexProError(
      "executionResultDigest must be a 64-character lowercase SHA-256 digest."
    );
  }

  return digest;
}

function normalizeVerdict(
  value: unknown
): WebGPTVerificationVerdict {
  if (value === "accepted") return value;
  if (value === "revision_required") return value;
  if (value === "blocked") return value;

  throw new CodexProError(
    `Unsupported verification verdict: ${String(value)}`
  );
}

function normalizeWorkspacePath(
  guard: PathGuard,
  workspace: Workspace,
  value: unknown,
  fieldName: string
): string {
  const raw = cleanText(value, fieldName, 2_000)
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
    throw new CodexProError(
      `${fieldName} must be a safe workspace-relative file path.`
    );
  }

  guard.resolve(workspace, raw);
  return raw;
}

function normalizeScopePath(value: string): string {
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

  const changed = normalizeScopePath(changedFile);

  return allowedFiles.some((value) => {
    const allowed = normalizeScopePath(value);

    return (
      changed === allowed ||
      changed.startsWith(`${allowed}/`)
    );
  });
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
    .map(
      (key) =>
        `${JSON.stringify(key)}:${canonicalJson(record[key])}`
    )
    .join(",")}}`;
}

function sha256(value: unknown): string {
  return createHash("sha256")
    .update(canonicalJson(value), "utf8")
    .digest("hex");
}

function normalizeFindingResults(
  value: unknown,
  findings: ReviewExecutionFindingInput[]
): NormalizedFindingVerification[] {
  if (!Array.isArray(value)) {
    throw new CodexProError("findingResults must be an array.");
  }

  if (value.length !== findings.length) {
    throw new CodexProError(
      `findingResults must contain exactly ${findings.length} item(s), one for every original finding.`
    );
  }

  const seen = new Set<number>();

  const normalized = value.map((item, arrayIndex) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      throw new CodexProError(
        `findingResults[${arrayIndex}] must be an object.`
      );
    }

    const record = item as Record<string, unknown>;
    const findingIndex = normalizePositiveIndex(
      record.findingIndex ?? record.finding_index,
      `findingResults[${arrayIndex}].findingIndex`,
      findings.length
    );

    if (seen.has(findingIndex)) {
      throw new CodexProError(
        `findingResults contains duplicate findingIndex ${findingIndex}.`
      );
    }

    seen.add(findingIndex);

    const status = record.status;

    if (
      status !== "verified" &&
      status !== "unresolved" &&
      status !== "blocked"
    ) {
      throw new CodexProError(
        `Unsupported finding status at findingIndex ${findingIndex}: ${String(status)}`
      );
    }

    return {
      findingIndex,
      status: status as FindingVerificationStatus,
      summary: cleanText(
        record.summary,
        `findingResults[${arrayIndex}].summary`,
        8_000
      ),
      evidence: cleanStringList(
        record.evidence,
        `findingResults[${arrayIndex}].evidence`,
        100
      ),
      finding: findings[findingIndex - 1]
    };
  });

  return normalized.sort(
    (left, right) => left.findingIndex - right.findingIndex
  );
}

function normalizeTestEvidence(
  value: unknown,
  instructions: string[]
): NormalizedTestEvidence[] {
  if (value === undefined || value === null) return [];

  if (!Array.isArray(value)) {
    throw new CodexProError("testEvidence must be an array.");
  }

  if (value.length > 200) {
    throw new CodexProError(
      "testEvidence has too many items. Limit: 200."
    );
  }

  const seenInstructionIndexes = new Set<number>();

  return value.map((item, index) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      throw new CodexProError(
        `testEvidence[${index}] must be an object.`
      );
    }

    const record = item as Record<string, unknown>;
    let instructionIndex: number | null = null;

    if (
      record.instructionIndex !== undefined ||
      record.instruction_index !== undefined
    ) {
      if (!instructions.length) {
        throw new CodexProError(
          `testEvidence[${index}] references a test instruction, but the task has none.`
        );
      }

      instructionIndex = normalizePositiveIndex(
        record.instructionIndex ?? record.instruction_index,
        `testEvidence[${index}].instructionIndex`,
        instructions.length
      );

      if (seenInstructionIndexes.has(instructionIndex)) {
        throw new CodexProError(
          `testEvidence contains duplicate instructionIndex ${instructionIndex}.`
        );
      }

      seenInstructionIndexes.add(instructionIndex);
    }

    const status = record.status;

    if (
      status !== "passed" &&
      status !== "failed" &&
      status !== "not_run" &&
      status !== "insufficient_evidence"
    ) {
      throw new CodexProError(
        `Unsupported test evidence status at index ${index}: ${String(status)}`
      );
    }

    return {
      instructionIndex,
      instruction:
        instructionIndex === null
          ? null
          : instructions[instructionIndex - 1],
      command: cleanOptionalOneLine(
        record.command,
        `testEvidence[${index}].command`,
        4_000
      ),
      status,
      summary: cleanText(
        record.summary,
        `testEvidence[${index}].summary`,
        12_000
      ),
      evidence: cleanStringList(
        record.evidence,
        `testEvidence[${index}].evidence`,
        100
      )
    };
  });
}

function normalizeRemainingIssueResults(
  value: unknown,
  remainingIssues: string[]
): NormalizedRemainingIssueVerification[] {
  if (value === undefined || value === null) return [];

  if (!Array.isArray(value)) {
    throw new CodexProError(
      "remainingIssueResults must be an array."
    );
  }

  const seen = new Set<number>();

  return value.map((item, index) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      throw new CodexProError(
        `remainingIssueResults[${index}] must be an object.`
      );
    }

    if (!remainingIssues.length) {
      throw new CodexProError(
        "remainingIssueResults was provided, but the execution result has no remaining issues."
      );
    }

    const record = item as Record<string, unknown>;
    const issueIndex = normalizePositiveIndex(
      record.issueIndex ?? record.issue_index,
      `remainingIssueResults[${index}].issueIndex`,
      remainingIssues.length
    );

    if (seen.has(issueIndex)) {
      throw new CodexProError(
        `remainingIssueResults contains duplicate issueIndex ${issueIndex}.`
      );
    }

    seen.add(issueIndex);

    const status = record.status;

    if (
      status !== "resolved" &&
      status !== "unresolved" &&
      status !== "blocked"
    ) {
      throw new CodexProError(
        `Unsupported remaining issue status at issueIndex ${issueIndex}: ${String(status)}`
      );
    }

    return {
      issueIndex,
      issue: remainingIssues[issueIndex - 1],
      status: status as RemainingIssueVerificationStatus,
      summary: cleanText(
        record.summary,
        `remainingIssueResults[${index}].summary`,
        8_000
      ),
      evidence: cleanStringList(
        record.evidence,
        `remainingIssueResults[${index}].evidence`,
        100
      )
    };
  }).sort((left, right) => left.issueIndex - right.issueIndex);
}

function normalizeRiskResults(
  value: unknown,
  risks: string[]
): NormalizedRiskVerification[] {
  if (value === undefined || value === null) return [];

  if (!Array.isArray(value)) {
    throw new CodexProError("riskResults must be an array.");
  }

  const seen = new Set<number>();

  return value.map((item, index) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      throw new CodexProError(
        `riskResults[${index}] must be an object.`
      );
    }

    if (!risks.length) {
      throw new CodexProError(
        "riskResults was provided, but the execution result has no risks."
      );
    }

    const record = item as Record<string, unknown>;
    const riskIndex = normalizePositiveIndex(
      record.riskIndex ?? record.risk_index,
      `riskResults[${index}].riskIndex`,
      risks.length
    );

    if (seen.has(riskIndex)) {
      throw new CodexProError(
        `riskResults contains duplicate riskIndex ${riskIndex}.`
      );
    }

    seen.add(riskIndex);

    const status = record.status;

    if (
      status !== "accepted" &&
      status !== "resolved" &&
      status !== "unresolved" &&
      status !== "blocked"
    ) {
      throw new CodexProError(
        `Unsupported risk status at riskIndex ${riskIndex}: ${String(status)}`
      );
    }

    return {
      riskIndex,
      risk: risks[riskIndex - 1],
      status: status as RiskVerificationStatus,
      summary: cleanText(
        record.summary,
        `riskResults[${index}].summary`,
        8_000
      ),
      evidence: cleanStringList(
        record.evidence,
        `riskResults[${index}].evidence`,
        100
      )
    };
  }).sort((left, right) => left.riskIndex - right.riskIndex);
}

function normalizeObservedChangedFiles(
  guard: PathGuard,
  workspace: Workspace,
  value: unknown
): ObservedChangedFileInput[] {
  if (value === undefined || value === null) return [];

  if (!Array.isArray(value)) {
    throw new CodexProError(
      "actualChangedFiles must be an array."
    );
  }

  if (value.length > MAX_LIST_ITEMS) {
    throw new CodexProError(
      `actualChangedFiles has too many items. Limit: ${MAX_LIST_ITEMS}.`
    );
  }

  const seen = new Set<string>();

  return value.map((item, index) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      throw new CodexProError(
        `actualChangedFiles[${index}] must be an object.`
      );
    }

    const record = item as Record<string, unknown>;
    const filePath = normalizeWorkspacePath(
      guard,
      workspace,
      record.path,
      `actualChangedFiles[${index}].path`
    );

    if (seen.has(filePath)) {
      throw new CodexProError(
        `actualChangedFiles contains duplicate path: ${filePath}`
      );
    }

    seen.add(filePath);

    const changeType =
      record.changeType ?? record.change_type;

    if (
      changeType !== "created" &&
      changeType !== "modified" &&
      changeType !== "deleted" &&
      changeType !== "renamed" &&
      changeType !== "unknown"
    ) {
      throw new CodexProError(
        `Unsupported change type for ${filePath}: ${String(changeType)}`
      );
    }

    return {
      path: filePath,
      changeType: changeType as ObservedChangedFileInput["changeType"],
      summary:
        cleanOptionalOneLine(
          record.summary,
          `actualChangedFiles[${index}].summary`,
          8_000
        ) ?? undefined
    };
  }).sort((left, right) => left.path.localeCompare(right.path));
}

function normalizeReviewedFiles(
  guard: PathGuard,
  workspace: Workspace,
  value: unknown
): string[] {
  if (value === undefined || value === null) return [];

  if (!Array.isArray(value)) {
    throw new CodexProError("reviewedFiles must be an array.");
  }

  return [...new Set(
    value.map((item, index) =>
      normalizeWorkspacePath(
        guard,
        workspace,
        item,
        `reviewedFiles[${index}]`
      )
    )
  )].sort();
}

function normalizeDeletedFileResults(
  guard: PathGuard,
  workspace: Workspace,
  value: unknown,
  deletedFiles: string[]
): NormalizedDeletedFileVerification[] {
  if (value === undefined || value === null) return [];

  if (!Array.isArray(value)) {
    throw new CodexProError(
      "deletedFileResults must be an array."
    );
  }

  const deletedSet = new Set(deletedFiles);
  const seen = new Set<string>();

  return value.map((item, index) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      throw new CodexProError(
        `deletedFileResults[${index}] must be an object.`
      );
    }

    const record = item as Record<string, unknown>;
    const filePath = normalizeWorkspacePath(
      guard,
      workspace,
      record.path,
      `deletedFileResults[${index}].path`
    );

    if (!deletedSet.has(filePath)) {
      throw new CodexProError(
        `deletedFileResults references a path that is not in actualChangedFiles as deleted: ${filePath}`
      );
    }

    if (seen.has(filePath)) {
      throw new CodexProError(
        `deletedFileResults contains duplicate path: ${filePath}`
      );
    }

    seen.add(filePath);

    const status = record.status;

    if (
      status !== "expected" &&
      status !== "unexpected" &&
      status !== "blocked"
    ) {
      throw new CodexProError(
        `Unsupported deleted file status for ${filePath}: ${String(status)}`
      );
    }

    return {
      path: filePath,
      status: status as DeletedFileVerificationStatus,
      summary: cleanText(
        record.summary,
        `deletedFileResults[${index}].summary`,
        8_000
      ),
      evidence: cleanStringList(
        record.evidence,
        `deletedFileResults[${index}].evidence`,
        100
      )
    };
  }).sort((left, right) => left.path.localeCompare(right.path));
}

function check(
  code: string,
  passed: boolean,
  passMessage: string,
  failMessage: string
): VerificationAcceptanceCheck {
  return {
    code,
    passed,
    message: passed ? passMessage : failMessage
  };
}

function buildAcceptanceGate(input: {
  dispatchStatus: string;
  result: ParsedExecutionResult;
  findings: NormalizedFindingVerification[];
  testEvidence: NormalizedTestEvidence[];
  testInstructions: string[];
  remainingIssueResults: NormalizedRemainingIssueVerification[];
  riskResults: NormalizedRiskVerification[];
  deletedFileResults: NormalizedDeletedFileVerification[];
  deletedFiles: string[];
  actualChangedFiles: ObservedChangedFileInput[];
  outOfScopeFiles: string[];
  unreportedChangedFiles: string[];
  reportedButNotObservedFiles: string[];
  unreviewedChangedFiles: string[];
  forbiddenActionViolations: string[];
}): VerificationAcceptanceGate {
  const checks: VerificationAcceptanceCheck[] = [];

  checks.push(check(
    "task_dispatched",
    input.dispatchStatus !== "draft",
    "The task has a recorded dispatch lifecycle.",
    "The task lifecycle is still draft."
  ));

  checks.push(check(
    "execution_result_valid",
    input.result.validationErrors.length === 0 &&
      input.result.digest !== null,
    "The execution result is valid and digestible.",
    "The execution result is missing, invalid, or cannot produce a trusted digest."
  ));

  checks.push(check(
    "execution_completed",
    input.result.status === "completed",
    "Claude Code reported completed.",
    `Execution result status is ${input.result.status}, not completed.`
  ));

  checks.push(check(
    "execution_changes_reported",
    input.result.changedFiles.length > 0,
    "The execution result reports at least one changed file.",
    "The execution result reports no changed files."
  ));

  checks.push(check(
    "repository_changes_observed",
    input.actualChangedFiles.length > 0,
    "The real repository evidence contains at least one changed file.",
    "No actual repository changes were supplied for independent verification."
  ));

  const unresolvedFindings = input.findings.filter(
    (item) => item.status !== "verified"
  );

  checks.push(check(
    "all_findings_verified",
    unresolvedFindings.length === 0,
    "Every original finding is independently verified.",
    `${unresolvedFindings.length} finding(s) remain unresolved or blocked.`
  ));

  const instructionEvidence = new Map(
    input.testEvidence
      .filter((item) => item.instructionIndex !== null)
      .map((item) => [item.instructionIndex as number, item])
  );

  const missingInstructionEvidence = input.testInstructions
    .map((_, index) => index + 1)
    .filter((index) => instructionEvidence.get(index)?.status !== "passed");

  checks.push(check(
    "required_tests_verified",
    missingInstructionEvidence.length === 0,
    "Every explicit test instruction has independent passing evidence.",
    `${missingInstructionEvidence.length} explicit test instruction(s) lack passing evidence.`
  ));

  const unacceptableTestEvidence = input.testEvidence.filter(
    (item) => item.status !== "passed"
  );

  checks.push(check(
    "test_evidence_passed",
    unacceptableTestEvidence.length === 0,
    "All supplied test evidence passed.",
    `${unacceptableTestEvidence.length} test evidence item(s) failed, were not run, or have insufficient evidence.`
  ));

  const failedReportedTests = input.result.tests.filter(
    (item) => item.status === "failed"
  );

  checks.push(check(
    "no_reported_failed_tests",
    failedReportedTests.length === 0,
    "The execution result reports no failed tests.",
    `${failedReportedTests.length} failed test(s) are present in the execution result.`
  ));

  const unverifiedReportedTests = input.result.tests.filter(
    (item) =>
      item.status === "not_run" ||
      item.status === "unknown"
  );

  const hasIndependentPassedEvidence = input.testEvidence.some(
    (item) => item.status === "passed"
  );

  checks.push(check(
    "reported_unverified_tests_resolved",
    unverifiedReportedTests.length === 0 ||
      hasIndependentPassedEvidence,
    "No unverified reported tests remain without independent passing evidence.",
    `${unverifiedReportedTests.length} reported test(s) are not_run or unknown and no independent passing evidence was supplied.`
  ));

  const remainingIssuesCovered =
    input.remainingIssueResults.length ===
      input.result.remainingIssues.length;

  const unresolvedIssues = input.remainingIssueResults.filter(
    (item) => item.status !== "resolved"
  );

  checks.push(check(
    "remaining_issues_resolved",
    remainingIssuesCovered && unresolvedIssues.length === 0,
    "All reported remaining issues are covered and resolved.",
    "Reported remaining issues are missing an assessment or remain unresolved/blocked."
  ));

  const risksCovered =
    input.riskResults.length === input.result.risks.length;

  const blockingRisks = input.riskResults.filter(
    (item) =>
      item.status !== "accepted" &&
      item.status !== "resolved"
  );

  checks.push(check(
    "risks_assessed",
    risksCovered && blockingRisks.length === 0,
    "All reported risks are covered and accepted as non-blocking or resolved.",
    "Reported risks are missing an assessment or remain unresolved/blocked."
  ));

  checks.push(check(
    "no_pending_input",
    input.result.needsInput === null,
    "The execution result requires no additional input.",
    "The execution result still requires additional input."
  ));

  checks.push(check(
    "files_within_scope",
    input.outOfScopeFiles.length === 0,
    "All observed changed files are within the allowed scope.",
    `${input.outOfScopeFiles.length} observed changed file(s) are outside allowedFiles.`
  ));

  checks.push(check(
    "all_actual_changes_reported",
    input.unreportedChangedFiles.length === 0,
    "Every observed changed file was reported by Claude Code.",
    `${input.unreportedChangedFiles.length} observed changed file(s) were not reported by Claude Code.`
  ));

  checks.push(check(
    "reported_changes_observed",
    input.reportedButNotObservedFiles.length === 0,
    "Every Claude-reported changed file appears in the observed repository changes.",
    `${input.reportedButNotObservedFiles.length} Claude-reported changed file(s) were not observed in the repository changes.`
  ));

  checks.push(check(
    "all_changed_files_reviewed",
    input.unreviewedChangedFiles.length === 0,
    "Every non-deleted observed changed file was independently read and reviewed.",
    `${input.unreviewedChangedFiles.length} non-deleted changed file(s) were not listed in reviewedFiles.`
  ));

  const deletedFilesCovered =
    input.deletedFileResults.length === input.deletedFiles.length;

  const unexpectedDeletedFiles = input.deletedFileResults.filter(
    (item) => item.status !== "expected"
  );

  checks.push(check(
    "deleted_files_reviewed",
    deletedFilesCovered && unexpectedDeletedFiles.length === 0,
    "Every deleted file is explicitly reviewed and expected.",
    "Deleted files are missing an assessment or include unexpected/blocked deletions."
  ));

  checks.push(check(
    "no_forbidden_action_violations",
    input.forbiddenActionViolations.length === 0,
    "No forbidden action violation was observed.",
    `${input.forbiddenActionViolations.length} forbidden action violation(s) were recorded.`
  ));

  return {
    passed: checks.every((item) => item.passed),
    checks
  };
}

function bulletList(
  values: string[],
  emptyText: string
): string[] {
  return values.length
    ? values.map((value) => `- ${value}`)
    : [`- ${emptyText}`];
}

function buildVerificationMarkdown(
  payload: Record<string, unknown>
): string {
  const findingResults = payload.finding_results as Array<Record<string, unknown>>;
  const testEvidence = payload.test_evidence as Array<Record<string, unknown>>;
  const remainingIssueResults = payload.remaining_issue_results as Array<Record<string, unknown>>;
  const riskResults = payload.risk_results as Array<Record<string, unknown>>;
  const fileReview = payload.file_review as Record<string, unknown>;
  const acceptanceGate = payload.acceptance_gate as Record<string, unknown>;
  const checks = acceptanceGate.checks as Array<Record<string, unknown>>;

  return [
    "# WebGPT Verification Result",
    "",
    `Task ID: ${String(payload.task_id)}`,
    `Verdict: ${String(payload.verdict)}`,
    `Execution Result Digest: ${String(payload.execution_result_digest)}`,
    `Verified At: ${String(payload.verified_at)}`,
    "",
    "## Summary",
    "",
    String(payload.summary),
    "",
    "## Acceptance Gate",
    "",
    `Passed: ${String(acceptanceGate.passed)}`,
    "",
    ...checks.map(
      (item) =>
        `- ${item.passed ? "PASS" : "FAIL"} [${String(item.code)}] ${String(item.message)}`
    ),
    "",
    "## Finding Results",
    "",
    ...findingResults.flatMap((item) => [
      `### Finding ${String(item.finding_index)} — ${String(item.status)}`,
      "",
      String(item.summary),
      "",
      ...bulletList(
        Array.isArray(item.evidence)
          ? item.evidence.map(String)
          : [],
        "No separate evidence note supplied."
      ),
      ""
    ]),
    "## Test Evidence",
    "",
    ...(testEvidence.length
      ? testEvidence.flatMap((item) => [
          `- ${String(item.status).toUpperCase()} ${item.command ? `\`${String(item.command)}\`` : "(no command)"}: ${String(item.summary)}`
        ])
      : ["- No independent test evidence supplied."]),
    "",
    "## Remaining Issue Assessments",
    "",
    ...(remainingIssueResults.length
      ? remainingIssueResults.map(
          (item) =>
            `- ${String(item.status).toUpperCase()} #${String(item.issue_index)}: ${String(item.summary)}`
        )
      : ["- No remaining issues were reported or assessed."]),
    "",
    "## Risk Assessments",
    "",
    ...(riskResults.length
      ? riskResults.map(
          (item) =>
            `- ${String(item.status).toUpperCase()} #${String(item.risk_index)}: ${String(item.summary)}`
        )
      : ["- No risks were reported or assessed."]),
    "",
    "## File Review",
    "",
    "### Actual Changed Files",
    "",
    ...bulletList(
      (fileReview.actual_changed_files as Array<Record<string, unknown>>)
        .map(
          (item) =>
            `${String(item.path)} (${String(item.change_type)})`
        ),
      "No actual changed files supplied."
    ),
    "",
    "### Out-of-Scope Files",
    "",
    ...bulletList(
      (fileReview.out_of_scope_files as string[]) ?? [],
      "None."
    ),
    "",
    "### Unreported Changed Files",
    "",
    ...bulletList(
      (fileReview.unreported_changed_files as string[]) ?? [],
      "None."
    ),
    "",
    "### Reported But Not Observed Files",
    "",
    ...bulletList(
      (fileReview.reported_but_not_observed_files as string[]) ?? [],
      "None."
    ),
    "",
    "## Safety Boundary",
    "",
    "- This record stores WebGPT's independent verification conclusion.",
    "- It does not send or submit anything to Claude Code.",
    "- It does not confirm permission prompts.",
    "- It does not create or dispatch a revision task.",
    "- A later Execution Result digest makes this record stale.",
    ""
  ].join("\n");
}

async function appendVerificationHistory(
  guard: PathGuard,
  workspace: Workspace,
  historyPath: string,
  data: Record<string, unknown>
): Promise<{
  updated: boolean;
  error: string | null;
}> {
  try {
    const resolved = guard.resolve(
      workspace,
      historyPath,
      { forWrite: true }
    );

    await fsp.mkdir(path.dirname(resolved.absPath), {
      recursive: true
    });

    await fsp.appendFile(
      resolved.absPath,
      `${JSON.stringify({
        ts: new Date().toISOString(),
        event: "review_verification_recorded",
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

async function readExistingSignature(
  config: CodexProConfig,
  guard: PathGuard,
  workspace: Workspace,
  jsonPath: string
): Promise<string | null> {
  const resolved = guard.resolve(workspace, jsonPath);

  try {
    const stat = await fsp.stat(resolved.absPath);
    if (!stat.isFile()) return null;
  } catch (error) {
    const code =
      error && typeof error === "object" && "code" in error
        ? String((error as { code?: unknown }).code)
        : "";

    if (code === "ENOENT") return null;
    throw error;
  }

  await guard.assertTextFile(
    resolved.absPath,
    Math.min(config.maxReadBytes, 1_000_000)
  );

  try {
    const payload = JSON.parse(
      await fsp.readFile(resolved.absPath, "utf8")
    ) as Record<string, unknown>;

    const signature = String(
      payload.verification_signature ?? ""
    ).trim();

    return /^[a-f0-9]{64}$/.test(signature)
      ? signature
      : null;
  } catch {
    return null;
  }
}

function resultTestsForPayload(
  tests: ClaudeExecutionTestResult[]
): Array<Record<string, unknown>> {
  return tests.map((item) => ({
    command: item.command,
    status: item.status,
    summary: item.summary
  }));
}

function reportedChangedFilesForPayload(
  files: ClaudeExecutionChangedFile[]
): Array<Record<string, unknown>> {
  return files.map((item) => ({
    path: item.path,
    change_type: item.changeType,
    summary: item.summary
  }));
}

export async function recordReviewVerification(
  config: CodexProConfig,
  guard: PathGuard,
  workspace: Workspace,
  rawInput: RecordReviewVerificationInput
): Promise<RecordReviewVerificationResult> {
  const task = await readStoredReviewExecutionTask(
    config,
    guard,
    workspace
  );

  const taskId = cleanText(rawInput.taskId, "taskId", 200);

  if (taskId !== task.taskId) {
    throw new CodexProError(
      `taskId does not match the current task. Expected ${task.taskId}, received ${taskId}.`
    );
  }

  const executionResultDigest = normalizeDigest(
    rawInput.executionResultDigest
  );

  const result = await readExecutionResult(
    config,
    guard,
    workspace,
    task.resultJsonPath,
    task.taskId
  );

  if (!result.digest) {
    throw new CodexProError(
      "The current Execution Result is missing or invalid and cannot be verified."
    );
  }

  if (executionResultDigest !== result.digest) {
    throw new CodexProError(
      "Stale Execution Result: the supplied executionResultDigest does not match the current result. " +
        `Expected ${result.digest}, received ${executionResultDigest}. Re-run inspect_review_execution_result and review the current code before recording verification.`
    );
  }

  if (task.status === "draft") {
    throw new CodexProError(
      "Cannot record verification for a task whose dispatch lifecycle is still draft."
    );
  }

  const verdict = normalizeVerdict(rawInput.verdict);
  const summary = cleanText(rawInput.summary, "summary", 20_000);

  const findingResults = normalizeFindingResults(
    rawInput.findingResults,
    task.findings
  );

  const testEvidence = normalizeTestEvidence(
    rawInput.testEvidence,
    task.testInstructions
  );

  const remainingIssueResults =
    normalizeRemainingIssueResults(
      rawInput.remainingIssueResults,
      result.remainingIssues
    );

  const riskResults = normalizeRiskResults(
    rawInput.riskResults,
    result.risks
  );

  const reviewedFiles = normalizeReviewedFiles(
    guard,
    workspace,
    rawInput.reviewedFiles
  );

  const actualChangedFiles =
    normalizeObservedChangedFiles(
      guard,
      workspace,
      rawInput.actualChangedFiles
    );

  const reportedChangedFiles = result.changedFiles
    .map((item) => item.path)
    .sort();

  const actualChangedPaths = actualChangedFiles
    .map((item) => item.path)
    .sort();

  const reportedSet = new Set(reportedChangedFiles);
  const actualSet = new Set(actualChangedPaths);
  const reviewedSet = new Set(reviewedFiles);

  const deletedFiles = actualChangedFiles
    .filter((item) => item.changeType === "deleted")
    .map((item) => item.path)
    .sort();

  const outOfScopeFiles = actualChangedPaths.filter(
    (filePath) => !pathAllowed(filePath, task.allowedFiles)
  );

  const unreportedChangedFiles = actualChangedPaths.filter(
    (filePath) => !reportedSet.has(filePath)
  );

  const reportedButNotObservedFiles = reportedChangedFiles.filter(
    (filePath) => !actualSet.has(filePath)
  );

  const unreviewedChangedFiles = actualChangedFiles
    .filter((item) => item.changeType !== "deleted")
    .map((item) => item.path)
    .filter((filePath) => !reviewedSet.has(filePath))
    .sort();

  const deletedFileResults = normalizeDeletedFileResults(
    guard,
    workspace,
    rawInput.deletedFileResults,
    deletedFiles
  );

  const forbiddenActionViolations = cleanStringList(
    rawInput.forbiddenActionViolations,
    "forbiddenActionViolations",
    200
  );

  const acceptanceGate = buildAcceptanceGate({
    dispatchStatus: task.status,
    result,
    findings: findingResults,
    testEvidence,
    testInstructions: task.testInstructions,
    remainingIssueResults,
    riskResults,
    deletedFileResults,
    deletedFiles,
    actualChangedFiles,
    outOfScopeFiles,
    unreportedChangedFiles,
    reportedButNotObservedFiles,
    unreviewedChangedFiles,
    forbiddenActionViolations
  });

  const hasBlockedEvidence =
    result.status === "blocked" ||
    result.status === "failed" ||
    result.status === "needs_input" ||
    result.needsInput !== null ||
    findingResults.some((item) => item.status === "blocked") ||
    remainingIssueResults.some((item) => item.status === "blocked") ||
    riskResults.some((item) => item.status === "blocked") ||
    deletedFileResults.some((item) => item.status === "blocked");

  if (verdict === "blocked" && !hasBlockedEvidence) {
    throw new CodexProError(
      "verdict=blocked requires an execution blocker, needs_input, or at least one blocked finding/issue/risk/deleted-file assessment."
    );
  }

  const confirmed = rawInput.confirmed === true;
  const canRecord =
    verdict !== "accepted" || acceptanceGate.passed;

  if (confirmed && !canRecord) {
    const failedChecks = acceptanceGate.checks
      .filter((item) => !item.passed)
      .map((item) => `${item.code}: ${item.message}`)
      .join("; ");

    throw new CodexProError(
      `Cannot record verdict=accepted because the Acceptance Gate failed. ${failedChecks}`
    );
  }

  const verifiedAt = new Date().toISOString();

  const signaturePayload = {
    task_id: task.taskId,
    execution_result_digest: result.digest,
    verdict,
    summary,
    finding_results: findingResults.map((item) => ({
      finding_index: item.findingIndex,
      status: item.status,
      summary: item.summary,
      evidence: item.evidence
    })),
    test_evidence: testEvidence.map((item) => ({
      instruction_index: item.instructionIndex,
      command: item.command,
      status: item.status,
      summary: item.summary,
      evidence: item.evidence
    })),
    remaining_issue_results: remainingIssueResults.map((item) => ({
      issue_index: item.issueIndex,
      status: item.status,
      summary: item.summary,
      evidence: item.evidence
    })),
    risk_results: riskResults.map((item) => ({
      risk_index: item.riskIndex,
      status: item.status,
      summary: item.summary,
      evidence: item.evidence
    })),
    reviewed_files: reviewedFiles,
    actual_changed_files: actualChangedFiles.map((item) => ({
      path: item.path,
      change_type: item.changeType,
      summary: item.summary ?? ""
    })),
    deleted_file_results: deletedFileResults.map((item) => ({
      path: item.path,
      status: item.status,
      summary: item.summary,
      evidence: item.evidence
    })),
    forbidden_action_violations: forbiddenActionViolations
  };

  const verificationSignature = sha256(signaturePayload);

  const jsonPayload: Record<string, unknown> = {
    schema_version: 1,
    kind: "webgpt_review_verification",

    task_id: task.taskId,
    task_title: task.title,
    execution_result_digest: result.digest,
    verification_signature: verificationSignature,

    verdict,
    summary,

    finding_results: findingResults.map((item) => ({
      finding_index: item.findingIndex,
      status: item.status,
      summary: item.summary,
      evidence: item.evidence,
      original_finding: {
        file: item.finding.file ?? null,
        issue: item.finding.issue,
        recommendation: item.finding.recommendation,
        priority: item.finding.priority ?? "medium",
        risk: item.finding.risk ?? null
      }
    })),

    test_evidence: testEvidence.map((item) => ({
      instruction_index: item.instructionIndex,
      instruction: item.instruction,
      command: item.command,
      status: item.status,
      summary: item.summary,
      evidence: item.evidence
    })),

    remaining_issue_results: remainingIssueResults.map((item) => ({
      issue_index: item.issueIndex,
      issue: item.issue,
      status: item.status,
      summary: item.summary,
      evidence: item.evidence
    })),

    risk_results: riskResults.map((item) => ({
      risk_index: item.riskIndex,
      risk: item.risk,
      status: item.status,
      summary: item.summary,
      evidence: item.evidence
    })),

    file_review: {
      reviewed_files: reviewedFiles,
      reported_changed_files:
        reportedChangedFilesForPayload(result.changedFiles),
      actual_changed_files: actualChangedFiles.map((item) => ({
        path: item.path,
        change_type: item.changeType,
        summary: item.summary ?? ""
      })),
      deleted_files: deletedFiles,
      deleted_file_results: deletedFileResults.map((item) => ({
        path: item.path,
        status: item.status,
        summary: item.summary,
        evidence: item.evidence
      })),
      out_of_scope_files: outOfScopeFiles,
      unreported_changed_files: unreportedChangedFiles,
      reported_but_not_observed_files: reportedButNotObservedFiles,
      unreviewed_changed_files: unreviewedChangedFiles
    },

    forbidden_action_violations: forbiddenActionViolations,

    execution_result_snapshot: {
      status: result.status,
      updated_at: result.updatedAt,
      tests: resultTestsForPayload(result.tests),
      remaining_issues: result.remainingIssues,
      risks: result.risks,
      needs_input: result.needsInput
    },

    acceptance_gate: acceptanceGate,

    lifecycle: {
      state: "recorded",
      recorded_at: verifiedAt
    },

    verified_at: verifiedAt,
    updated_at: verifiedAt,
    producer: "webgpt"
  };

  const markdown = buildVerificationMarkdown(jsonPayload);

  const baseResult: RecordReviewVerificationResult = {
    taskId: task.taskId,
    title: task.title,
    verdict,

    confirmed,
    recorded: false,
    idempotent: false,
    canRecord,

    executionResultDigest,
    currentExecutionResultDigest: result.digest,
    verificationSignature,

    files: {
      markdown: task.verificationMarkdownPath,
      json: task.verificationJsonPath,
      history: task.historyPath
    },

    fileReview: {
      reviewedFiles,
      reportedChangedFiles,
      actualChangedFiles,
      deletedFiles,
      outOfScopeFiles,
      unreportedChangedFiles,
      reportedButNotObservedFiles,
      unreviewedChangedFiles
    },

    acceptanceGate,

    jsonPayload,
    markdown,

    writes: {
      markdown: null,
      json: null
    },

    audit: {
      updated: false,
      error: null
    },

    safety: {
      requiredExplicitConfirmation: true,
      previewDefault: true,
      autoDispatchAllowed: false,
      autoSubmitAllowed: false,
      autoPermissionConfirmAllowed: false,
      autoRevisionLoopAllowed: false
    }
  };

  if (!confirmed) {
    return baseResult;
  }

  if (config.writeMode === "off") {
    throw new CodexProError(
      "record_review_verification requires write mode handoff or workspace when confirmed=true. Preview remains available with confirmed=false."
    );
  }

  await ensureAiBridge(config, guard, workspace);

  const existingSignature = await readExistingSignature(
    config,
    guard,
    workspace,
    task.verificationJsonPath
  );

  if (existingSignature === verificationSignature) {
    return {
      ...baseResult,
      recorded: true,
      idempotent: true
    };
  }

  const markdownWrite = await writeTextFile(
    config,
    guard,
    workspace,
    task.verificationMarkdownPath,
    markdown,
    {
      createDirs: true,
      overwrite: true
    }
  );

  const jsonWrite = await writeTextFile(
    config,
    guard,
    workspace,
    task.verificationJsonPath,
    `${JSON.stringify(jsonPayload, null, 2)}\n`,
    {
      createDirs: true,
      overwrite: true
    }
  );

  const audit = await appendVerificationHistory(
    guard,
    workspace,
    task.historyPath,
    {
      task_id: task.taskId,
      execution_result_digest: result.digest,
      verification_signature: verificationSignature,
      verdict,
      acceptance_gate_passed: acceptanceGate.passed,
      verification_markdown: task.verificationMarkdownPath,
      verification_json: task.verificationJsonPath
    }
  );

  return {
    ...baseResult,
    recorded: true,
    writes: {
      markdown: {
        path: markdownWrite.path,
        bytes: markdownWrite.bytes,
        additions: markdownWrite.diff.additions,
        deletions: markdownWrite.diff.deletions
      },
      json: {
        path: jsonWrite.path,
        bytes: jsonWrite.bytes,
        additions: jsonWrite.diff.additions,
        deletions: jsonWrite.diff.deletions
      }
    },
    audit
  };
}
