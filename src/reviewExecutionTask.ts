import fsp from "node:fs/promises";
import path from "node:path";
import { randomUUID } from "node:crypto";

import type { CodexProConfig } from "./config.js";
import { CodexProError, PathGuard, type Workspace } from "./guard.js";
import { ensureAiBridge, writeTextFile } from "./fsOps.js";
import {
  buildClaudeCodeSkillCommand,
  invokeClaudeCodeSkill,
  listClaudeCodeTargets,
  sendToClaudeCode,
  type ClaudeCodeSendResult,
  type ClaudeCodeSkillResult
} from "./claudeCodeBridge.js";

export type ReviewFindingPriority = "low" | "medium" | "high";

export type ReviewExecutionPromptMode =
  | "inline"
  | "file_reference"
  | "skill_file_reference";

export type ReviewExecutionTaskFileForClaude =
  | "execution_task"
  | "review_plan";

export type ReviewExecutionDispatchMode = "text" | "skill";

export type ReviewExecutionTaskLifecycleStatus =
  | "draft"
  | "pasted"
  | "submitted"
  | "dispatch_failed";

export interface ReviewTaskLineage {
  rootTaskId: string;
  parentTaskId: string | null;
  revisionNumber: number;
  sourceVerificationSignature: string | null;
  sourceExecutionResultDigest: string | null;
  sourceVerdict: "revision_required" | null;
}

export interface DispatchReviewExecutionTaskInput {
  /**
   * Must exactly match task_id in claude-execution-task.json.
   */
  taskId: string;

  /**
   * Optional target override for this dispatch.
   * If omitted, the target stored in the task JSON is used.
   */
  target?: string;

  /**
   * text:
   *   Send the stored text/file-reference prompt.
   *
   * skill:
   *   Invoke the Skill selected in this dispatch and pass the task file
   *   path as the Skill arguments.
   *
   * Default: text.
   */
  dispatchMode?: ReviewExecutionDispatchMode;

  /**
   * Required only when dispatchMode=skill.
   * This is the actual Skill used for this dispatch.
   *
   * It is not taken from the task JSON.
   */
  skill?: string;

  /**
   * false:
   *   Preview only. No tmux input.
   *
   * true:
   *   Perform the dispatch.
   *
   * Default: false.
   */
  confirmed?: boolean;

  /**
   * false:
   *   Paste the content but do not press Enter.
   *
   * true:
   *   Paste the content and press Enter.
   *
   * Default: false.
   */
  submit?: boolean;
}

export interface DispatchReviewExecutionTaskResult {
  dispatchId: string;
  taskId: string;
  title: string;
  target: string;

  dispatchMode: ReviewExecutionDispatchMode;
  skill: string | null;
  recommendedSkill: string | null;

  taskFilePath: string;
  prompt: string;

  confirmed: boolean;
  submit: boolean;
  sent: boolean;

  statusBefore: ReviewExecutionTaskLifecycleStatus;
  statusAfter: ReviewExecutionTaskLifecycleStatus;

  bridgeResult: ClaudeCodeSendResult | ClaudeCodeSkillResult | null;

  auditUpdated: boolean;
  auditError?: string;

  taskWrite?: {
    path: string;
    bytes: number;
    additions: number;
    deletions: number;
  };

  safety: {
    requiredExplicitConfirmation: true;
    submitDefault: false;
    permissionAutoConfirmAllowed: false;
    automaticLoopAllowed: false;
  };
}

export interface StoredReviewExecutionTask {
  jsonPath: string;
  payload: Record<string, unknown>;

  schemaVersion: number;
  taskId: string;
  title: string;
  status: ReviewExecutionTaskLifecycleStatus;
  lineage: ReviewTaskLineage;

  target: string | null;
  promptMode: ReviewExecutionPromptMode;

  promptForClaude: string;
  executionInstructions: string;

  taskFilePath: string;
  recommendedSkill: string | null;

  reviewGoal: string;
  reviewedFiles: string[];
  findings: ReviewExecutionFindingInput[];
  allowedFiles: string[];
  forbiddenActions: string[];
  testInstructions: string[];

  resultMarkdownPath: string;
  resultJsonPath: string;

  verificationMarkdownPath: string;
  verificationJsonPath: string;

  historyPath: string;
}

export interface ReviewExecutionFindingInput {
  file?: string;
  issue: string;
  recommendation: string;
  priority?: ReviewFindingPriority;
  risk?: string;
}

export interface CreateReviewExecutionTaskInput {
  target?: string;

  /**
   * Internal deterministic task id used by create_review_revision_task.
   * The MCP create_review_execution_task tool does not expose this field.
   */
  taskId?: string;

  /**
   * Internal lineage supplied by create_review_revision_task.
   * Ordinary tasks receive a root lineage automatically.
   */
  lineage?: ReviewTaskLineage;
  title: string;
  reviewGoal: string;
  reviewedFiles?: string[];
  findings: ReviewExecutionFindingInput[];
  allowedFiles?: string[];
  forbiddenActions?: string[];
  testInstructions?: string[];
  extraContext?: string;

  /**
   * inline:
   *   Return the full task prompt for Claude Code.
   *
   * file_reference:
   *   Return a short prompt that tells Claude Code to read the handoff file.
   *
   * skill_file_reference:
   *   Return a direct slash command such as:
   *   /implement-task .ai-bridge/claude-execution-task.md
   */
  promptMode?: ReviewExecutionPromptMode;

  /**
   * Required when promptMode=skill_file_reference.
   * Can be passed as "implement-task" or "/implement-task".
   */
  claudeSkill?: string;

  /**
   * Which handoff file Claude Code should be asked to read.
   * Default: execution_task.
   */
  taskFileForClaude?: ReviewExecutionTaskFileForClaude;
}

export interface ReviewExecutionTaskFiles {
  reviewPlanMarkdown: string;
  executionTaskMarkdown: string;
  executionTaskJson: string;

  executionResultMarkdown: string;
  executionResultJson: string;

  verificationResultMarkdown: string;
  verificationResultJson: string;

  historyJsonl: string;
}

export interface CreateReviewExecutionTaskResult {
  taskId: string;
  target: string | null;
  title: string;
  status: "draft";
  lineage: ReviewTaskLineage;

  promptMode: ReviewExecutionPromptMode;
  claudeSkill: string | null;
  taskFileForClaude: ReviewExecutionTaskFileForClaude;
  taskFilePath: string;

  files: ReviewExecutionTaskFiles;

  /**
   * The short or inline prompt that should be sent to Claude Code
   * only after user confirmation.
   */
  promptForClaude: string;

  /**
   * The full execution instructions saved into claude-execution-task.md.
   */
  executionInstructions: string;

  reviewPlanMarkdown: string;
  executionTaskMarkdown: string;
  jsonPayload: Record<string, unknown>;
  writes: {
    reviewPlanMarkdown: {
      path: string;
      bytes: number;
      additions: number;
      deletions: number;
    };
    executionTaskMarkdown: {
      path: string;
      bytes: number;
      additions: number;
      deletions: number;
    };
    executionTaskJson: {
      path: string;
      bytes: number;
      additions: number;
      deletions: number;
    };
    executionResultMarkdown: {
      path: string;
      bytes: number;
      additions: number;
      deletions: number;
    };
    
    executionResultJson: {
      path: string;
      bytes: number;
      additions: number;
      deletions: number;
    };
  };
  safety: {
    requiresUserConfirmation: true;
    submitDefault: false;
    autoSendAllowed: false;
    autoSubmitAllowed: false;
    autoPermissionConfirmAllowed: false;
    autoLoopAllowed: false;
  };
}

const MAX_TITLE_LENGTH = 160;
const MAX_TEXT_LENGTH = 20_000;
const MAX_LIST_ITEMS = 200;
const MAX_FINDINGS = 80;

function cleanText(value: unknown, fieldName: string, maxLength = MAX_TEXT_LENGTH): string {
  const text = String(value ?? "").replace(/\r\n/g, "\n").trim();

  if (!text) {
    throw new CodexProError(`${fieldName} must not be empty.`);
  }

  if (text.includes("\0")) {
    throw new CodexProError(`${fieldName} contains a NUL byte.`);
  }

  if (Buffer.byteLength(text, "utf8") > maxLength) {
    throw new CodexProError(`${fieldName} is too large. Limit: ${maxLength} bytes.`);
  }

  return text;
}

function cleanOptionalText(value: unknown, fieldName: string, maxLength = MAX_TEXT_LENGTH): string | undefined {
  if (value === undefined || value === null) return undefined;

  const text = String(value).replace(/\r\n/g, "\n").trim();
  if (!text) return undefined;

  if (text.includes("\0")) {
    throw new CodexProError(`${fieldName} contains a NUL byte.`);
  }

  if (Buffer.byteLength(text, "utf8") > maxLength) {
    throw new CodexProError(`${fieldName} is too large. Limit: ${maxLength} bytes.`);
  }

  return text;
}

function cleanOneLine(value: unknown, fieldName: string, maxLength = 300): string {
  const text = cleanText(value, fieldName, maxLength).replace(/\s+/g, " ").trim();

  if (text.includes("\n")) {
    throw new CodexProError(`${fieldName} must be one line.`);
  }

  return text;
}

function cleanOptionalOneLine(value: unknown, fieldName: string, maxLength = 300): string | undefined {
  if (value === undefined || value === null) return undefined;
  const text = String(value).replace(/\s+/g, " ").trim();
  if (!text) return undefined;

  if (text.includes("\0")) {
    throw new CodexProError(`${fieldName} contains a NUL byte.`);
  }

  if (Buffer.byteLength(text, "utf8") > maxLength) {
    throw new CodexProError(`${fieldName} is too large. Limit: ${maxLength} bytes.`);
  }

  return text;
}

function cleanStringList(value: unknown, fieldName: string, maxItems = MAX_LIST_ITEMS): string[] {
  if (value === undefined || value === null) return [];

  if (!Array.isArray(value)) {
    throw new CodexProError(`${fieldName} must be an array.`);
  }

  if (value.length > maxItems) {
    throw new CodexProError(`${fieldName} has too many items. Limit: ${maxItems}.`);
  }

  const out: string[] = [];

  for (const item of value) {
    const text = String(item ?? "").replace(/\s+/g, " ").trim();
    if (!text) continue;

    if (text.includes("\0")) {
      throw new CodexProError(`${fieldName} contains a NUL byte.`);
    }

    if (Buffer.byteLength(text, "utf8") > 500) {
      throw new CodexProError(`${fieldName} contains an item that is too large.`);
    }

    out.push(text);
  }

  return [...new Set(out)];
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

function normalizeLineage(
  value: unknown,
  taskId: string
): ReviewTaskLineage {
  if (value === undefined || value === null) {
    return {
      rootTaskId: taskId,
      parentTaskId: null,
      revisionNumber: 0,
      sourceVerificationSignature: null,
      sourceExecutionResultDigest: null,
      sourceVerdict: null
    };
  }

  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new CodexProError("lineage must be an object.");
  }

  const record = value as Record<string, unknown>;
  const rootTaskId = normalizeTaskId(
    record.rootTaskId ?? record.root_task_id,
    "lineage.rootTaskId"
  );

  const parentRaw = record.parentTaskId ?? record.parent_task_id;
  const parentTaskId =
    parentRaw === undefined || parentRaw === null || String(parentRaw).trim() === ""
      ? null
      : normalizeTaskId(parentRaw, "lineage.parentTaskId");

  const revisionNumber = Number(
    record.revisionNumber ?? record.revision_number ?? 0
  );

  if (!Number.isInteger(revisionNumber) || revisionNumber < 0 || revisionNumber > 10_000) {
    throw new CodexProError(
      "lineage.revisionNumber must be an integer between 0 and 10000."
    );
  }

  const sourceVerificationSignature = cleanOptionalOneLine(
    record.sourceVerificationSignature ??
      record.source_verification_signature,
    "lineage.sourceVerificationSignature",
    64
  ) ?? null;

  const sourceExecutionResultDigest = cleanOptionalOneLine(
    record.sourceExecutionResultDigest ??
      record.source_execution_result_digest,
    "lineage.sourceExecutionResultDigest",
    64
  ) ?? null;

  const sourceVerdictRaw =
    record.sourceVerdict ?? record.source_verdict;
  const sourceVerdict =
    sourceVerdictRaw === "revision_required"
      ? "revision_required"
      : null;

  if (revisionNumber === 0) {
    if (rootTaskId !== taskId || parentTaskId !== null) {
      throw new CodexProError(
        "Root tasks must use their own task id as rootTaskId and have no parentTaskId."
      );
    }

    return {
      rootTaskId,
      parentTaskId: null,
      revisionNumber: 0,
      sourceVerificationSignature: null,
      sourceExecutionResultDigest: null,
      sourceVerdict: null
    };
  }

  if (!parentTaskId) {
    throw new CodexProError(
      "Revision tasks require lineage.parentTaskId."
    );
  }

  if (!sourceVerificationSignature || !/^[a-f0-9]{64}$/i.test(sourceVerificationSignature)) {
    throw new CodexProError(
      "Revision tasks require a valid lineage.sourceVerificationSignature."
    );
  }

  if (!sourceExecutionResultDigest || !/^[a-f0-9]{64}$/i.test(sourceExecutionResultDigest)) {
    throw new CodexProError(
      "Revision tasks require a valid lineage.sourceExecutionResultDigest."
    );
  }

  if (sourceVerdict !== "revision_required") {
    throw new CodexProError(
      "Revision tasks require lineage.sourceVerdict=revision_required."
    );
  }

  return {
    rootTaskId,
    parentTaskId,
    revisionNumber,
    sourceVerificationSignature: sourceVerificationSignature.toLowerCase(),
    sourceExecutionResultDigest: sourceExecutionResultDigest.toLowerCase(),
    sourceVerdict
  };
}

function normalizePriority(value: unknown): ReviewFindingPriority {
  if (value === "low" || value === "medium" || value === "high") return value;
  return "medium";
}

function normalizePromptMode(value: unknown): ReviewExecutionPromptMode {
  if (value === "inline") return "inline";
  if (value === "skill_file_reference") return "skill_file_reference";
  return "file_reference";
}

function normalizeTaskFileForClaude(value: unknown): ReviewExecutionTaskFileForClaude {
  if (value === "review_plan") return "review_plan";
  return "execution_task";
}

function normalizeClaudeSkill(value: unknown): string | null {
  if (value === undefined || value === null) return null;

  const raw = String(value).trim();
  if (!raw) return null;

  const skill = raw.startsWith("/") ? raw.slice(1).trim() : raw;

  if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(skill)) {
    throw new CodexProError(
      "Claude Code skill must use only letters, numbers, underscores, and hyphens. " +
        "It may optionally include a leading slash. Example: implement-task or /implement-task."
    );
  }

  return skill;
}

function taskFilePathForClaude(
  files: ReviewExecutionTaskFiles,
  taskFileForClaude: ReviewExecutionTaskFileForClaude
): string {
  return taskFileForClaude === "review_plan"
    ? files.reviewPlanMarkdown
    : files.executionTaskMarkdown;
}

function buildFileReferencePrompt(taskFilePath: string): string {
  return [
    "请读取并执行以下 Claude Code 任务文件：",
    "",
    taskFilePath,
    "",
    "执行要求：",
    "- 先阅读任务文件内容。",
    "- 只按任务文件中的允许修改范围进行最小必要修改。",
    "- 不要扩大任务范围。",
    "- 不要自动确认任何权限请求；如遇权限提示，请停下来让用户人工确认。",
    "- 完成后请汇报修改文件、修改原因、测试结果和剩余风险。"
  ].join("\n");
}

function buildSkillFileReferencePrompt(skill: string, taskFilePath: string): string {
  return [
    `/${skill} ${taskFilePath}`,
    "",
    "请先读取上述任务文件，再按该 Skill 的流程执行。",
    "",
    "执行要求：",
    "- 只处理任务文件中定义的范围。",
    "- 不要扩大任务范围。",
    "- 不要自动确认任何权限请求；如遇权限提示，请停下来让用户人工确认。",
    "- 完成后请汇报修改文件、修改原因、测试结果和剩余风险。"
  ].join("\n");
}

function buildPromptForClaudeByMode(input: {
  promptMode: ReviewExecutionPromptMode;
  claudeSkill: string | null;
  taskFilePath: string;
  executionInstructions: string;
}): string {
  if (input.promptMode === "inline") {
    return input.executionInstructions;
  }

  if (input.promptMode === "skill_file_reference") {
    if (!input.claudeSkill) {
      throw new CodexProError(
        "claudeSkill is required when promptMode is skill_file_reference."
      );
    }

    return buildSkillFileReferencePrompt(input.claudeSkill, input.taskFilePath);
  }

  return buildFileReferencePrompt(input.taskFilePath);
}

function priorityRank(priority: ReviewFindingPriority): number {
  if (priority === "high") return 0;
  if (priority === "medium") return 1;
  return 2;
}

function normalizeFindings(value: unknown): Required<ReviewExecutionFindingInput>[] {
  if (!Array.isArray(value)) {
    throw new CodexProError("findings must be an array.");
  }

  if (value.length < 1) {
    throw new CodexProError("findings must contain at least one review finding.");
  }

  if (value.length > MAX_FINDINGS) {
    throw new CodexProError(`findings has too many items. Limit: ${MAX_FINDINGS}.`);
  }

  const findings = value.map((item, index) => {
    const raw = item && typeof item === "object" ? (item as Record<string, unknown>) : {};
    return {
      file: cleanOptionalOneLine(raw.file, `findings[${index}].file`) ?? "",
      issue: cleanText(raw.issue, `findings[${index}].issue`, 4_000),
      recommendation: cleanText(raw.recommendation, `findings[${index}].recommendation`, 4_000),
      priority: normalizePriority(raw.priority),
      risk: cleanOptionalText(raw.risk, `findings[${index}].risk`, 2_000) ?? ""
    };
  });

  return findings.sort((a, b) => priorityRank(a.priority) - priorityRank(b.priority));
}

function validateTarget(target: string | undefined): string | null {
  const cleanTarget = cleanOptionalOneLine(target, "target", 80);
  if (!cleanTarget) return null;

  const allowed = listClaudeCodeTargets();
  const matched = allowed.find((item) => item.name === cleanTarget);

  if (!matched) {
    const names = allowed.map((item) => item.name).join(", ") || "none";
    throw new CodexProError(
      `Claude Code target is not allowlisted: ${cleanTarget}. Allowed targets from CODEXPRO_CLAUDE_TARGETS: ${names}`
    );
  }

  return matched.name;
}

function bulletList(items: string[], emptyText: string): string {
  if (!items.length) return `- ${emptyText}`;
  return items.map((item) => `- ${item}`).join("\n");
}

function numberedFindings(findings: Required<ReviewExecutionFindingInput>[]): string {
  return findings
    .map((finding, index) => {
      const lines = [
        `### ${index + 1}. [${finding.priority}] ${finding.file || "workspace"}`,
        "",
        `- File: ${finding.file || "(not specified)"}`,
        `- Issue: ${finding.issue}`,
        `- Recommendation: ${finding.recommendation}`
      ];

      if (finding.risk) {
        lines.push(`- Risk: ${finding.risk}`);
      }

      return lines.join("\n");
    })
    .join("\n\n");
}

function buildReviewPlanMarkdown(input: {
  taskId: string;
  target: string | null;
  title: string;
  reviewGoal: string;
  reviewedFiles: string[];
  findings: Required<ReviewExecutionFindingInput>[];
  allowedFiles: string[];
  forbiddenActions: string[];
  testInstructions: string[];
  extraContext?: string;
  workspace: Workspace;
  createdAt: string;
  lineage: ReviewTaskLineage;
}): string {
  const lineageLines = input.lineage.revisionNumber > 0
    ? [
        `Root Task ID: ${input.lineage.rootTaskId}`,
        `Parent Task ID: ${input.lineage.parentTaskId}`,
        `Revision Number: ${input.lineage.revisionNumber}`,
        `Source Verification Signature: ${input.lineage.sourceVerificationSignature}`,
        `Source Execution Result Digest: ${input.lineage.sourceExecutionResultDigest}`
      ]
    : [
        `Root Task ID: ${input.lineage.rootTaskId}`,
        "Revision Number: 0"
      ];

  return [
    `# WebGPT Review Plan`,
    "",
    `Task ID: ${input.taskId}`,
    `Created: ${input.createdAt}`,
    `Workspace: ${input.workspace.root}`,
    `Claude Code target: ${input.target ?? "(not selected)"}`,
    `Status: draft`,
    ...lineageLines,
    "",
    "## Review Goal",
    "",
    input.reviewGoal,
    "",
    "## Reviewed Files",
    "",
    bulletList(input.reviewedFiles, "No explicit reviewed files were provided."),
    "",
    "## Findings",
    "",
    numberedFindings(input.findings),
    "",
    "## Allowed Files For Claude Code",
    "",
    bulletList(input.allowedFiles, "Not explicitly constrained. Claude Code must still keep edits scoped to the task."),
    "",
    "## Forbidden Actions",
    "",
    bulletList(input.forbiddenActions, "No extra forbidden actions were provided. Default safety rules still apply."),
    "",
    "## Test Instructions",
    "",
    bulletList(input.testInstructions, "No explicit test instructions were provided. Claude Code should run focused verification when practical."),
    "",
    input.extraContext
      ? ["## Extra Context", "", input.extraContext, ""].join("\n")
      : "",
    "## Safety Boundary",
    "",
    "- This is a WebGPT review handoff artifact.",
    "- It does not execute code.",
    "- It does not send text to Claude Code.",
    "- It does not confirm Claude Code permission prompts.",
    "- User confirmation is required before sending any prompt to Claude Code.",
    ""
  ]
    .filter((part) => part !== "")
    .join("\n");
}

function buildPromptForClaude(input: {
  taskId: string;
  title: string;
  reviewGoal: string;
  reviewedFiles: string[];
  findings: Required<ReviewExecutionFindingInput>[];
  allowedFiles: string[];
  forbiddenActions: string[];
  testInstructions: string[];
  extraContext?: string;
  lineage: ReviewTaskLineage;
}): string {
  const lineageText = input.lineage.revisionNumber > 0
    ? [
        "## Revision Lineage",
        "",
        `Root Task ID：${input.lineage.rootTaskId}`,
        `Parent Task ID：${input.lineage.parentTaskId}`,
        `Revision Number：${input.lineage.revisionNumber}`,
        "",
        "本任务是用户确认后创建的修订任务。只处理本轮 findings，不要自动创建或发送下一轮任务。",
        ""
      ]
    : [];

  return [
    `你正在执行 WebGPT 生成的代码修改任务。`,
    "",
    `任务 ID：${input.taskId}`,
    `任务标题：${input.title}`,
    "",
    ...lineageText,
    "## 背景",
    "",
    input.reviewGoal,
    "",
    "## 本轮目标",
    "",
    "根据 WebGPT 的代码审查结果，在本地仓库中完成小步、可复核的代码修改。",
    "",
    "## 已审查文件",
    "",
    bulletList(input.reviewedFiles, "WebGPT 未显式提供已审查文件列表。请根据任务内容自行定位，但保持修改范围收敛。"),
    "",
    "## 允许修改的文件范围",
    "",
    bulletList(input.allowedFiles, "未显式限制到具体文件。请只修改完成本任务所必需的最小范围文件。"),
    "",
    "## 禁止事项",
    "",
    bulletList(
      [
        ...input.forbiddenActions,
        "不要扩大任务范围。",
        "不要重写无关模块。",
        "不要删除用户未要求删除的功能。",
        "不要自动确认任何权限请求；如遇权限提示，请停止并说明需要用户人工确认。"
      ],
      "不要扩大任务范围。"
    ),
    "",
    "## WebGPT 审查发现",
    "",
    numberedFindings(input.findings),
    "",
    input.extraContext ? ["## 额外上下文", "", input.extraContext, ""].join("\n") : "",
    "## 修改要求",
    "",
    "1. 按优先级处理上述 findings。",
    "2. 每次修改保持小步、局部、可解释。",
    "3. 保持现有项目风格和命名习惯。",
    "4. 如果发现 WebGPT 建议与实际代码不一致，请不要盲改；先说明差异并选择最小安全修改。",
    "",
    "## 测试要求",
    "",
    bulletList(
      input.testInstructions.length
        ? input.testInstructions
        : [
            "优先运行与修改范围相关的最小测试。",
            "如果项目支持 TypeScript 编译，请运行 npm run build 或等价命令。",
            "如果无法运行测试，请说明原因。"
          ],
      "无测试要求。"
    ),
    "",
    "## 完成后请汇报",
    "",
    "- 修改了哪些文件。",
    "- 每个文件为什么修改。",
    "- 是否运行了测试或构建命令。",
    "- 测试结果是什么。",
    "- 是否还有风险、阻塞或需要 WebGPT 继续复核的地方。",
    "",
    "重要：如果 Claude Code 出现权限确认提示，不要替用户确认，请停下来让用户人工处理。"
  ]
    .filter((part) => part !== "")
    .join("\n");
}

function buildExecutionTaskMarkdown(input: {
  taskId: string;
  target: string | null;
  title: string;
  executionInstructions: string;
  reviewPlanPath: string;
  jsonPath: string;
  resultMarkdownPath: string;
  resultJsonPath: string;
  verificationMarkdownPath: string;
  verificationJsonPath: string;
  createdAt: string;
  lineage: ReviewTaskLineage;
}): string {
  return [
    "# Claude Code Execution Task",
    "",
    `Task ID: ${input.taskId}`,
    `Created: ${input.createdAt}`,
    `Target: ${input.target ?? "(not selected)"}`,
    "Status: draft",
    `Root Task ID: ${input.lineage.rootTaskId}`,
    `Parent Task ID: ${input.lineage.parentTaskId ?? "(none)"}`,
    `Revision Number: ${input.lineage.revisionNumber}`,
    `Source Verification Signature: ${input.lineage.sourceVerificationSignature ?? "(none)"}`,
    `Review plan: ${input.reviewPlanPath}`,
    `Task JSON: ${input.jsonPath}`,
    `Execution result Markdown: ${input.resultMarkdownPath}`,
    `Execution result JSON: ${input.resultJsonPath}`,
    `WebGPT verification Markdown: ${input.verificationMarkdownPath}`,
    `WebGPT verification JSON: ${input.verificationJsonPath}`,
    "",
    "## Send Policy",
    "",
    "- Default submit: false",
    "- User confirmation required: true",
    "- Auto-send allowed: false",
    "- Auto-permission-confirm allowed: false",
    "- Auto-loop allowed: false",
    "",
    "## Execution Instructions",
    "",
    input.executionInstructions,
    ""
  ].join("\n");
}

function buildInitialExecutionResultMarkdown(input: {
  taskId: string;
  title: string;
  createdAt: string;
}): string {
  return [
    "# Claude Code Execution Result",
    "",
    `Task ID: ${input.taskId}`,
    `Task: ${input.title}`,
    "Status: pending",
    `Updated: ${input.createdAt}`,
    "",
    "This file was initialized by CodexPro.",
    "",
    "A Claude Code local result-reporting Skill should replace this",
    "placeholder after the task is completed, blocked, failed,",
    "partially completed, or waiting for user input.",
    ""
  ].join("\n");
}

function buildInitialExecutionResultPayload(input: {
  taskId: string;
  title: string;
  taskFilePath: string;
  createdAt: string;
}): Record<string, unknown> {
  return {
    schema_version: 1,
    kind: "claude_execution_result",

    task_id: input.taskId,
    task_title: input.title,
    task_file: input.taskFilePath,

    status: "pending",
    summary: "",

    changed_files: [],
    change_summary: [],
    tests: [],
    remaining_issues: [],
    risks: [],

    needs_input: null,

    started_at: null,
    completed_at: null,
    updated_at: input.createdAt,

    initialized_by: "codexpro"
  };
}

function buildJsonPayload(input: {
  taskId: string;
  target: string | null;
  title: string;
  reviewGoal: string;
  reviewedFiles: string[];
  findings: Required<ReviewExecutionFindingInput>[];
  allowedFiles: string[];
  forbiddenActions: string[];
  testInstructions: string[];
  extraContext?: string;

  promptMode: ReviewExecutionPromptMode;
  claudeSkill: string | null;
  taskFileForClaude: ReviewExecutionTaskFileForClaude;
  taskFilePath: string;
  promptForClaude: string;
  executionInstructions: string;

  files: ReviewExecutionTaskFiles;
  workspace: Workspace;
  createdAt: string;
  lineage: ReviewTaskLineage;
}): Record<string, unknown> {
  return {
    schema_version: 5,
    kind: "webgpt_review_execution_task",
    task_id: input.taskId,
    created_at: input.createdAt,
    workspace_id: input.workspace.id,
    workspace_root: input.workspace.root,
    target: input.target,
    title: input.title,
    status: "draft",

    lineage: {
      root_task_id: input.lineage.rootTaskId,
      parent_task_id: input.lineage.parentTaskId,
      revision_number: input.lineage.revisionNumber,
      source_verification_signature:
        input.lineage.sourceVerificationSignature,
      source_execution_result_digest:
        input.lineage.sourceExecutionResultDigest,
      source_verdict: input.lineage.sourceVerdict
    },

    review_goal: input.reviewGoal,
    reviewed_files: input.reviewedFiles,
    findings: input.findings,
    allowed_files: input.allowedFiles,
    forbidden_actions: input.forbiddenActions,
    test_instructions: input.testInstructions,
    extra_context: input.extraContext ?? null,

    prompt_mode: input.promptMode,
    
    // Backward-compatible field from Review Handoff v2.
    // It is only a recommendation and is never automatically used by dispatch.
    claude_skill: input.claudeSkill,
    
    recommended_skill: input.claudeSkill,
    
    task_file_for_claude: input.taskFileForClaude,
    task_file_path: input.taskFilePath,
    
    handoff: {
      task_file: input.taskFilePath,
      recommended_skill: input.claudeSkill
    },

    /**
     * This is what WebGPT should send to Claude Code after user confirmation.
     * It may be a short file-reference prompt, a direct slash command, or a full inline task.
     */
    prompt_for_claude: input.promptForClaude,

    /**
     * This is the full execution task content saved in claude-execution-task.md.
     */
    execution_instructions: input.executionInstructions,

    files: {
      review_plan_markdown: input.files.reviewPlanMarkdown,
      execution_task_markdown: input.files.executionTaskMarkdown,
      execution_task_json: input.files.executionTaskJson,
    
      execution_result_markdown: input.files.executionResultMarkdown,
      execution_result_json: input.files.executionResultJson,

      verification_result_markdown: input.files.verificationResultMarkdown,
      verification_result_json: input.files.verificationResultJson,
    
      history_jsonl: input.files.historyJsonl
    },

    result_contract: {
      schema_version: 1,
      producer: "claude_code_local_skill",
      consumer_tool: "inspect_review_execution_result",
      result_markdown: input.files.executionResultMarkdown,
      result_json: input.files.executionResultJson
    },

    verification_contract: {
      schema_version: 1,
      producer: "webgpt",
      consumer_tool: "record_review_verification",
      verification_markdown: input.files.verificationResultMarkdown,
      verification_json: input.files.verificationResultJson,
      verdicts: ["accepted", "revision_required", "blocked"],
      requires_execution_result_digest: true
    },
    
    lifecycle: {
      status: "draft",
      created_at: input.createdAt,
      updated_at: input.createdAt,
      last_dispatch: null
    },
    
    safety: {
      requires_user_confirmation: true,
      submit_default: false,
      auto_send_allowed: false,
      auto_submit_allowed: false,
      auto_permission_confirm_allowed: false,
      auto_loop_allowed: false
    }
  };
}

async function appendHistory(
  guard: PathGuard,
  workspace: Workspace,
  historyPath: string,
  eventName: string,
  data: Record<string, unknown>
): Promise<void> {
  const resolved = guard.resolve(workspace, historyPath, { forWrite: true });

  await fsp.mkdir(path.dirname(resolved.absPath), {
    recursive: true
  });

  await fsp.appendFile(
    resolved.absPath,
    `${JSON.stringify({
      ts: new Date().toISOString(),
      event: eventName,
      ...data
    })}\n`,
    "utf8"
  );
}

function objectValue(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }

  return value as Record<string, unknown>;
}

function requiredStoredString(
  value: unknown,
  fieldName: string,
  maxLength = 100_000
): string {
  const text = String(value ?? "").trim();

  if (!text) {
    throw new CodexProError(
      `Invalid review execution task: ${fieldName} is missing or empty.`
    );
  }

  if (text.includes("\0")) {
    throw new CodexProError(
      `Invalid review execution task: ${fieldName} contains a NUL byte.`
    );
  }

  if (Buffer.byteLength(text, "utf8") > maxLength) {
    throw new CodexProError(
      `Invalid review execution task: ${fieldName} is too large.`
    );
  }

  return text;
}

function optionalStoredString(
  value: unknown,
  maxLength = 100_000
): string | null {
  if (value === undefined || value === null) return null;

  const text = String(value).trim();
  if (!text) return null;

  if (text.includes("\0")) {
    throw new CodexProError(
      "Invalid review execution task: stored text contains a NUL byte."
    );
  }

  if (Buffer.byteLength(text, "utf8") > maxLength) {
    throw new CodexProError(
      "Invalid review execution task: stored text is too large."
    );
  }

  return text;
}

function normalizeStoredLifecycleStatus(
  value: unknown
): ReviewExecutionTaskLifecycleStatus {
  if (value === undefined || value === null || value === "draft") {
    return "draft";
  }

  if (value === "pasted") return "pasted";
  if (value === "submitted") return "submitted";
  if (value === "dispatch_failed") return "dispatch_failed";

  throw new CodexProError(
    `Invalid review execution task lifecycle status: ${String(value)}`
  );
}

function normalizeStoredPromptMode(
  value: unknown
): ReviewExecutionPromptMode {
  if (value === "inline") return "inline";
  if (value === "skill_file_reference") return "skill_file_reference";

  return "file_reference";
}

function normalizeDispatchMode(
  value: unknown
): ReviewExecutionDispatchMode {
  return value === "skill" ? "skill" : "text";
}

function normalizeAiBridgeRelativePath(
  config: CodexProConfig,
  value: unknown,
  fieldName: string
): string {
  const raw = requiredStoredString(value, fieldName, 2_000)
    .replace(/\\/g, "/")
    .replace(/^\.\/+/, "")
    .replace(/\/+/g, "/");

  if (
    raw === ".." ||
    raw.startsWith("../") ||
    raw.includes("/../") ||
    raw.startsWith("/")
  ) {
    throw new CodexProError(
      `Invalid review execution task: ${fieldName} must be a relative path inside ${config.contextDir}/.`
    );
  }

  const contextDir = config.contextDir
    .replace(/\\/g, "/")
    .replace(/^\.\/+/, "")
    .replace(/\/+$/, "");

  if (raw !== contextDir && !raw.startsWith(`${contextDir}/`)) {
    throw new CodexProError(
      `Invalid review execution task: ${fieldName} must be inside ${contextDir}/.`
    );
  }

  return raw;
}

function errorMessage(error: unknown): string {
  const text = error instanceof Error ? error.message : String(error);

  return text
    .replace(/\0/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 2_000);
}

async function assertTaskFileExists(
  config: CodexProConfig,
  guard: PathGuard,
  workspace: Workspace,
  taskFilePath: string
): Promise<void> {
  const resolved = guard.resolve(workspace, taskFilePath);

  await guard.assertTextFile(
    resolved.absPath,
    Math.min(config.maxReadBytes, 1_000_000)
  );
}

function storedStringList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];

  return [
    ...new Set(
      value
        .map((item) => String(item ?? "").trim())
        .filter(Boolean)
    )
  ];
}

function storedFindings(
  value: unknown
): ReviewExecutionFindingInput[] {
  if (!Array.isArray(value)) return [];

  const out: ReviewExecutionFindingInput[] = [];

  for (const item of value) {
    if (
      !item ||
      typeof item !== "object" ||
      Array.isArray(item)
    ) {
      continue;
    }

    const record = item as Record<string, unknown>;

    const issue = String(
      record.issue ?? ""
    ).trim();

    const recommendation = String(
      record.recommendation ?? ""
    ).trim();

    if (!issue || !recommendation) {
      continue;
    }

    out.push({
      file:
        typeof record.file === "string" &&
        record.file.trim()
          ? record.file.trim()
          : undefined,

      issue,
      recommendation,

      priority:
        record.priority === "low" ||
        record.priority === "medium" ||
        record.priority === "high"
          ? record.priority
          : "medium",

      risk:
        typeof record.risk === "string" &&
        record.risk.trim()
          ? record.risk.trim()
          : undefined
    });
  }

  return out;
}

export async function readStoredReviewExecutionTask(
  config: CodexProConfig,
  guard: PathGuard,
  workspace: Workspace
): Promise<StoredReviewExecutionTask> {
  const jsonPath =
    `${config.contextDir}/claude-execution-task.json`;

  const resolved = guard.resolve(
    workspace,
    jsonPath
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
    throw new CodexProError(
      `Invalid JSON in ${jsonPath}: ${errorMessage(error)}`
    );
  }

  const payload = objectValue(parsed);

  if (
    payload.kind !==
    "webgpt_review_execution_task"
  ) {
    throw new CodexProError(
      `Invalid review execution task kind in ${jsonPath}.`
    );
  }

  const schemaVersion = Number(
    payload.schema_version ?? 1
  );

  /*
   * 第三轮将任务 schema 升级到 3。
   * 第四轮新增 Verification 文件与契约，将 schema 升级到 4。
   */
  if (
    !Number.isInteger(schemaVersion) ||
    schemaVersion < 1 ||
    schemaVersion > 5
  ) {
    throw new CodexProError(
      `Unsupported review execution task schema_version: ${String(
        payload.schema_version
      )}`
    );
  }

  const taskId = requiredStoredString(
    payload.task_id,
    "task_id",
    200
  );

  const title =
    optionalStoredString(
      payload.title,
      500
    ) ??
    "Review Execution Task";

  const lifecycle = objectValue(
    payload.lifecycle
  );

  const status =
    normalizeStoredLifecycleStatus(
      lifecycle.status ?? payload.status
    );

  const lineage = normalizeLineage(
    payload.lineage,
    taskId
  );

  const target = optionalStoredString(
    payload.target,
    80
  );

  const promptMode =
    normalizeStoredPromptMode(
      payload.prompt_mode
    );

  const promptForClaude =
    optionalStoredString(
      payload.prompt_for_claude,
      200_000
    ) ?? "";

  const executionInstructions =
    optionalStoredString(
      payload.execution_instructions,
      200_000
    ) ?? "";

  const files = objectValue(
    payload.files
  );

  const handoff = objectValue(
    payload.handoff
  );

  /*
   * Claude Code 实际读取的任务文件。
   */
  const taskFilePath =
    normalizeAiBridgeRelativePath(
      config,
      payload.task_file_path ??
        handoff.task_file ??
        files.execution_task_markdown,
      "task_file_path"
    );

  /*
   * 创建任务时记录的推荐 Skill。
   *
   * 注意：
   * dispatch 时不会自动使用它。
   * 实际 Skill 仍由 dispatch 请求指定。
   */
  const recommendedSkill =
    normalizeClaudeSkill(
      payload.recommended_skill ??
        handoff.recommended_skill ??
        payload.claude_skill
    );

  /*
   * 第三轮新增：Claude Code 结构化结果文件路径。
   */
  const resultMarkdownPath =
    normalizeAiBridgeRelativePath(
      config,
      files.execution_result_markdown ??
        `${config.contextDir}/claude-execution-result.md`,
      "files.execution_result_markdown"
    );

  const resultJsonPath =
    normalizeAiBridgeRelativePath(
      config,
      files.execution_result_json ??
        `${config.contextDir}/claude-execution-result.json`,
      "files.execution_result_json"
    );

  const verificationMarkdownPath =
    normalizeAiBridgeRelativePath(
      config,
      files.verification_result_markdown ??
        `${config.contextDir}/webgpt-verification-result.md`,
      "files.verification_result_markdown"
    );

  const verificationJsonPath =
    normalizeAiBridgeRelativePath(
      config,
      files.verification_result_json ??
        `${config.contextDir}/webgpt-verification-result.json`,
      "files.verification_result_json"
    );

  const historyPath =
    normalizeAiBridgeRelativePath(
      config,
      files.history_jsonl ??
        `${config.contextDir}/review-task-history.jsonl`,
      "files.history_jsonl"
    );

  /*
   * 第三轮新增：恢复原始 WebGPT 审查上下文。
   *
   * inspect_review_execution_result 会把这些内容返回给
   * 网页端 GPT，供它复核实际代码。
   */
  const reviewGoal =
    optionalStoredString(
      payload.review_goal,
      20_000
    ) ?? "";

  const reviewedFiles =
    storedStringList(
      payload.reviewed_files
    );

  const findings =
    storedFindings(
      payload.findings
    );

  const allowedFiles =
    storedStringList(
      payload.allowed_files
    );

  const forbiddenActions =
    storedStringList(
      payload.forbidden_actions
    );

  const testInstructions =
    storedStringList(
      payload.test_instructions
    );

  /*
   * 任务文件必须存在，否则无法正常 dispatch。
   *
   * 结果文件不在这里强制要求存在：
   * inspect_review_execution_result 会区分
   * missing / invalid / pending 等状态。
   */
  await assertTaskFileExists(
    config,
    guard,
    workspace,
    taskFilePath
  );

  return {
    jsonPath,
    payload,

    schemaVersion,
    taskId,
    title,
    status,
    lineage,

    target,
    promptMode,

    promptForClaude,
    executionInstructions,

    taskFilePath,
    recommendedSkill,

    reviewGoal,
    reviewedFiles,
    findings,
    allowedFiles,
    forbiddenActions,
    testInstructions,

    resultMarkdownPath,
    resultJsonPath,

    verificationMarkdownPath,
    verificationJsonPath,

    historyPath
  };
}

function resolveDispatchTarget(
  requestedTarget: string | undefined,
  storedTarget: string | null
): string {
  const candidate =
    cleanOptionalOneLine(requestedTarget, "target", 80) ??
    storedTarget;

  if (!candidate) {
    const available =
      listClaudeCodeTargets()
        .map((item) => item.name)
        .join(", ") || "none";

    throw new CodexProError(
      `No Claude Code target was selected. ` +
        `Pass target explicitly or create the task with a target. ` +
        `Allowed targets: ${available}`
    );
  }

  const target = validateTarget(candidate);

  if (!target) {
    throw new CodexProError(
      "Unable to resolve Claude Code target."
    );
  }

  return target;
}

function textPromptForStoredTask(
  task: StoredReviewExecutionTask
): string {
  /*
   * A task created in skill_file_reference mode may contain a suggested
   * slash command in prompt_for_claude. A text dispatch must not
   * accidentally use that suggested Skill.
   */
  if (
    task.promptMode !== "skill_file_reference" &&
    task.promptForClaude
  ) {
    return task.promptForClaude;
  }

  if (task.promptMode === "inline" && task.executionInstructions) {
    return task.executionInstructions;
  }

  return buildFileReferencePrompt(task.taskFilePath);
}

function updatedPayloadWithDispatch(
  task: StoredReviewExecutionTask,
  status: ReviewExecutionTaskLifecycleStatus,
  lastDispatch: Record<string, unknown>
): Record<string, unknown> {
  const now = new Date().toISOString();
  const oldLifecycle = objectValue(task.payload.lifecycle);

  return {
    ...task.payload,

    status,

    lifecycle: {
      ...oldLifecycle,
      status,
      updated_at: now,
      last_dispatch: lastDispatch
    },

    last_dispatch: lastDispatch
  };
}

async function writeStoredReviewExecutionTask(
  config: CodexProConfig,
  guard: PathGuard,
  workspace: Workspace,
  task: StoredReviewExecutionTask,
  payload: Record<string, unknown>
): Promise<{
  path: string;
  bytes: number;
  additions: number;
  deletions: number;
}> {
  const result = await writeTextFile(
    config,
    guard,
    workspace,
    task.jsonPath,
    `${JSON.stringify(payload, null, 2)}\n`,
    {
      createDirs: true,
      overwrite: true
    }
  );

  return {
    path: result.path,
    bytes: result.bytes,
    additions: result.diff.additions,
    deletions: result.diff.deletions
  };
}


export async function createReviewExecutionTask(
  config: CodexProConfig,
  guard: PathGuard,
  workspace: Workspace,
  rawInput: CreateReviewExecutionTaskInput
): Promise<CreateReviewExecutionTaskResult> {
  if (config.writeMode === "off") {
    throw new CodexProError(
      "create_review_execution_task requires write mode handoff or workspace because it writes .ai-bridge task artifacts."
    );
  }

  await ensureAiBridge(config, guard, workspace);

  const createdAt = new Date().toISOString();
  const taskId = rawInput.taskId
    ? normalizeTaskId(rawInput.taskId, "taskId")
    : `review_${createdAt.replace(/[-:.TZ]/g, "").slice(0, 14)}_${randomUUID().slice(0, 8)}`;
  const lineage = normalizeLineage(rawInput.lineage, taskId);

  const target = validateTarget(rawInput.target);
  const title = cleanOneLine(rawInput.title, "title", MAX_TITLE_LENGTH);
  const reviewGoal = cleanText(rawInput.reviewGoal, "reviewGoal", 8_000);
  const reviewedFiles = cleanStringList(rawInput.reviewedFiles, "reviewedFiles");
  const findings = normalizeFindings(rawInput.findings);
  const allowedFiles = cleanStringList(rawInput.allowedFiles, "allowedFiles");
  const forbiddenActions = cleanStringList(rawInput.forbiddenActions, "forbiddenActions");
  const testInstructions = cleanStringList(rawInput.testInstructions, "testInstructions", 80);
  const extraContext = cleanOptionalText(rawInput.extraContext, "extraContext", 10_000);

  const promptMode = normalizePromptMode(rawInput.promptMode);
  const claudeSkill = normalizeClaudeSkill(rawInput.claudeSkill);
  const taskFileForClaude = normalizeTaskFileForClaude(rawInput.taskFileForClaude);

  if (promptMode === "skill_file_reference" && !claudeSkill) {
    throw new CodexProError(
      "claudeSkill is required when promptMode is skill_file_reference."
    );
  }

  const files: ReviewExecutionTaskFiles = {
    reviewPlanMarkdown: `${config.contextDir}/webgpt-review-plan.md`,
    executionTaskMarkdown: `${config.contextDir}/claude-execution-task.md`,
    executionTaskJson: `${config.contextDir}/claude-execution-task.json`,
  
    executionResultMarkdown: `${config.contextDir}/claude-execution-result.md`,
    executionResultJson: `${config.contextDir}/claude-execution-result.json`,

    verificationResultMarkdown: `${config.contextDir}/webgpt-verification-result.md`,
    verificationResultJson: `${config.contextDir}/webgpt-verification-result.json`,
  
    historyJsonl: `${config.contextDir}/review-task-history.jsonl`
  };

  const taskFilePath = taskFilePathForClaude(files, taskFileForClaude);

  const reviewPlanMarkdown = buildReviewPlanMarkdown({
    taskId,
    target,
    title,
    reviewGoal,
    reviewedFiles,
    findings,
    allowedFiles,
    forbiddenActions,
    testInstructions,
    extraContext,
    workspace,
    createdAt,
    lineage
  });

  const executionInstructions = buildPromptForClaude({
    taskId,
    title,
    reviewGoal,
    reviewedFiles,
    findings,
    allowedFiles,
    forbiddenActions,
    testInstructions,
    extraContext,
    lineage
  });

  const promptForClaude = buildPromptForClaudeByMode({
    promptMode,
    claudeSkill,
    taskFilePath,
    executionInstructions
  });

  const executionTaskMarkdown = buildExecutionTaskMarkdown({
    taskId,
    target,
    title,
    executionInstructions,
  
    reviewPlanPath: files.reviewPlanMarkdown,
    jsonPath: files.executionTaskJson,
  
    resultMarkdownPath: files.executionResultMarkdown,
    resultJsonPath: files.executionResultJson,
    verificationMarkdownPath: files.verificationResultMarkdown,
    verificationJsonPath: files.verificationResultJson,
  
    createdAt,
    lineage
  });

  const initialExecutionResultMarkdown =
    buildInitialExecutionResultMarkdown({
      taskId,
      title,
      createdAt
    });
  
  const initialExecutionResultPayload =
    buildInitialExecutionResultPayload({
      taskId,
      title,
      taskFilePath,
      createdAt
    });
  
  const jsonPayload = buildJsonPayload({
    taskId,
    target,
    title,
    reviewGoal,
    reviewedFiles,
    findings,
    allowedFiles,
    forbiddenActions,
    testInstructions,
    extraContext,

    promptMode,
    claudeSkill,
    taskFileForClaude,
    taskFilePath,
    promptForClaude,
    executionInstructions,

    files,
    workspace,
    createdAt,
    lineage
  });

  const reviewWrite = await writeTextFile(
    config,
    guard,
    workspace,
    files.reviewPlanMarkdown,
    reviewPlanMarkdown,
    {
      createDirs: true,
      overwrite: true
    }
  );

  const executionWrite = await writeTextFile(
    config,
    guard,
    workspace,
    files.executionTaskMarkdown,
    executionTaskMarkdown,
    {
      createDirs: true,
      overwrite: true
    }
  );

  const jsonWrite = await writeTextFile(
    config,
    guard,
    workspace,
    files.executionTaskJson,
    `${JSON.stringify(jsonPayload, null, 2)}\n`,
    {
      createDirs: true,
      overwrite: true
    }
  );

  const resultMarkdownWrite = await writeTextFile(
    config,
    guard,
    workspace,
    files.executionResultMarkdown,
    initialExecutionResultMarkdown,
    {
      createDirs: true,
      overwrite: true
    }
  );
  
  const resultJsonWrite = await writeTextFile(
    config,
    guard,
    workspace,
    files.executionResultJson,
    `${JSON.stringify(initialExecutionResultPayload, null, 2)}\n`,
    {
      createDirs: true,
      overwrite: true
    }
  );

  for (const verificationPath of [
    files.verificationResultMarkdown,
    files.verificationResultJson
  ]) {
    const resolvedVerification = guard.resolve(
      workspace,
      verificationPath,
      { forWrite: true }
    );

    try {
      await fsp.unlink(resolvedVerification.absPath);
    } catch (error) {
      const code =
        error && typeof error === "object" && "code" in error
          ? String((error as { code?: unknown }).code)
          : "";

      if (code !== "ENOENT") throw error;
    }
  }
  
  await appendHistory(
    guard,
    workspace,
    files.historyJsonl,
    "create_review_execution_task",
    {
      task_id: taskId,
      target,
      title,
      prompt_mode: promptMode,
      recommended_skill: claudeSkill,
      task_file_for_claude: taskFileForClaude,
      task_file_path: taskFilePath,
      review_plan_markdown: files.reviewPlanMarkdown,
      execution_task_markdown: files.executionTaskMarkdown,
      execution_task_json: files.executionTaskJson,
      execution_result_markdown: files.executionResultMarkdown,
      execution_result_json: files.executionResultJson,
      verification_result_markdown: files.verificationResultMarkdown,
      verification_result_json: files.verificationResultJson,
      root_task_id: lineage.rootTaskId,
      parent_task_id: lineage.parentTaskId,
      revision_number: lineage.revisionNumber,
      source_verification_signature:
        lineage.sourceVerificationSignature,
      source_execution_result_digest:
        lineage.sourceExecutionResultDigest
    }
  );

  await appendHistory(
    guard,
    workspace,
    files.historyJsonl,
    "execution_result_initialized",
    {
      task_id: taskId,
      status: "pending",
      execution_result_markdown: files.executionResultMarkdown,
      execution_result_json: files.executionResultJson
    }
  );
  
  return {
    taskId,
    target,
    title,
    status: "draft",
    lineage,

    promptMode,
    claudeSkill,
    taskFileForClaude,
    taskFilePath,

    files,
    promptForClaude,
    executionInstructions,
    reviewPlanMarkdown,
    executionTaskMarkdown,
    jsonPayload,

    writes: {
      reviewPlanMarkdown: {
        path: reviewWrite.path,
        bytes: reviewWrite.bytes,
        additions: reviewWrite.diff.additions,
        deletions: reviewWrite.diff.deletions
      },
      executionTaskMarkdown: {
        path: executionWrite.path,
        bytes: executionWrite.bytes,
        additions: executionWrite.diff.additions,
        deletions: executionWrite.diff.deletions
      },
      executionTaskJson: {
        path: jsonWrite.path,
        bytes: jsonWrite.bytes,
        additions: jsonWrite.diff.additions,
        deletions: jsonWrite.diff.deletions
      },
      executionResultMarkdown: {
        path: resultMarkdownWrite.path,
        bytes: resultMarkdownWrite.bytes,
        additions: resultMarkdownWrite.diff.additions,
        deletions: resultMarkdownWrite.diff.deletions
      },
      
      executionResultJson: {
        path: resultJsonWrite.path,
        bytes: resultJsonWrite.bytes,
        additions: resultJsonWrite.diff.additions,
        deletions: resultJsonWrite.diff.deletions
      }
    },

    safety: {
      requiresUserConfirmation: true,
      submitDefault: false,
      autoSendAllowed: false,
      autoSubmitAllowed: false,
      autoPermissionConfirmAllowed: false,
      autoLoopAllowed: false
    }
  };
}

export async function dispatchReviewExecutionTask(
  config: CodexProConfig,
  guard: PathGuard,
  workspace: Workspace,
  rawInput: DispatchReviewExecutionTaskInput
): Promise<DispatchReviewExecutionTaskResult> {
  if (config.writeMode === "off") {
    throw new CodexProError(
      "dispatch_review_execution_task requires write mode handoff or workspace " +
        "because every dispatch must update the task audit files."
    );
  }

  await ensureAiBridge(config, guard, workspace);

  const task = await readStoredReviewExecutionTask(
    config,
    guard,
    workspace
  );

  const requestedTaskId = cleanOneLine(
    rawInput.taskId,
    "taskId",
    200
  );

  if (requestedTaskId !== task.taskId) {
    throw new CodexProError(
      `taskId does not match the current review execution task. ` +
        `Expected ${task.taskId}, received ${requestedTaskId}.`
    );
  }

  const target = resolveDispatchTarget(
    rawInput.target,
    task.target
  );

  const dispatchMode = normalizeDispatchMode(
    rawInput.dispatchMode
  );

  const confirmed = rawInput.confirmed === true;
  const submit = rawInput.submit === true;

  let skill: string | null = null;
  let prompt: string;

  if (dispatchMode === "skill") {
    skill = normalizeClaudeSkill(rawInput.skill);

    if (!skill) {
      throw new CodexProError(
        "skill is required when dispatchMode is skill."
      );
    }

    /*
     * This validates the Skill and creates exactly the same command
     * that invokeClaudeCodeSkill will send.
     */
    prompt = buildClaudeCodeSkillCommand(
      skill,
      task.taskFilePath
    );
  } else {
    prompt = textPromptForStoredTask(task);
  }

  if (!prompt.trim()) {
    throw new CodexProError(
      "The selected review execution task does not contain a dispatchable prompt."
    );
  }

  const dispatchId =
    `dispatch_${new Date()
      .toISOString()
      .replace(/[-:.TZ]/g, "")
      .slice(0, 14)}_${randomUUID().slice(0, 8)}`;

  /*
   * Preview mode:
   * - no sendToClaudeCode
   * - no invokeClaudeCodeSkill
   * - no tmux interaction
   * - only an audit history entry is appended
   */
  if (!confirmed) {
    await appendHistory(
      guard,
      workspace,
      task.historyPath,
      "dispatch_preview",
      {
        dispatch_id: dispatchId,
        task_id: task.taskId,
        target,
        dispatch_mode: dispatchMode,
        skill,
        recommended_skill: task.recommendedSkill,
        task_file_path: task.taskFilePath,
        submit,
        sent: false,
        status: task.status
      }
    );

    return {
      dispatchId,
      taskId: task.taskId,
      title: task.title,
      target,

      dispatchMode,
      skill,
      recommendedSkill: task.recommendedSkill,

      taskFilePath: task.taskFilePath,
      prompt,

      confirmed: false,
      submit,
      sent: false,

      statusBefore: task.status,
      statusAfter: task.status,

      bridgeResult: null,

      auditUpdated: true,

      safety: {
        requiredExplicitConfirmation: true,
        submitDefault: false,
        permissionAutoConfirmAllowed: false,
        automaticLoopAllowed: false
      }
    };
  }

  /*
   * Prevent accidental duplicate dispatch.
   *
   * A task that was pasted or submitted may only be sent again by
   * creating a new review execution task.
   */
  if (
    task.status === "pasted" ||
    task.status === "submitted"
  ) {
    throw new CodexProError(
      `Task ${task.taskId} has already been dispatched with status ${task.status}. ` +
        "Create a new review execution task before sending another instruction."
    );
  }

  const dispatchedAt = new Date().toISOString();

  try {
    const bridgeResult:
      | ClaudeCodeSendResult
      | ClaudeCodeSkillResult =
      dispatchMode === "skill"
        ? await invokeClaudeCodeSkill({
            target,
            skill: skill!,
            arguments: task.taskFilePath,
            submit
          })
        : await sendToClaudeCode({
            target,
            text: prompt,
            submit
          });

    const statusAfter: ReviewExecutionTaskLifecycleStatus =
      submit ? "submitted" : "pasted";

    const successEvent = submit
      ? "dispatch_submitted"
      : "dispatch_pasted";

    const lastDispatch: Record<string, unknown> = {
      dispatch_id: dispatchId,
      dispatched_at: dispatchedAt,
      result: "success",

      target,
      tmux_target: bridgeResult.tmuxTarget,

      dispatch_mode: dispatchMode,
      skill,

      /*
       * Only informational. It was not used automatically.
       */
      recommended_skill: task.recommendedSkill,

      task_file_path: task.taskFilePath,

      submit,
      submitted: bridgeResult.submitted,
      bytes: bridgeResult.bytes,

      status_before: task.status,
      status_after: statusAfter
    };

    let auditUpdated = true;
    let auditError: string | undefined;
    let taskWrite:
      | {
          path: string;
          bytes: number;
          additions: number;
          deletions: number;
        }
      | undefined;

    /*
     * Sending has already succeeded at this point.
     * Audit failure must not pretend that nothing was sent.
     */
    try {
      const updatedPayload = updatedPayloadWithDispatch(
        task,
        statusAfter,
        lastDispatch
      );

      taskWrite = await writeStoredReviewExecutionTask(
        config,
        guard,
        workspace,
        task,
        updatedPayload
      );

      await appendHistory(
        guard,
        workspace,
        task.historyPath,
        successEvent,
        {
          ...lastDispatch,
          task_id: task.taskId,
          title: task.title
        }
      );
    } catch (error) {
      auditUpdated = false;
      auditError = errorMessage(error);
    }

    return {
      dispatchId,
      taskId: task.taskId,
      title: task.title,
      target,

      dispatchMode,
      skill,
      recommendedSkill: task.recommendedSkill,

      taskFilePath: task.taskFilePath,
      prompt,

      confirmed: true,
      submit,
      sent: true,

      statusBefore: task.status,
      statusAfter,

      bridgeResult,

      auditUpdated,
      auditError,
      taskWrite,

      safety: {
        requiredExplicitConfirmation: true,
        submitDefault: false,
        permissionAutoConfirmAllowed: false,
        automaticLoopAllowed: false
      }
    };
  } catch (error) {
    const dispatchError = errorMessage(error);

    const failedAt = new Date().toISOString();

    const failedDispatch: Record<string, unknown> = {
      dispatch_id: dispatchId,
      dispatched_at: failedAt,
      result: "failed",

      target,
      dispatch_mode: dispatchMode,
      skill,
      recommended_skill: task.recommendedSkill,

      task_file_path: task.taskFilePath,

      submit,
      error: dispatchError,

      status_before: task.status,
      status_after: "dispatch_failed"
    };

    let auditFailure: string | undefined;

    try {
      const failedPayload = updatedPayloadWithDispatch(
        task,
        "dispatch_failed",
        failedDispatch
      );

      await writeStoredReviewExecutionTask(
        config,
        guard,
        workspace,
        task,
        failedPayload
      );

      await appendHistory(
        guard,
        workspace,
        task.historyPath,
        "dispatch_failed",
        {
          ...failedDispatch,
          task_id: task.taskId,
          title: task.title
        }
      );
    } catch (auditError) {
      auditFailure = errorMessage(auditError);
    }

    throw new CodexProError(
      `Review execution task dispatch failed: ${dispatchError}` +
        (auditFailure
          ? ` Audit update also failed: ${auditFailure}`
          : "")
    );
  }
}
