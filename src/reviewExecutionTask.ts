import fsp from "node:fs/promises";
import path from "node:path";
import { randomUUID } from "node:crypto";

import type { CodexProConfig } from "./config.js";
import { CodexProError, PathGuard, type Workspace } from "./guard.js";
import { ensureAiBridge, writeTextFile } from "./fsOps.js";
import { listClaudeCodeTargets } from "./claudeCodeBridge.js";

export type ReviewFindingPriority = "low" | "medium" | "high";

export interface ReviewExecutionFindingInput {
  file?: string;
  issue: string;
  recommendation: string;
  priority?: ReviewFindingPriority;
  risk?: string;
}

export interface CreateReviewExecutionTaskInput {
  target?: string;
  title: string;
  reviewGoal: string;
  reviewedFiles?: string[];
  findings: ReviewExecutionFindingInput[];
  allowedFiles?: string[];
  forbiddenActions?: string[];
  testInstructions?: string[];
  extraContext?: string;
}

export interface ReviewExecutionTaskFiles {
  reviewPlanMarkdown: string;
  executionTaskMarkdown: string;
  executionTaskJson: string;
  historyJsonl: string;
}

export interface CreateReviewExecutionTaskResult {
  taskId: string;
  target: string | null;
  title: string;
  status: "draft";
  files: ReviewExecutionTaskFiles;
  promptForClaude: string;
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

function normalizePriority(value: unknown): ReviewFindingPriority {
  if (value === "low" || value === "medium" || value === "high") return value;
  return "medium";
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
}): string {
  return [
    `# WebGPT Review Plan`,
    "",
    `Task ID: ${input.taskId}`,
    `Created: ${input.createdAt}`,
    `Workspace: ${input.workspace.root}`,
    `Claude Code target: ${input.target ?? "(not selected)"}`,
    `Status: draft`,
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
}): string {
  return [
    `你正在执行 WebGPT 生成的代码修改任务。`,
    "",
    `任务 ID：${input.taskId}`,
    `任务标题：${input.title}`,
    "",
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
  promptForClaude: string;
  reviewPlanPath: string;
  jsonPath: string;
  createdAt: string;
}): string {
  return [
    `# Claude Code Execution Task`,
    "",
    `Task ID: ${input.taskId}`,
    `Created: ${input.createdAt}`,
    `Target: ${input.target ?? "(not selected)"}`,
    `Status: draft`,
    `Review plan: ${input.reviewPlanPath}`,
    `JSON: ${input.jsonPath}`,
    "",
    "## Send Policy",
    "",
    "- Default submit: false",
    "- User confirmation required: true",
    "- Auto-send allowed: false",
    "- Auto-permission-confirm allowed: false",
    "- Auto-loop allowed: false",
    "",
    "## Prompt For Claude Code",
    "",
    "```text",
    input.promptForClaude,
    "```",
    ""
  ].join("\n");
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
  promptForClaude: string;
  files: ReviewExecutionTaskFiles;
  workspace: Workspace;
  createdAt: string;
}): Record<string, unknown> {
  return {
    schema_version: 1,
    kind: "webgpt_review_execution_task",
    task_id: input.taskId,
    created_at: input.createdAt,
    workspace_id: input.workspace.id,
    workspace_root: input.workspace.root,
    target: input.target,
    title: input.title,
    status: "draft",
    review_goal: input.reviewGoal,
    reviewed_files: input.reviewedFiles,
    findings: input.findings,
    allowed_files: input.allowedFiles,
    forbidden_actions: input.forbiddenActions,
    test_instructions: input.testInstructions,
    extra_context: input.extraContext ?? null,
    prompt_for_claude: input.promptForClaude,
    files: {
      review_plan_markdown: input.files.reviewPlanMarkdown,
      execution_task_markdown: input.files.executionTaskMarkdown,
      execution_task_json: input.files.executionTaskJson,
      history_jsonl: input.files.historyJsonl
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
  event: Record<string, unknown>
): Promise<void> {
  const resolved = guard.resolve(workspace, historyPath, { forWrite: true });
  await fsp.mkdir(path.dirname(resolved.absPath), { recursive: true });
  await fsp.appendFile(
    resolved.absPath,
    `${JSON.stringify({ ts: new Date().toISOString(), event: "create_review_execution_task", ...event })}\n`,
    "utf8"
  );
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
  const taskId = `review_${createdAt.replace(/[-:.TZ]/g, "").slice(0, 14)}_${randomUUID().slice(0, 8)}`;

  const target = validateTarget(rawInput.target);
  const title = cleanOneLine(rawInput.title, "title", MAX_TITLE_LENGTH);
  const reviewGoal = cleanText(rawInput.reviewGoal, "reviewGoal", 8_000);
  const reviewedFiles = cleanStringList(rawInput.reviewedFiles, "reviewedFiles");
  const findings = normalizeFindings(rawInput.findings);
  const allowedFiles = cleanStringList(rawInput.allowedFiles, "allowedFiles");
  const forbiddenActions = cleanStringList(rawInput.forbiddenActions, "forbiddenActions");
  const testInstructions = cleanStringList(rawInput.testInstructions, "testInstructions", 80);
  const extraContext = cleanOptionalText(rawInput.extraContext, "extraContext", 10_000);

  const files: ReviewExecutionTaskFiles = {
    reviewPlanMarkdown: `${config.contextDir}/webgpt-review-plan.md`,
    executionTaskMarkdown: `${config.contextDir}/claude-execution-task.md`,
    executionTaskJson: `${config.contextDir}/claude-execution-task.json`,
    historyJsonl: `${config.contextDir}/review-task-history.jsonl`
  };

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
    createdAt
  });

  const promptForClaude = buildPromptForClaude({
    taskId,
    title,
    reviewGoal,
    reviewedFiles,
    findings,
    allowedFiles,
    forbiddenActions,
    testInstructions,
    extraContext
  });

  const executionTaskMarkdown = buildExecutionTaskMarkdown({
    taskId,
    target,
    title,
    promptForClaude,
    reviewPlanPath: files.reviewPlanMarkdown,
    jsonPath: files.executionTaskJson,
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
    promptForClaude,
    files,
    workspace,
    createdAt
  });

  const reviewWrite = await writeTextFile(config, guard, workspace, files.reviewPlanMarkdown, reviewPlanMarkdown, {
    createDirs: true,
    overwrite: true
  });

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

  const jsonWrite = await writeTextFile(config, guard, workspace, files.executionTaskJson, `${JSON.stringify(jsonPayload, null, 2)}\n`, {
    createDirs: true,
    overwrite: true
  });

  await appendHistory(guard, workspace, files.historyJsonl, {
    task_id: taskId,
    target,
    title,
    review_plan_markdown: files.reviewPlanMarkdown,
    execution_task_markdown: files.executionTaskMarkdown,
    execution_task_json: files.executionTaskJson
  });

  return {
    taskId,
    target,
    title,
    status: "draft",
    files,
    promptForClaude,
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
