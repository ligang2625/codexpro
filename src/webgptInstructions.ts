import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";

import type { CodexProConfig } from "./config.js";
import type { Workspace } from "./guard.js";
import { CodexProError, PathGuard, normalizeRelPath } from "./guard.js";

export const WEBGPT_INSTRUCTIONS_PATH = ".claude/WEBGPT.md";

export interface WebgptInstructionContext {
  text: string;
  files: string[];
  warnings: string[];
  found: boolean;
  entryPath: string;
  bytes: number;
}

export interface WebgptInstructionOptions {
  maxDepth?: number;
  maxFileBytes?: number;
  maxTotalBytes?: number;
}

const DEFAULT_MAX_IMPORT_DEPTH = 4;
const DEFAULT_MAX_FILE_BYTES = 60_000;
const DEFAULT_MAX_TOTAL_BYTES = 120_000;

// 第一阶段只支持独占一行的 Markdown import：
//   @relative/path.md
//
// 不解析正文里的 inline @path，也不解析代码块里的 @path。
// 这样可以避免误伤 GitHub handle、npm scope、邮箱、普通正文等。
const SIMPLE_MARKDOWN_IMPORT_RE = /^\s*@([^\s`]+\.md)\s*$/i;

function unique(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))];
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function normalizeWorkspaceRelPath(value: string): string {
  const normalized = path.posix.normalize(value.replace(/\\/g, "/"));
  return normalizeRelPath(normalized).replace(/^\.\//, "");
}

function isAbsoluteImportPath(value: string): boolean {
  return path.posix.isAbsolute(value) || path.win32.isAbsolute(value);
}

function resolveImportRelPath(importerRelPath: string, importSpec: string): string {
  const cleanSpec = importSpec.trim().replace(/\\/g, "/");

  if (!cleanSpec) {
    throw new CodexProError("Empty WebGPT @import path.");
  }

  if (isAbsoluteImportPath(cleanSpec)) {
    throw new CodexProError(
      `Absolute WebGPT @imports are not supported: ${cleanSpec}. ` +
        "Use a workspace-relative or file-relative markdown path instead."
    );
  }

  if (cleanSpec.startsWith("~")) {
    throw new CodexProError(
      `Home-directory WebGPT @imports are not supported: ${cleanSpec}. ` +
        "Only workspace-local markdown files may be imported."
    );
  }

  if (path.posix.extname(cleanSpec).toLowerCase() !== ".md") {
    throw new CodexProError(
      `Only markdown WebGPT @imports are supported: ${cleanSpec}`
    );
  }

  const importerDir = path.posix.dirname(importerRelPath.replace(/\\/g, "/"));
  const joined =
    importerDir === "." ? cleanSpec : path.posix.join(importerDir, cleanSpec);

  return normalizeWorkspaceRelPath(joined);
}

async function readWorkspaceMarkdownFile(
  config: CodexProConfig,
  guard: PathGuard,
  workspace: Workspace,
  relPath: string,
  maxBytes: number
): Promise<{ relPath: string; absPath: string; text: string; bytes: number }> {
  const normalizedRelPath = normalizeWorkspaceRelPath(relPath);

  if (path.posix.extname(normalizedRelPath).toLowerCase() !== ".md") {
    throw new CodexProError(
      `Only markdown WebGPT instruction files are supported: ${normalizedRelPath}`
    );
  }

  const resolved = guard.resolve(workspace, normalizedRelPath);
  await guard.assertTextFile(
    resolved.absPath,
    Math.min(maxBytes, config.maxReadBytes)
  );

  const text = await fsp.readFile(resolved.absPath, "utf8");

  return {
    relPath: resolved.relPath,
    absPath: resolved.absPath,
    text,
    bytes: Buffer.byteLength(text, "utf8")
  };
}

export async function findWebgptInstructionsPath(
  guard: PathGuard,
  workspace: Workspace
): Promise<string | undefined> {
  try {
    const resolved = guard.resolve(workspace, WEBGPT_INSTRUCTIONS_PATH);

    return fs.existsSync(resolved.absPath) && fs.statSync(resolved.absPath).isFile()
      ? resolved.relPath
      : undefined;
  } catch {
    return undefined;
  }
}

export function formatWebgptInstructionsForContext(
  webgpt: WebgptInstructionContext
): string {
  if (!webgpt.found) {
    return webgpt.text;
  }

  const warnings = webgpt.warnings.length
    ? [
        "",
        "",
        "### WebGPT Import Warnings",
        "",
        ...webgpt.warnings.map((warning) => `- ${warning}`)
      ].join("\n")
    : "";

  return [
    `Source: ${webgpt.entryPath}`,
    `Loaded files: ${webgpt.files.length ? webgpt.files.join(", ") : "none"}`,
    "",
    webgpt.text,
    warnings
  ]
    .filter(Boolean)
    .join("\n");
}

export async function readWebgptInstructions(
  config: CodexProConfig,
  guard: PathGuard,
  workspace: Workspace,
  options: WebgptInstructionOptions = {}
): Promise<WebgptInstructionContext> {
  const entryPath = WEBGPT_INSTRUCTIONS_PATH;
  const entry = await findWebgptInstructionsPath(guard, workspace);

  if (!entry) {
    return {
      text:
        `No ${WEBGPT_INSTRUCTIONS_PATH} found. ` +
        "WebGPT-specific project instructions are not loaded. " +
        "CodexPro does not load CLAUDE.md by default.",
      files: [],
      warnings: [],
      found: false,
      entryPath,
      bytes: 0
    };
  }

  const maxDepth = Math.max(
    0,
    Math.min(options.maxDepth ?? DEFAULT_MAX_IMPORT_DEPTH, 12)
  );

  const maxFileBytes = Math.max(
    1,
    Math.min(options.maxFileBytes ?? DEFAULT_MAX_FILE_BYTES, config.maxReadBytes)
  );

  const maxTotalBytes = Math.max(
    1,
    Math.min(options.maxTotalBytes ?? DEFAULT_MAX_TOTAL_BYTES, config.maxReadBytes)
  );

  const files: string[] = [];
  const warnings: string[] = [];
  const seenRealPaths = new Set<string>();
  let totalBytes = 0;

  async function expandFile(currentRelPath: string, depth: number): Promise<string> {
    const current = await readWorkspaceMarkdownFile(
      config,
      guard,
      workspace,
      currentRelPath,
      maxFileBytes
    );

    const realPath = await fsp.realpath(current.absPath);
    const seenKey = realPath.toLowerCase();

    if (seenRealPaths.has(seenKey)) {
      const warning = `Skipped repeated WebGPT import: ${current.relPath}`;
      warnings.push(warning);
      return `[import skipped: ${warning}]`;
    }

    seenRealPaths.add(seenKey);
    files.push(current.relPath);

    totalBytes += current.bytes;
    if (totalBytes > maxTotalBytes) {
      throw new CodexProError(
        `Expanded WebGPT instructions are too large (${totalBytes} bytes). ` +
          `Limit: ${maxTotalBytes} bytes.`
      );
    }

    const out: string[] = [];
    let inFence = false;

    const lines = current.text.replace(/\r\n/g, "\n").split("\n");

    for (const line of lines) {
      if (/^\s*(```|~~~)/.test(line)) {
        inFence = !inFence;
        out.push(line);
        continue;
      }

      const match = !inFence ? line.match(SIMPLE_MARKDOWN_IMPORT_RE) : null;

      if (!match) {
        out.push(line);
        continue;
      }

      const importSpec = match[1];

      if (depth >= maxDepth) {
        const warning =
          `Skipped @${importSpec} from ${current.relPath}: ` +
          `max import depth ${maxDepth} reached.`;

        warnings.push(warning);
        out.push(`[import skipped: ${warning}]`);
        continue;
      }

      try {
        const importedRelPath = resolveImportRelPath(current.relPath, importSpec);
        const importedText = await expandFile(importedRelPath, depth + 1);

        out.push("");
        out.push(
          `<!-- begin WebGPT import: @${importSpec} -> ${importedRelPath} -->`
        );
        out.push(importedText);
        out.push(`<!-- end WebGPT import: ${importedRelPath} -->`);
        out.push("");
      } catch (error) {
        const warning =
          `Failed to import @${importSpec} from ${current.relPath}: ` +
          errorMessage(error);

        warnings.push(warning);
        out.push(`[import failed: ${warning}]`);
      }
    }

    return [`--- ${current.relPath} ---`, out.join("\n")].join("\n");
  }

  try {
    const text = await expandFile(entry, 0);

    return {
      text,
      files: unique(files),
      warnings,
      found: true,
      entryPath: entry,
      bytes: totalBytes
    };
  } catch (error) {
    const warning = `Failed to load ${entry}: ${errorMessage(error)}`;

    return {
      text: `--- ${entry} ---\n[unreadable: ${warning}]`,
      files: [entry],
      warnings: [warning],
      found: true,
      entryPath: entry,
      bytes: totalBytes
    };
  }
}
