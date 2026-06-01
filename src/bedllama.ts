#!/usr/bin/env node

import { spawn, spawnSync } from "node:child_process";
import {
  appendFileSync,
  closeSync,
  existsSync,
  mkdirSync,
  openSync,
  readFileSync,
  readSync,
  statSync,
  unlinkSync,
  watchFile,
  writeFileSync,
} from "node:fs";
import type {
  SpawnSyncOptionsWithStringEncoding,
  SpawnSyncReturns,
} from "node:child_process";
import http, { IncomingMessage, ServerResponse } from "node:http";
import net from "node:net";
import os from "node:os";
import path from "node:path";

type JsonMap = Record<string, any>;
type NodeResponse = ServerResponse<IncomingMessage>;
type NodeFetchInit = RequestInit & { duplex?: "half" };
type LogTarget = "litellm" | "front" | "ollama" | "server" | "all";

interface ModelSpec {
  upstreamModel: string;
  shimModel: string;
  basename: string;
  family: string;
  parameterSize: string;
  contextLength: number;
  capabilities: string[];
  size: number;
  maxOutputTokens: number;
}

interface LitellmModel {
  modelName: string;
  bedrockModel: string;
}

interface ProcessState {
  pid: number | null;
  running: boolean;
}

interface RunOptions {
  stdio?: SpawnSyncOptionsWithStringEncoding["stdio"];
  env?: NodeJS.ProcessEnv;
}

interface OpenAIUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
}

interface OpenAIMessage {
  content?: unknown;
  tool_calls?: unknown;
}

interface OpenAIChoice {
  finish_reason?: string;
  message?: OpenAIMessage;
  delta?: OpenAIMessage;
}

interface OpenAIResponsePayload {
  usage?: OpenAIUsage;
  choices?: OpenAIChoice[];
}

const home = os.homedir();

function env(name: string, fallback: string): string {
  return process.env[name] || fallback;
}

function envNumber(name: string, fallback: number): number {
  const value = process.env[name];
  if (!value) {
    return fallback;
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    fail(`Invalid numeric value for ${name}: ${value}`);
  }
  return parsed;
}

function envList(name: string, fallback: string[]): string[] {
  const value = process.env[name];
  if (!value) {
    return fallback;
  }
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

// ---------------------------------------------------------------------------
// User config file (config.jsonc in the repo root, next to package.json)
// ---------------------------------------------------------------------------

interface UserConfig {
  adminUi?: boolean;
  adminUiUsername?: string;
  adminUiPassword?: string;
  postgresPort?: number;
  postgresPassword?: string;
  litellmPort?: number;
  frontPort?: number;
  ollamaPort?: number;
  awsProfile?: string;
  awsRegion?: string;
  apiKey?: string;
  models?: string;
}

function stripJsoncComments(src: string): string {
  // Remove single-line (//) and block (/* */) comments outside strings.
  let result = "";
  let i = 0;
  while (i < src.length) {
    if (src[i] === '"') {
      // consume string literal
      result += src[i++];
      while (i < src.length) {
        const ch = src[i];
        result += ch;
        i++;
        if (ch === '\\') {
          if (i < src.length) {
            result += src[i++];
          }
        } else if (ch === '"') {
          break;
        }
      }
    } else if (src[i] === '/' && src[i + 1] === '/') {
      // single-line comment — skip to end of line
      while (i < src.length && src[i] !== '\n') {
        i++;
      }
    } else if (src[i] === '/' && src[i + 1] === '*') {
      // block comment
      i += 2;
      while (i < src.length && !(src[i] === '*' && src[i + 1] === '/')) {
        i++;
      }
      i += 2;
    } else {
      result += src[i++];
    }
  }
  return result;
}

function loadUserConfig(): UserConfig {
  const candidates = [
    path.join(path.dirname(path.dirname(__filename)), "config.jsonc"),
    path.join(process.cwd(), "config.jsonc"),
  ];
  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      try {
        const raw = readFileSync(candidate, "utf8");
        return JSON.parse(stripJsoncComments(raw)) as UserConfig;
      } catch (err) {
        process.stderr.write(`Warning: failed to parse ${candidate}: ${err}\n`);
      }
    }
  }
  return {};
}

const userConfig = loadUserConfig();

const config = {
  processNames: {
    litellm: env("BEDLLAMA_LITELLM_PROCESS", "litellm"),
    server: env("BEDLLAMA_SERVER_PROCESS", "bedllama"),
  },
  hosts: {
    front: env("BEDLLAMA_FRONT_HOST", "127.0.0.1"),
    ollama: env("BEDLLAMA_OLLAMA_HOST", "127.0.0.1"),
  },
  ports: {
    litellm: envNumber("BEDLLAMA_LITELLM_PORT", userConfig.litellmPort ?? 4001),
    front: envNumber("BEDLLAMA_FRONT_PORT", userConfig.frontPort ?? 4000),
    ollama: envNumber("BEDLLAMA_OLLAMA_PORT", userConfig.ollamaPort ?? 11434),
  },
  stateDir: env("BEDLLAMA_STATE_DIR", path.join(home, ".cache", "bedllama")),
  apiKey: env("BEDLLAMA_API_KEY", userConfig.apiKey ?? "sk-local"),
  awsProfile: env("BEDLLAMA_AWS_PROFILE", userConfig.awsProfile ?? "bedrock"),
  awsRegion: env("BEDLLAMA_AWS_REGION", userConfig.awsRegion ?? "eu-west-3"),
  litellmBin: env("BEDLLAMA_LITELLM_BIN", "litellm"),
  litellmConfig: env(
    "BEDLLAMA_LITELLM_CONFIG",
    path.join(home, ".cache", "bedllama", "litellm.config.yaml")
  ),
  ollamaVersion: env("BEDLLAMA_OLLAMA_VERSION", "0.6.4"),
  ollamaDefaultModel: env("BEDLLAMA_OLLAMA_DEFAULT_MODEL", "claude-sonnet-4-6"),
  enableServerLog: env("BEDLLAMA_LOG", "1") !== "0",
  adminUi: userConfig.adminUi ?? false,
  adminUiUsername: userConfig.adminUiUsername ?? "admin",
  adminUiPassword: userConfig.adminUiPassword ?? "bedllama-admin",
  postgresPort: envNumber("BEDLLAMA_POSTGRES_PORT", userConfig.postgresPort ?? 5432),
  postgresPassword: env("BEDLLAMA_POSTGRES_PASSWORD", userConfig.postgresPassword ?? "bedllama"),
  postgresContainerName: env("BEDLLAMA_POSTGRES_CONTAINER", "bedllama-postgres"),
  models: envList("BEDLLAMA_MODELS", userConfig.models ? userConfig.models.split(",").map((m) => m.trim()).filter(Boolean) : []),
};

// ---------------------------------------------------------------------------
// Dynamic model discovery from AWS Bedrock inference profiles
// ---------------------------------------------------------------------------

// Models that expose a 1M-context variant via a separate shim name.
// Key: upstream model name, value: shim suffix for the 1M variant.
// Threshold: Anthropic models whose probed max-output-tokens meet or exceed this
// value automatically get a 1M-context shim variant registered at startup.
const LONG_CONTEXT_OUTPUT_THRESHOLD = 100_000;

// Model name fragments that are known NOT to support tool use via the Bedrock
// Converse API, even though they output TEXT and support streaming.
// Everything else is assumed to support tools.
const NO_TOOLS_MODELS = [
  "mistral-7b",
  "mixtral-8x7b",
  "llama3-2-1b",
  "llama3-2-3b",
  "pegasus",
];

function deriveModelNameFromProfileId(profileId: string): string {
  // eu.anthropic.claude-sonnet-4-6        →  claude-sonnet-4-6
  // eu.meta.llama3-2-1b-instruct-v1:0     →  llama3-2-1b-instruct
  // eu.amazon.nova-pro-v1:0               →  nova-pro
  // eu.mistral.pixtral-large-2502-v1:0    →  pixtral-large
  const withoutPrefix = profileId.replace(/^(?:eu|us|global)\.[^.]+\./, "");
  // Strip trailing version suffix like -20251001-v1:0 or -v1:0 or -v1
  return withoutPrefix.replace(/-\d{8}-v\d+(?::\d+)?$/, "").replace(/-v\d+(?::\d+)?$/, "");
}

function deriveProviderFromProfileId(profileId: string): string {
  // eu.anthropic.claude-sonnet-4-6  →  anthropic
  const m = profileId.match(/^(?:eu|us|global)\.([^.]+)\./);
  return m ? m[1]! : "unknown";
}

function deriveFamilyFromModelName(modelName: string, provider: string): string {
  if (provider === "anthropic") return "claude";
  if (provider === "amazon") return "nova";
  if (provider === "meta") return "llama";
  if (provider === "mistral") return "mistral";
  if (provider === "cohere") return "cohere";
  return modelName.split("-")[0] ?? provider;
}

function deriveCapabilitiesFromMap(
  profileId: string,
  capabilityMap: Map<string, FoundationModelCapabilities>
): string[] {
  const baseId = profileId.replace(/^(?:eu|us|global)\./, "");
  const caps = capabilityMap.get(baseId);
  if (!caps) {
    // Unknown model — fall back to NO_TOOLS_MODELS denylist + no vision assumption.
    const modelName = deriveModelNameFromProfileId(profileId);
    const tools = !NO_TOOLS_MODELS.some((f) => modelName.includes(f));
    return tools ? ["tools"] : [];
  }
  const result: string[] = [];
  if (caps.tools) result.push("tools");
  if (caps.vision) result.push("vision");
  return result;
}

function deriveParamSize(modelName: string, provider: string): string {
  if (provider === "anthropic") {
    // claude-sonnet-4-6  →  4.6-sonnet
    const m = modelName.match(/^claude-([a-z]+)-(\d+)-(\d+)/);
    if (m) return `${m[2]}.${m[3]}-${m[1]}`;
    return modelName.replace(/^claude-/, "");
  }
  return modelName;
}

function buildModelSpecFromProfile(
  profileId: string,
  capabilityMap: Map<string, FoundationModelCapabilities>,
  maxOutputTokens = 32_000
): ModelSpec {
  const modelName = deriveModelNameFromProfileId(profileId);
  const provider = deriveProviderFromProfileId(profileId);
  const family = deriveFamilyFromModelName(modelName, provider);
  return {
    upstreamModel: modelName,
    shimModel: `${modelName}:latest`,
    basename: modelName,
    family,
    parameterSize: deriveParamSize(modelName, provider),
    contextLength: 200000,
    capabilities: deriveCapabilitiesFromMap(profileId, capabilityMap),
    size: 3338801804,
    maxOutputTokens,
  };
}

// ---------------------------------------------------------------------------
// Bedrock output-token probing
// ---------------------------------------------------------------------------

// Probes the real max-output-token limit for a single inference profile by
// sending maxTokens=9_999_999 to the Converse API. Bedrock rejects the call
// instantly (no generation, no charge) with a ValidationException that
// includes the actual limit: "model limit of X".
function probeMaxOutputTokensAsync(profileId: string): Promise<number | null> {
  return new Promise((resolve) => {
    const child = spawn(
      "aws",
      [
        "bedrock-runtime", "converse",
        "--model-id", profileId,
        "--messages", JSON.stringify([{ role: "user", content: [{ text: "hi" }] }]),
        "--inference-config", JSON.stringify({ maxTokens: 9_999_999 }),
        "--profile", config.awsProfile,
        "--region", config.awsRegion,
      ],
      { stdio: ["ignore", "pipe", "pipe"] }
    );
    let stderr = "";
    child.stderr?.on("data", (d: Buffer) => { stderr += d.toString(); });
    child.on("close", () => {
      const m = stderr.match(/model limit of (\d+)/);
      resolve(m ? parseInt(m[1], 10) : null);
    });
  });
}

// Probes all given inference profiles concurrently and returns a map of
// basename → maxOutputTokens. Deduplicates: when a basename has both eu and
// global profiles, the eu one is preferred.
async function probeAllMaxOutputTokens(
  profileIds: string[]
): Promise<Map<string, number>> {
  const deduped = new Map<string, string>(); // basename → profileId
  for (const id of profileIds) {
    const basename = deriveModelNameFromProfileId(id);
    const existing = deduped.get(basename);
    // Prefer eu prefix; accept global as fallback.
    if (!existing || id.startsWith("eu.")) {
      deduped.set(basename, id);
    }
  }
  const entries = await Promise.all(
    Array.from(deduped.entries()).map(async ([basename, profileId]) => {
      const limit = await probeMaxOutputTokensAsync(profileId);
      return [basename, limit ?? 32_000] as const;
    })
  );
  return new Map(entries);
}

interface FoundationModelCapabilities {
  vision: boolean;
  tools: boolean;
}

function fetchFoundationModelCapabilities(): Map<string, FoundationModelCapabilities> {
  // Fetches inputModalities and outputModalities for all foundation models.
  // Returns a map keyed by base model ID (without region prefix).
  // vision = inputModalities includes IMAGE
  // tools  = outputModalities includes TEXT (streaming) AND not in NO_TOOLS_MODELS denylist
  const result = run("aws", [
    "bedrock", "list-foundation-models",
    "--profile", config.awsProfile,
    "--region", config.awsRegion,
    "--query", "modelSummaries[*].{id:modelId,in:inputModalities,out:outputModalities}",
    "--output", "json",
  ]);
  const map = new Map<string, FoundationModelCapabilities>();
  if (result.status !== 0) {
    return map;
  }
  try {
    const entries = JSON.parse(result.stdout.trim()) as Array<{
      id: string;
      in: string[];
      out: string[];
    }>;
    for (const entry of entries) {
      const vision = Array.isArray(entry.in) && entry.in.includes("IMAGE");
      const isTextOutput = Array.isArray(entry.out) && entry.out.includes("TEXT");
      const tools = isTextOutput && !NO_TOOLS_MODELS.some((f) => entry.id.includes(f));
      map.set(entry.id, { vision, tools });
    }
  } catch {
    // ignore parse errors — callers fall back gracefully
  }
  return map;
}

function fetchBedrockInferenceProfiles(
  capabilityMap: Map<string, FoundationModelCapabilities>
): string[] {
  const result = run("aws", [
    "bedrock", "list-inference-profiles",
    "--profile", config.awsProfile,
    "--region", config.awsRegion,
    "--query", "inferenceProfileSummaries[*].inferenceProfileId",
    "--output", "json",
  ]);
  if (result.status !== 0) {
    process.stderr.write(`Warning: could not list Bedrock inference profiles: ${(result.stderr || result.stdout || "").trim()}\n`);
    return [];
  }
  try {
    const ids = JSON.parse(result.stdout.trim()) as string[];
    // Filter out embedding models using the capability map.
    // Profile IDs like eu.cohere.embed-v4:0 map to base model cohere.embed-v4:0.
    return ids.filter((id) => {
      const baseId = id.replace(/^(?:eu|us|global)\./, "");
      const caps = capabilityMap.get(baseId);
      // If not in the map, keep it (unknown model — let LiteLLM handle it).
      // If in the map, keep only if it has TEXT output (i.e. not embedding-only).
      return !caps || caps.tools || caps.vision;
    });
  } catch {
    return [];
  }
}

function buildDynamicModels(
  profileIds: string[],
  capabilityMap: Map<string, FoundationModelCapabilities>,
  maxOutputMap: Map<string, number> = new Map()
): { modelSpecs: ModelSpec[]; litellmModelMap: LitellmModel[] } {
  const seen = new Set<string>();
  const modelSpecs: ModelSpec[] = [];
  const litellmModelMap: LitellmModel[] = [];

  for (const profileId of profileIds) {
    const modelName = deriveModelNameFromProfileId(profileId);
    if (seen.has(modelName)) continue;
    seen.add(modelName);

    const provider = deriveProviderFromProfileId(profileId);
    const family = deriveFamilyFromModelName(modelName, provider);
    const capabilities = deriveCapabilitiesFromMap(profileId, capabilityMap);
    const maxOutput = maxOutputMap.get(modelName) ?? 32_000;

    modelSpecs.push(buildModelSpecFromProfile(profileId, capabilityMap, maxOutput));
    litellmModelMap.push({
      modelName,
      bedrockModel: `bedrock/${profileId}`,
    });

    // Dynamically add a 1M-context shim variant for Anthropic models whose
    // probed output capacity meets the threshold. The shimModel name is what
    // clients use; proxyV1Request rewrites it to upstreamModel before
    // forwarding to LiteLLM, so LiteLLM never needs to know the -1m name.
    if (provider === "anthropic" && maxOutput >= LONG_CONTEXT_OUTPUT_THRESHOLD) {
      const longBasename = `${modelName}-1m`;
      modelSpecs.push({
        upstreamModel: modelName,
        shimModel: `${longBasename}:latest`,
        basename: longBasename,
        family,
        parameterSize: `${deriveParamSize(modelName, provider)}-1m`,
        contextLength: 1_000_000,
        capabilities,
        size: 3338801804,
        maxOutputTokens: maxOutput,
      });
    }
  }

  return { modelSpecs, litellmModelMap };
}

// Populated at startup by startStack(); used by the serve path via makeFallbackSpec.
let dynamicModelSpecs: ModelSpec[] = [];
let dynamicLitellmModelMap: LitellmModel[] = [];

function getModelSpecs(): ModelSpec[] {
  return dynamicModelSpecs;
}

function getLitellmModelMap(): LitellmModel[] {
  return dynamicLitellmModelMap;
}

function getDefaultSpec(): ModelSpec {
  const specs = dynamicModelSpecs;
  return (
    specs.find((s) => s.upstreamModel === config.ollamaDefaultModel) ??
    specs.find((s) => s.basename === config.ollamaDefaultModel) ??
    specs[0] ??
    makeFallbackSpec(config.ollamaDefaultModel)
  );
}

function fail(message: string, code = 1): never {
  process.stderr.write(`${message}\n`);
  process.exit(code);
}

function run(command: string, args: string[], options: RunOptions = {}): SpawnSyncReturns<string> {
  const result = spawnSync(command, args, {
    encoding: "utf8",
    stdio: options.stdio ?? "pipe",
    env: options.env ?? process.env,
  });

  if (result.error) {
    fail(`Failed to run ${command}: ${result.error.message}`);
  }

  return result;
}

function assertAvailable(command: string): void {
  const checker = process.platform === "win32" ? "where" : "which";
  if (run(checker, [command]).status !== 0) {
    fail(`Required command not found: ${command}`);
  }
}

function ensureStateDir(): void {
  mkdirSync(config.stateDir, { recursive: true });
}

function pidFilePath(name: string): string {
  return path.join(config.stateDir, `${name}.pid`);
}

function logFilePath(name: string): string {
  return path.join(config.stateDir, `${name}.log`);
}

function writePidFile(name: string, pid: number): void {
  ensureStateDir();
  writeFileSync(pidFilePath(name), `${pid}\n`, "utf8");
}

function readPidFile(name: string): number | null {
  const file = pidFilePath(name);
  if (!existsSync(file)) {
    return null;
  }
  const raw = readFileSync(file, "utf8").trim();
  if (!raw) {
    return null;
  }
  const pid = Number(raw);
  return Number.isInteger(pid) && pid > 0 ? pid : null;
}

function clearPidFile(name: string): void {
  try {
    unlinkSync(pidFilePath(name));
  } catch {
    // ignore missing pid files
  }
}

function isProcessRunning(pid: number | null): boolean {
  if (!pid) {
    return false;
  }
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function stopPid(pid: number | null): void {
  if (!pid) {
    return;
  }
  try {
    if (process.platform === "win32") {
      run("taskkill", ["/PID", String(pid), "/T", "/F"], { stdio: "ignore" });
      return;
    }
    process.kill(pid, "SIGTERM");
  } catch {
    // ignore dead processes
  }
}

function appendLogBanner(name: string, message: string): void {
  ensureStateDir();
  appendFileSync(logFilePath(name), `${new Date().toISOString()} ${message}\n`);
}

function startDetachedProcess(
  name: string,
  command: string,
  args: string[],
  childEnv: NodeJS.ProcessEnv
): number {
  ensureStateDir();
  const logFd = openSync(logFilePath(name), "a");
  const child = spawn(command, args, {
    detached: true,
    stdio: ["ignore", logFd, logFd],
    env: childEnv,
    windowsHide: true,
  });
  child.unref();
  closeSync(logFd);
  if (typeof child.pid !== "number") {
    fail(`Failed to start ${name}: child process has no pid`);
  }
  writePidFile(name, child.pid);
  appendLogBanner(name, `[bedllama] started pid=${child.pid}`);
  return child.pid;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForHttp(url: string, headers: HeadersInit = {}): Promise<void> {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    try {
      const response = await fetch(url, { headers });
      if (response.ok) {
        return;
      }
    } catch {
      // retry
    }
    await sleep(200);
  }
  fail(`Timed out waiting for ${url}`);
}

function waitForTcp(host: string, port: number, timeoutMs = 30000): Promise<void> {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + timeoutMs;
    const attempt = (): void => {
      const socket = new net.Socket();
      socket.setTimeout(1000);
      socket.once("connect", () => {
        socket.destroy();
        resolve();
      });
      socket.once("error", () => {
        socket.destroy();
        if (Date.now() >= deadline) {
          reject(new Error(`Timed out waiting for TCP ${host}:${port}`));
        } else {
          setTimeout(attempt, 300);
        }
      });
      socket.once("timeout", () => {
        socket.destroy();
        if (Date.now() >= deadline) {
          reject(new Error(`Timed out waiting for TCP ${host}:${port}`));
        } else {
          setTimeout(attempt, 300);
        }
      });
      socket.connect(port, host);
    };
    attempt();
  });
}

async function waitForPostgresReady(timeoutMs = 30000): Promise<void> {
  // Use pg_isready (bundled in the postgres:16 image) via docker/podman exec
  // to confirm the DB cluster is fully up and accepting queries.
  const checker = process.platform === "win32" ? "where" : "which";
  const cli =
    run(checker, ["podman"]).status === 0
      ? "podman"
      : run(checker, ["docker"]).status === 0
      ? "docker"
      : null;

  if (!cli) {
    // No container CLI — fall back to a short sleep.
    await sleep(2000);
    return;
  }

  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const result = run(cli, [
      "exec", config.postgresContainerName,
      "pg_isready", "-U", "litellm", "-d", "litellm",
    ]);
    if (result.status === 0) {
      return;
    }
    await sleep(500);
  }
  fail(`Timed out waiting for PostgreSQL to accept connections on port ${config.postgresPort}`);
}

function requireDependencies(): void {
  // Check litellm binary is available.
  assertAvailable(config.litellmBin);

  // Check litellm[proxy] extra is installed by asking uv tool list.
  // This is cross-platform: uv is available on Linux, macOS, and Windows.
  const uvCheck = run("uv", ["tool", "list"]);
  const uvOut = (uvCheck.stdout || "") + (uvCheck.stderr || "");
  if (uvCheck.status !== 0 || !uvOut.includes("litellm-proxy")) {
    // Fall back to checking for the binary directly (handles non-uv installs).
    const checker = process.platform === "win32" ? "where" : "which";
    const proxyBin = config.litellmBin === "litellm" ? "litellm-proxy" : config.litellmBin;
    if (run(checker, [proxyBin]).status !== 0) {
      fail(
        `litellm[proxy] does not appear to be installed.\n` +
        `Install it with: uv tool install 'litellm[proxy]'`
      );
    }
  }

  if (config.adminUi) {
    requirePrisma();
  }

  // Check AWS credentials for the configured profile.
  process.stdout.write(`Checking AWS credentials (profile: ${config.awsProfile})...\n`);
  const awsCheck = run("aws", [
    "sts", "get-caller-identity",
    "--profile", config.awsProfile,
    "--output", "text",
    "--query", "Arn",
  ]);
  if (awsCheck.status !== 0) {
    const detail = (awsCheck.stderr || awsCheck.stdout || "").trim();
    fail(
      `AWS credentials check failed for profile '${config.awsProfile}'.\n` +
      (detail ? `${detail}\n` : "") +
      `Run your AWS login flow first, then retry \`bedllama start\`.`
    );
  }
  const arn = (awsCheck.stdout || "").trim();
  process.stdout.write(`AWS credentials OK: ${arn}\n`);
}

function litellmToolBinDir(): string | null {
  // Resolve the bin directory of the litellm uv tool environment.
  // `which litellm` may return a shim — follow symlinks to the real binary.
  const which = run(process.platform === "win32" ? "where" : "which", [config.litellmBin]);
  if (which.status !== 0) {
    return null;
  }
  const shimPath = which.stdout.trim();
  // Follow symlinks (realpath) to get the actual tool env binary.
  const realpath = run("realpath", [shimPath]);
  const resolved = realpath.status === 0 ? realpath.stdout.trim() : shimPath;
  return path.dirname(resolved);
}

function litellmPrismaContext(): { binDir: string; prismaBin: string; schemaPath: string } | null {
  const binDir = litellmToolBinDir();
  if (!binDir) {
    return null;
  }
  const prismaBin = path.join(binDir, "prisma");
  const libDir = path.join(binDir, "..", "lib");
  const schemaSearch = run("find", [
    libDir,
    "-name", "schema.prisma",
    "-path", "*/litellm/proxy/*",
  ]);
  const schemaPath = (schemaSearch.stdout || "").trim().split("\n")[0] ?? "";
  if (!schemaPath || !existsSync(schemaPath)) {
    return null;
  }
  return { binDir, prismaBin, schemaPath };
}

function requirePrisma(): void {
  const binDir = litellmToolBinDir();
  if (!binDir) {
    return; // can't locate — let LiteLLM fail with its own error
  }

  const prismaBin = path.join(binDir, "prisma");
  const prismaClientBin = path.join(binDir, "prisma-client-py");

  // Ensure prisma package is installed in the litellm tool env.
  if (!existsSync(prismaBin) || !existsSync(prismaClientBin)) {
    process.stdout.write("Installing prisma into litellm tool environment...\n");
    const install = run("uv", ["tool", "install", "--force", "litellm[proxy]", "--with", "prisma"]);
    if (install.status !== 0) {
      fail(
        `Failed to install prisma:\n${(install.stderr || install.stdout || "").trim()}\n` +
        `Run manually: uv tool install --force 'litellm[proxy]' --with prisma`
      );
    }
  }

  const ctx = litellmPrismaContext();
  if (!ctx) {
    process.stdout.write("Warning: could not locate litellm schema.prisma — skipping prisma generate\n");
    return;
  }

  // Check if the Prisma client has already been generated.
  const sitePackages = path.dirname(path.dirname(ctx.schemaPath)); // .../site-packages
  const prismaPackage = path.join(sitePackages, "prisma");
  const clientGenerated = existsSync(path.join(prismaPackage, "client")) ||
    existsSync(path.join(prismaPackage, "_builder.py"));

  if (clientGenerated) {
    return;
  }

  process.stdout.write("Running prisma generate for LiteLLM DB support...\n");
  const generate = run(ctx.prismaBin, ["generate", "--schema", ctx.schemaPath], {
    env: { ...process.env, PATH: `${ctx.binDir}${path.delimiter}${process.env.PATH ?? ""}` },
  });
  if (generate.status !== 0) {
    const detail = (generate.stderr || generate.stdout || "").trim();
    fail(`prisma generate failed:\n${detail}`);
  }
  process.stdout.write("Prisma client generated\n");
}

function prismaDbPush(): void {
  const ctx = litellmPrismaContext();
  if (!ctx) {
    process.stdout.write("Warning: could not locate litellm schema.prisma — skipping prisma db push\n");
    return;
  }
  process.stdout.write("Applying LiteLLM database schema (prisma db push)...\n");
  const result = run(ctx.prismaBin, ["db", "push", "--schema", ctx.schemaPath, "--accept-data-loss", "--skip-generate"], {
    env: {
      ...process.env,
      PATH: `${ctx.binDir}${path.delimiter}${process.env.PATH ?? ""}`,
      DATABASE_URL: postgresUrl(),
    },
  });
  if (result.status !== 0) {
    // Redact DATABASE_URL (contains password) from error output before printing.
    const raw = (result.stderr || result.stdout || "").trim();
    const redacted = raw.replace(/postgresql:\/\/[^@]+@[^\s"']*/gi, "postgresql://<redacted>");
    fail(`prisma db push failed:\n${redacted}`);
  }
  process.stdout.write("Database schema up to date\n");
}

function litellmArgs(): string[] {
  return ["--config", config.litellmConfig, "--port", String(config.ports.litellm)];
}

function litellmEnv(): NodeJS.ProcessEnv {
  const childEnv: NodeJS.ProcessEnv = { ...process.env };
  delete childEnv.AWS_ACCESS_KEY_ID;
  delete childEnv.AWS_SECRET_ACCESS_KEY;
  delete childEnv.AWS_SESSION_TOKEN;
  delete childEnv.AWS_CREDENTIAL_EXPIRATION;
  childEnv.AWS_PROFILE = config.awsProfile;
  childEnv.AWS_DEFAULT_PROFILE = config.awsProfile;
  return childEnv;
}

function serverArgs(): string[] {
  return [__filename, "serve"];
}

function serverEnv(modelSpecs: ModelSpec[]): NodeJS.ProcessEnv {
  return {
    ...process.env,
    BEDLLAMA_STATE_DIR: config.stateDir,
    BEDLLAMA_FRONT_HOST: config.hosts.front,
    BEDLLAMA_FRONT_PORT: String(config.ports.front),
    BEDLLAMA_OLLAMA_HOST: config.hosts.ollama,
    BEDLLAMA_OLLAMA_PORT: String(config.ports.ollama),
    BEDLLAMA_LITELLM_PORT: String(config.ports.litellm),
    BEDLLAMA_API_KEY: config.apiKey,
    BEDLLAMA_OLLAMA_VERSION: config.ollamaVersion,
    BEDLLAMA_OLLAMA_DEFAULT_MODEL: config.ollamaDefaultModel,
    BEDLLAMA_MODELS: config.models.join(","),
    BEDLLAMA_LOG: config.enableServerLog ? "1" : "0",
    BEDLLAMA_MODEL_SPECS: JSON.stringify(modelSpecs),
  };
}

function yamlScalar(value: string): string {
  return JSON.stringify(String(value));
}

function detectContainerCli(): string {
  // Prefer podman if available, fall back to docker.
  const checker = process.platform === "win32" ? "where" : "which";
  if (run(checker, ["podman"]).status === 0) {
    return "podman";
  }
  if (run(checker, ["docker"]).status === 0) {
    return "docker";
  }
  fail(
    "Neither podman nor docker was found.\n" +
    "Install one of them to use the adminUi feature (requires a PostgreSQL container)."
  );
}

function postgresUrl(): string {
  return `postgresql://litellm:${config.postgresPassword}@127.0.0.1:${config.postgresPort}/litellm`;
}

function startPostgres(): void {
  const cli = detectContainerCli();

  // Check if the container already exists and is running.
  const inspect = run(cli, ["inspect", "--format", "{{.State.Running}}", config.postgresContainerName]);
  if (inspect.status === 0 && inspect.stdout.trim() === "true") {
    process.stdout.write(`PostgreSQL container '${config.postgresContainerName}' already running\n`);
    return;
  }

  // Remove a stopped container with the same name so we can recreate it.
  if (inspect.status === 0) {
    run(cli, ["rm", config.postgresContainerName], { stdio: "ignore" });
  }

  process.stdout.write(`Starting PostgreSQL container '${config.postgresContainerName}'...\n`);
  // Write a temporary env file so the password is not visible in `ps aux`.
  const envFile = path.join(config.stateDir, "postgres.env");
  ensureStateDir();
  writeFileSync(envFile, [
    "POSTGRES_USER=litellm",
    `POSTGRES_PASSWORD=${config.postgresPassword}`,
    "POSTGRES_DB=litellm",
  ].join("\n") + "\n", { encoding: "utf8", mode: 0o600 });
  const result = run(cli, [
    "run",
    "--detach",
    "--name", config.postgresContainerName,
    "--publish", `127.0.0.1:${config.postgresPort}:5432`,
    "--env-file", envFile,
    "--volume", `${config.postgresContainerName}-data:/var/lib/postgresql/data`,
    "--restart", "unless-stopped",
    "postgres:16",
  ]);
  if (result.status !== 0) {
    const detail = (result.stderr || result.stdout || "").trim();
    fail(`Failed to start PostgreSQL container:\n${detail}`);
  }
  process.stdout.write(`PostgreSQL container started (port ${config.postgresPort})\n`);
}

function stopPostgres(): void {
  // Detect CLI without failing if neither is present (stop should be lenient).
  const checker = process.platform === "win32" ? "where" : "which";
  const cli =
    run(checker, ["podman"]).status === 0
      ? "podman"
      : run(checker, ["docker"]).status === 0
      ? "docker"
      : null;
  if (!cli) {
    return;
  }
  const inspect = run(cli, ["inspect", "--format", "{{.State.Running}}", config.postgresContainerName]);
  if (inspect.status !== 0) {
    return; // container doesn't exist
  }
  run(cli, ["stop", config.postgresContainerName], { stdio: "ignore" });
}

function generateLitellmConfig(): string {
  const lines: string[] = ["model_list:"];
  for (const model of getLitellmModelMap()) {
    lines.push(`  - model_name: ${yamlScalar(model.modelName)}`);
    lines.push("    litellm_params:");
    lines.push(`      model: ${yamlScalar(model.bedrockModel)}`);
    lines.push(`      aws_region_name: ${yamlScalar(config.awsRegion)}`);
    lines.push(`      aws_profile_name: ${yamlScalar(config.awsProfile)}`);
    lines.push("");
  }
  lines.push("general_settings:");
  lines.push(`  master_key: ${yamlScalar(config.apiKey)}`);
  if (config.adminUi) {
    lines.push(`  database_url: ${yamlScalar(postgresUrl())}`);
    lines.push("  store_model_in_db: true");
  }
  lines.push("");
  if (config.adminUi) {
    lines.push("litellm_settings:");
    lines.push("  ui_access_mode: \"all\"");
    lines.push("");
    lines.push("environment_variables:");
    lines.push(`  UI_USERNAME: ${yamlScalar(config.adminUiUsername)}`);
    lines.push(`  UI_PASSWORD: ${yamlScalar(config.adminUiPassword)}`);
    lines.push("");
  }
  return lines.join("\n");
}

function writeLitellmConfig(): void {
  mkdirSync(path.dirname(config.litellmConfig), { recursive: true });
  // mode 0o600: owner read/write only — file contains passwords and API keys.
  writeFileSync(config.litellmConfig, generateLitellmConfig(), { encoding: "utf8", mode: 0o600 });
}

function processStatus(name: string): ProcessState {
  const pid = readPidFile(name);
  const running = isProcessRunning(pid);
  if (!running) {
    clearPidFile(name);
  }
  return { pid, running };
}

async function startStack(): Promise<void> {
  requireDependencies();
  stopStack({ quiet: true });

  // Discover available models from AWS Bedrock inference profiles.
  process.stdout.write("Fetching available Bedrock model capabilities...\n");
  const capabilityMap = fetchFoundationModelCapabilities();

  process.stdout.write("Fetching available Bedrock inference profiles...\n");
  const allProfileIds = fetchBedrockInferenceProfiles(capabilityMap);

  // If config.models is set, use it as a filter on the fetched profiles.
  // Entries can be shim names ("claude-sonnet-4-6:latest") or bare model names.
  const profileIds = config.models.length > 0
    ? (() => {
        const allowed = new Set(
          config.models.map((m) => (m.endsWith(":latest") ? m.slice(0, -7) : m))
        );
        return allProfileIds.filter((id) => allowed.has(deriveModelNameFromProfileId(id)));
      })()
    : allProfileIds;

  if (profileIds.length === 0) {
    fail("No Bedrock inference profiles found. Check your AWS credentials and region.");
  }

  process.stdout.write(`Probing output limits for ${profileIds.length} model(s) in parallel...\n`);
  const maxOutputMap = await probeAllMaxOutputTokens(profileIds);

  const { modelSpecs, litellmModelMap } = buildDynamicModels(profileIds, capabilityMap, maxOutputMap);
  dynamicModelSpecs = modelSpecs;
  dynamicLitellmModelMap = litellmModelMap;

  if (config.adminUi) {
    startPostgres();
    process.stdout.write("Waiting for PostgreSQL to be ready...\n");
    await waitForTcp("127.0.0.1", config.postgresPort);
    // TCP is up but postgres may still be initialising the DB cluster.
    // Poll until it accepts a real connection on the litellm database.
    await waitForPostgresReady();
    // Push the schema (idempotent — safe to run on every start).
    prismaDbPush();
  }

  writeLitellmConfig();

  startDetachedProcess(config.processNames.litellm, config.litellmBin, litellmArgs(), litellmEnv());
  try {
    await waitForHttp(`http://127.0.0.1:${config.ports.litellm}/v1/models`, {
      Authorization: `Bearer ${config.apiKey}`,
    });
  } catch (error) {
    const litellmLog = tailLines(logFilePath(config.processNames.litellm), 40);
    stopStack({ quiet: true });
    process.stderr.write("Warning: LiteLLM did not become ready.\n");
    process.stderr.write(
      `Warning: Bedrock credentials for AWS profile ${config.awsProfile} may be missing, expired, or invalid.\n`
    );
    process.stderr.write(
      "Warning: run your normal AWS login flow first, then retry `bedllama start`.\n"
    );
    if (litellmLog) {
      process.stderr.write("\nRecent LiteLLM log output:\n");
      process.stderr.write(`${litellmLog}\n`);
    }
    fail(String(error));
  }

  startDetachedProcess(config.processNames.server, process.execPath, serverArgs(), serverEnv(dynamicModelSpecs));
  await waitForHttp(`http://${config.hosts.front}:${config.ports.front}/health`);
  await waitForHttp(`http://${config.hosts.ollama}:${config.ports.ollama}/api/version`);

  process.stdout.write("bedllama stack is up\n");
  process.stdout.write(`Ollama endpoint: http://${config.hosts.ollama}:${config.ports.ollama}\n`);
  process.stdout.write(
    `OpenAI-compatible endpoint: http://${config.hosts.front}:${config.ports.front}/v1\n`
  );
  if (config.adminUi) {
    process.stdout.write(
      `LiteLLM admin UI:           http://127.0.0.1:${config.ports.litellm}/ui\n`
    );
  }
  process.stdout.write("Models:\n");
  for (const spec of dynamicModelSpecs) {
    process.stdout.write(`  - ${spec.shimModel}\n`);
  }
}

function stopStack(options: { quiet?: boolean } = {}): void {
  for (const name of [config.processNames.server, config.processNames.litellm]) {
    const pid = readPidFile(name);
    stopPid(pid);
    clearPidFile(name);
  }

  if (config.adminUi) {
    stopPostgres();
  }

  if (!options.quiet) {
    process.stdout.write("bedllama stack stopped\n");
  }
}

function statusStack(): void {
  const litellm = processStatus(config.processNames.litellm);
  const server = processStatus(config.processNames.server);

  process.stdout.write("processes:\n");
  process.stdout.write(
    `  ${litellm.running ? "up  " : "down"} litellm (${config.processNames.litellm})${
      litellm.pid ? ` pid=${litellm.pid}` : ""
    }\n`
  );
  process.stdout.write(
    `  ${server.running ? "up  " : "down"} bedllama (${config.processNames.server})${
      server.pid ? ` pid=${server.pid}` : ""
    }\n`
  );

  process.stdout.write("\nendpoints:\n");
  process.stdout.write(`  litellm: http://127.0.0.1:${config.ports.litellm}/v1\n`);
  if (config.adminUi) {
    process.stdout.write(`  admin:   http://127.0.0.1:${config.ports.litellm}/ui\n`);
  }
  process.stdout.write(`  front:   http://${config.hosts.front}:${config.ports.front}/v1\n`);
  process.stdout.write(`  ollama:  http://${config.hosts.ollama}:${config.ports.ollama}\n`);
}

function tailLines(filePath: string, count: number): string {
  if (!existsSync(filePath)) {
    return "";
  }
  const raw = readFileSync(filePath, "utf8").trimEnd();
  if (!raw) {
    return "";
  }
  const lines = raw.split("\n");
  return lines.slice(-count).join("\n");
}

function followLogs(files: string[]): void {
  // Print existing tail then watch each file for new content.
  const offsets = new Map<string, number>();

  const printNew = (filePath: string): void => {
    if (!existsSync(filePath)) {
      return;
    }
    const size = statSync(filePath).size;
    const prev = offsets.get(filePath) ?? size;
    if (size > prev) {
      const buf = Buffer.alloc(size - prev);
      const fd = openSync(filePath, "r");
      try {
        const bytesRead = readSync(fd, buf, 0, buf.length, prev);
        if (bytesRead > 0) {
          process.stdout.write(buf.subarray(0, bytesRead));
        }
      } finally {
        closeSync(fd);
      }
    }
    offsets.set(filePath, size);
  };

  for (const filePath of files) {
    // Print existing tail first.
    if (existsSync(filePath)) {
      const raw = readFileSync(filePath, "utf8").trimEnd();
      if (raw) {
        process.stdout.write(`${raw}\n`);
      }
      offsets.set(filePath, statSync(filePath).size);
    } else {
      offsets.set(filePath, 0);
    }
    watchFile(filePath, { interval: 250, persistent: true }, () => {
      printNew(filePath);
    });
  }

  // Keep the process alive; Ctrl-C exits cleanly.
  process.on("SIGINT", () => process.exit(0));
  process.on("SIGTERM", () => process.exit(0));
}

function logsStack(which: LogTarget = "all", follow = false): void {
  const targetMap: Record<Exclude<LogTarget, "all">, [string, number]> = {
    litellm: [config.processNames.litellm, 120],
    front: [config.processNames.server, 120],
    ollama: [config.processNames.server, 120],
    server: [config.processNames.server, 120],
  };

  if (which === "all") {
    if (follow) {
      followLogs([
        logFilePath(config.processNames.litellm),
        logFilePath(config.processNames.server),
      ]);
      return;
    }
    for (const [label, name, count] of [
      ["litellm", config.processNames.litellm, 60],
      ["bedllama", config.processNames.server, 120],
    ] as const) {
      process.stdout.write(`=== ${label} (${name}) ===\n`);
      const text = tailLines(logFilePath(name), count);
      process.stdout.write(`${text}${text ? "\n" : ""}`);
      if (label !== "bedllama") {
        process.stdout.write("\n");
      }
    }
    return;
  }

  const target = targetMap[which];
  if (follow) {
    followLogs([logFilePath(target[0])]);
    return;
  }
  const text = tailLines(logFilePath(target[0]), target[1]);
  process.stdout.write(`${text}${text ? "\n" : ""}`);
}

function serverLog(message: string): void {
  if (!config.enableServerLog) {
    return;
  }
  const ts = new Date().toISOString().slice(11, 23); // HH:MM:SS.mmm
  process.stdout.write(`[bedllama] ${ts} ${message}\n`);
}

function ms(start: number): string {
  return `${Date.now() - start}ms`;
}

function sendJson(res: NodeResponse, status: number, body: unknown): void {
  const data = JSON.stringify(body);
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(data),
  });
  res.end(data);
}

function sendText(res: NodeResponse, status: number, body: string): void {
  res.writeHead(status, {
    "content-type": "text/plain; charset=utf-8",
    "content-length": Buffer.byteLength(body),
  });
  res.end(body);
}

async function readJson(req: IncomingMessage): Promise<JsonMap> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  if (chunks.length === 0) {
    return {};
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8")) as JsonMap;
}

async function readBodyText(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return Buffer.concat(chunks).toString("utf8");
}

async function litellmFetch(requestPath: string, init: NodeFetchInit = {}): Promise<Response> {
  const headers = new Headers(init.headers ?? {});
  headers.set("authorization", `Bearer ${config.apiKey}`);
  if (init.body && !headers.has("content-type")) {
    headers.set("content-type", "application/json");
  }
  return fetch(`http://127.0.0.1:${config.ports.litellm}${requestPath}`, {
    ...init,
    headers,
  });
}

async function proxyStream(upstream: Response, res: NodeResponse): Promise<void> {
  const extraHeaders: Record<string, string> = {};
  for (const [key, value] of upstream.headers.entries()) {
    if (key.startsWith("x-litellm-")) {
      extraHeaders[key] = value;
    }
  }
  res.writeHead(upstream.status, {
    "content-type": upstream.headers.get("content-type") || "application/json",
    "transfer-encoding": "chunked",
    ...extraHeaders,
  });
  if (!upstream.body) {
    res.end();
    return;
  }
  for await (const chunk of upstream.body as AsyncIterable<Uint8Array>) {
    res.write(Buffer.from(chunk));
  }
  res.end();
}

function makeFallbackSpec(model: string): ModelSpec {
  const basename = model.endsWith(":latest") ? model.slice(0, -7) : model;
  return {
    upstreamModel: basename,
    shimModel: model.endsWith(":latest") ? model : `${basename}:latest`,
    basename,
    family: "claude",
    parameterSize: basename.replace(/^claude-/, ""),
    contextLength: 200000,
    capabilities: NO_TOOLS_MODELS.some((f) => basename.includes(f)) ? [] : ["tools", "vision"],
    size: 3338801804,
    maxOutputTokens: 32_000,
  };
}

function getModelSpec(model: string | undefined): ModelSpec {
  if (!model) {
    return getDefaultSpec();
  }
  const basename = model.endsWith(":latest") ? model.slice(0, -7) : model;
  const specs = getModelSpecs();
  return (
    specs.find((s) => s.shimModel === model) ??
    specs.find((s) => s.upstreamModel === model) ??
    specs.find((s) => s.basename === basename) ??
    makeFallbackSpec(model)
  );
}

function sanitizeSamplingParams(payload: JsonMap, spec?: ModelSpec): JsonMap {
  if (!payload || typeof payload !== "object") {
    return payload;
  }

  // Strip image content parts from messages when the model doesn't support vision.
  if (spec && !spec.capabilities.includes("vision") && Array.isArray(payload.messages)) {
    payload.messages = (payload.messages as JsonMap[]).map((msg) => {
      if (!Array.isArray(msg.content)) return msg;
      const textOnly = (msg.content as JsonMap[]).filter(
        (part) => part.type !== "image_url" && part.type !== "image"
      );
      return { ...msg, content: textOnly };
    });
  }

  if (payload.options && typeof payload.options === "object") {
    if (payload.temperature == null && payload.options.temperature != null) {
      payload.temperature = payload.options.temperature;
    }
    if (payload.top_p == null && payload.options.top_p != null) {
      payload.top_p = payload.options.top_p;
    }
    if (payload.top_k == null && payload.options.top_k != null) {
      payload.top_k = payload.options.top_k;
    }
  }

  if (spec?.upstreamModel === "claude-opus-4-7") {
    delete payload.temperature;
    delete payload.top_p;
    delete payload.top_k;
  } else if (payload.temperature != null && payload.top_p != null) {
    delete payload.top_p;
  }

  return payload;
}

function usageFromOpenAI(payload: OpenAIResponsePayload): JsonMap {
  const usage = payload.usage || {};
  return {
    total_duration: 0,
    load_duration: 0,
    prompt_eval_count: usage.prompt_tokens || 0,
    prompt_eval_duration: 0,
    eval_count: usage.completion_tokens || 0,
    eval_duration: 0,
  };
}

function textFromOpenAiContent(contentValue: unknown): string {
  if (typeof contentValue === "string") {
    return contentValue;
  }
  if (Array.isArray(contentValue)) {
    return contentValue
      .map((part) =>
        typeof part === "object" && part !== null && "text" in part && typeof part.text === "string"
          ? part.text
          : ""
      )
      .join("");
  }
  return "";
}

function ollamaChatFromOpenAI(payload: OpenAIResponsePayload, spec: ModelSpec): JsonMap {
  const choice = payload.choices?.[0] || {};
  const message = choice.message || {};
  return {
    model: spec.shimModel,
    created_at: new Date().toISOString(),
    message: {
      role: "assistant",
      content: textFromOpenAiContent(message.content),
      tool_calls: message.tool_calls,
    },
    done: true,
    done_reason: choice.finish_reason || "stop",
    ...usageFromOpenAI(payload),
  };
}

function ollamaGenerateFromOpenAI(payload: OpenAIResponsePayload, spec: ModelSpec): JsonMap {
  const choice = payload.choices?.[0] || {};
  const message = choice.message || {};
  return {
    model: spec.shimModel,
    created_at: new Date().toISOString(),
    response: textFromOpenAiContent(message.content),
    done: true,
    done_reason: choice.finish_reason || "stop",
    ...usageFromOpenAI(payload),
  };
}

async function streamOpenAiToOllama(
  upstream: Response,
  res: NodeResponse,
  spec: ModelSpec,
  kind: "chat" | "generate",
  logCtx?: { label: string; start: number; ttfb: number; readDone: number }
): Promise<void> {
  res.writeHead(200, {
    "content-type": "application/x-ndjson",
    "transfer-encoding": "chunked",
  });

  if (!upstream.body) {
    res.end();
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  const writeChunk = (content: string, toolCalls: unknown): void => {
    const chunk =
      kind === "chat"
        ? {
            model: spec.shimModel,
            created_at: new Date().toISOString(),
            message: {
              role: "assistant",
              content,
              ...(toolCalls ? { tool_calls: toolCalls } : {}),
            },
            done: false,
          }
        : {
            model: spec.shimModel,
            created_at: new Date().toISOString(),
            response: content,
            done: false,
          };
    res.write(`${JSON.stringify(chunk)}\n`);
  };

  for await (const rawChunk of upstream.body as AsyncIterable<Uint8Array>) {
    buffer += decoder.decode(rawChunk, { stream: true });
    let splitIndex: number;
    while ((splitIndex = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, splitIndex).trim();
      buffer = buffer.slice(splitIndex + 1);

      if (!line.startsWith("data:")) {
        continue;
      }

      const data = line.slice(5).trim();
      if (data === "[DONE]") {
        const doneChunk =
          kind === "chat"
            ? {
                model: spec.shimModel,
                created_at: new Date().toISOString(),
                message: { role: "assistant", content: "" },
                done: true,
                done_reason: "stop",
              }
            : {
                model: spec.shimModel,
                created_at: new Date().toISOString(),
                response: "",
                done: true,
                done_reason: "stop",
              };
        res.write(`${JSON.stringify(doneChunk)}\n`);
        res.end();
        return;
      }

      const payload = JSON.parse(data) as OpenAIResponsePayload;
      const delta = payload.choices?.[0]?.delta || {};
      const content = textFromOpenAiContent(delta.content);
      const toolCalls = delta.tool_calls;

      if (content || toolCalls) {
        writeChunk(content, toolCalls);
      }

      if (logCtx && payload.usage) {
        const u = payload.usage;
        const streamMs = Date.now() - logCtx.ttfb;
        const proc = (logCtx.readDone - logCtx.start) + streamMs;
        serverLog(
          `${logCtx.label} done total=${ms(logCtx.start)} read=${logCtx.readDone - logCtx.start}ms ttfb=${logCtx.ttfb - logCtx.readDone}ms stream=${streamMs}ms proc=${proc}ms prompt=${u.prompt_tokens ?? "?"} completion=${u.completion_tokens ?? "?"}`
        );
      }
    }
  }

  res.end();
}

function modelDetails(spec: ModelSpec): JsonMap {
  return {
    parent_model: "",
    format: "gguf",
    family: spec.family,
    families: [spec.family],
    parameter_size: spec.parameterSize,
    quantization_level: "Q4_K_M",
  };
}

function modelTag(spec: ModelSpec): JsonMap {
  return {
    name: spec.shimModel,
    model: spec.shimModel,
    modified_at: new Date().toISOString(),
    size: spec.size,
    digest: `litellm-${spec.upstreamModel}`,
    details: modelDetails(spec),
  };
}

async function availableModelSpecs(): Promise<ModelSpec[]> {
  try {
    const upstream = await litellmFetch("/v1/models");
    if (!upstream.ok) {
      throw new Error(await upstream.text());
    }
    const payload = (await upstream.json()) as { data?: Array<{ id?: string }> };
    const models = Array.isArray(payload.data) ? payload.data : [];
    // Build specs dynamically from what LiteLLM reports, including 1M shim variants.
    // Exclude embedding models (model name contains "embed") — they are not chat models.
    const specs: ModelSpec[] = [];
    const seen = new Set<string>();
    for (const m of models) {
      if (!m.id) continue;
      if (m.id.includes("embed")) continue;
      const spec = getModelSpec(m.id);
      if (!seen.has(spec.shimModel)) {
        seen.add(spec.shimModel);
        specs.push(spec);
      }
      // Include the 1M variant if buildDynamicModels registered one.
      const longShimName = `${spec.upstreamModel}-1m:latest`;
      const longSpec = dynamicModelSpecs.find((s) => s.shimModel === longShimName);
      if (longSpec && !seen.has(longSpec.shimModel)) {
        seen.add(longSpec.shimModel);
        specs.push(longSpec);
      }
    }
    return specs;
  } catch (error) {
    serverLog(`warning: falling back to dynamic model list: ${String(error)}`);
    return getModelSpecs();
  }
}

async function handleApiTags(res: NodeResponse): Promise<void> {
  try {
    const specs = await availableModelSpecs();
    sendJson(res, 200, { models: specs.map(modelTag) });
  } catch (error) {
    sendJson(res, 502, { error: String(error) });
  }
}

async function handleApiShow(req: IncomingMessage, res: NodeResponse): Promise<void> {
  try {
    const body = await readJson(req);
    const spec = getModelSpec((body.name as string | undefined) || (body.model as string | undefined) || getDefaultSpec().shimModel);
    sendJson(res, 200, {
      license: "proprietary",
      modelfile: "# LiteLLM Bedrock shim",
      parameters: "temperature 0.7",
      template: "{{ .Prompt }}",
      capabilities: spec.capabilities,
      remote_model: spec.shimModel,
      details: modelDetails(spec),
      model_info: {
        "general.architecture": spec.family,
        "general.basename": spec.basename,
        [`${spec.family}.context_length`]: spec.contextLength,
      },
      modified_at: new Date().toISOString(),
      name: spec.shimModel,
      model: spec.shimModel,
      size: spec.size,
      digest: `litellm-${spec.upstreamModel}`,
    });
  } catch (error) {
    sendJson(res, 400, { error: String(error) });
  }
}

async function handleApiPs(res: NodeResponse): Promise<void> {
  try {
    const specs = await availableModelSpecs();
    sendJson(res, 200, {
      models: specs.map((spec) => ({
        name: spec.shimModel,
        model: spec.shimModel,
        size: spec.size,
        digest: `litellm-${spec.upstreamModel}`,
        details: modelDetails(spec),
        expires_at: new Date(Date.now() + 60 * 60 * 1000).toISOString(),
        size_vram: spec.size,
      })),
    });
  } catch (error) {
    sendJson(res, 502, { error: String(error) });
  }
}

async function handleApiChat(req: IncomingMessage, res: NodeResponse): Promise<void> {
  const start = Date.now();
  try {
    const body = await readJson(req);
    const readDone = Date.now();
    const spec = getModelSpec(body.model as string | undefined);
    const stream = body.stream !== false;
    serverLog(`POST /api/chat model=${spec.shimModel} stream=${stream}`);

    const payload: JsonMap = {
      model: spec.upstreamModel,
      messages: Array.isArray(body.messages) ? body.messages : [],
      stream,
      tools: spec.capabilities.includes("tools") ? body.tools : undefined,
      options: body.options,
      format: body.format,
      keep_alive: body.keep_alive,
    };
    sanitizeSamplingParams(payload, spec);

    const upstream = await litellmFetch("/v1/chat/completions", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    const ttfb = Date.now();

    if (!upstream.ok) {
      const text = await upstream.text();
      serverLog(`POST /api/chat model=${spec.shimModel} error=${upstream.status} total=${ms(start)} read=${readDone - start}ms ttfb=${ttfb - readDone}ms`);
      sendJson(res, upstream.status, { error: text });
      return;
    }

    if (payload.stream) {
      await streamOpenAiToOllama(upstream, res, spec, "chat", { label: `POST /api/chat model=${spec.shimModel}`, start, ttfb, readDone });
      return;
    }

    const json = (await upstream.json()) as OpenAIResponsePayload;
    const u = json.usage;
    const done = Date.now();
    const proc = (readDone - start) + (done - ttfb);
    serverLog(
      `POST /api/chat model=${spec.shimModel} done total=${ms(start)} read=${readDone - start}ms ttfb=${ttfb - readDone}ms proc=${proc}ms${u ? ` prompt=${u.prompt_tokens ?? "?"} completion=${u.completion_tokens ?? "?"}` : ""}`
    );
    sendJson(res, 200, ollamaChatFromOpenAI(json, spec));
  } catch (error) {
    serverLog(`POST /api/chat error total=${ms(start)} ${String(error)}`);
    sendJson(res, 502, { error: String(error) });
  }
}

async function handleApiGenerate(req: IncomingMessage, res: NodeResponse): Promise<void> {
  const start = Date.now();
  try {
    const body = await readJson(req);
    const readDone = Date.now();
    const spec = getModelSpec(body.model as string | undefined);
    const stream = body.stream !== false;
    serverLog(`POST /api/generate model=${spec.shimModel} stream=${stream}`);

    const prompt = [
      body.system ? { role: "system", content: body.system } : null,
      body.prompt ? { role: "user", content: body.prompt } : null,
    ].filter((value): value is { role: string; content: unknown } => value !== null);

    const payload: JsonMap = {
      model: spec.upstreamModel,
      messages: prompt,
      stream,
      options: body.options,
    };
    sanitizeSamplingParams(payload, spec);

    const upstream = await litellmFetch("/v1/chat/completions", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    const ttfb = Date.now();

    if (!upstream.ok) {
      const text = await upstream.text();
      serverLog(`POST /api/generate model=${spec.shimModel} error=${upstream.status} total=${ms(start)} read=${readDone - start}ms ttfb=${ttfb - readDone}ms`);
      sendJson(res, upstream.status, { error: text });
      return;
    }

    if (payload.stream) {
      await streamOpenAiToOllama(upstream, res, spec, "generate", { label: `POST /api/generate model=${spec.shimModel}`, start, ttfb, readDone });
      return;
    }

    const json = (await upstream.json()) as OpenAIResponsePayload;
    const u = json.usage;
    const done = Date.now();
    const proc = (readDone - start) + (done - ttfb);
    serverLog(
      `POST /api/generate model=${spec.shimModel} done total=${ms(start)} read=${readDone - start}ms ttfb=${ttfb - readDone}ms proc=${proc}ms${u ? ` prompt=${u.prompt_tokens ?? "?"} completion=${u.completion_tokens ?? "?"}` : ""}`
    );
    sendJson(res, 200, ollamaGenerateFromOpenAI(json, spec));
  } catch (error) {
    serverLog(`POST /api/generate error total=${ms(start)} ${String(error)}`);
    sendJson(res, 502, { error: String(error) });
  }
}

async function proxyV1Request(req: IncomingMessage, res: NodeResponse, url: URL): Promise<void> {
  const start = Date.now();
  let model = "";
  try {
    const headers = new Headers();
    for (const [key, value] of Object.entries(req.headers)) {
      if (typeof value === "string" && key.toLowerCase() !== "host") {
        headers.set(key, value);
      }
    }
    headers.set("authorization", `Bearer ${config.apiKey}`);

    let body: string | undefined;
    let duplex: "half" | undefined;
    let readDone = start;
    if (req.method !== "GET" && req.method !== "HEAD") {
      const rawBody = await readBodyText(req);
      readDone = Date.now();
      body = rawBody;
      duplex = "half";

      if (
        rawBody &&
        headers.get("content-type")?.includes("application/json") &&
        (url.pathname === "/v1/chat/completions" || url.pathname === "/v1/responses")
      ) {
        try {
          const parsed = JSON.parse(rawBody) as JsonMap;
          const spec = getModelSpec((parsed.model as string | undefined) || getDefaultSpec().shimModel);
          model = spec.shimModel;
          parsed.model = spec.upstreamModel;
          if (!spec.capabilities.includes("tools")) {
            delete parsed.tools;
            delete parsed.tool_choice;
          }
          sanitizeSamplingParams(parsed, spec);
          body = JSON.stringify(parsed);
          headers.set("content-length", Buffer.byteLength(body).toString());
        } catch (error) {
          serverLog(`warning: failed to rewrite ${url.pathname} model: ${String(error)}`);
        }
      }
    }

    serverLog(`proxy ${req.method} ${url.pathname}${url.search}${model ? ` model=${model}` : ""}`);

    const init: NodeFetchInit = {
      method: req.method,
      headers,
      body,
      duplex,
    };
    const upstream = await fetch(`http://127.0.0.1:${config.ports.litellm}${url.pathname}${url.search}`, init);
    const ttfb = Date.now();
    await proxyStream(upstream, res);
    const done = Date.now();
    const proc = (readDone - start) + (done - ttfb);
    serverLog(`proxy ${req.method} ${url.pathname} done total=${ms(start)} read=${readDone - start}ms ttfb=${ttfb - readDone}ms stream=${done - ttfb}ms proc=${proc}ms status=${upstream.status}${model ? ` model=${model}` : ""}`);
  } catch (error) {
    serverLog(`proxy ${req.method} ${url.pathname} error total=${ms(start)} ${String(error)}`);
    sendJson(res, 502, { error: String(error) });
  }
}

async function handleFrontRequest(req: IncomingMessage, res: NodeResponse): Promise<void> {
  const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);

  if (req.method === "GET" && url.pathname === "/health") {
    sendJson(res, 200, { status: "ok" });
    return;
  }

  if (url.pathname.startsWith("/v1/")) {
    await proxyV1Request(req, res, url);
    return;
  }

  serverLog(`front ${req.method} ${url.pathname}${url.search}`);

  if (url.pathname === "/") {
    sendJson(res, 200, { service: "bedllama-front" });
    return;
  }

  sendJson(res, 404, { error: `Unsupported path: ${req.method} ${url.pathname}` });
}

async function handleOllamaRequest(req: IncomingMessage, res: NodeResponse): Promise<void> {
  const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);

  if (req.method === "GET" && url.pathname === "/") {
    sendText(res, 200, "bedllama\n");
    return;
  }
  if (req.method === "GET" && url.pathname === "/api/version") {
    sendJson(res, 200, { version: config.ollamaVersion });
    return;
  }
  if (req.method === "GET" && url.pathname === "/api/tags") {
    serverLog(`ollama GET /api/tags`);
    await handleApiTags(res);
    return;
  }
  if (req.method === "GET" && url.pathname === "/api/ps") {
    serverLog(`ollama GET /api/ps`);
    await handleApiPs(res);
    return;
  }
  if (req.method === "POST" && url.pathname === "/api/show") {
    serverLog(`ollama POST /api/show`);
    await handleApiShow(req, res);
    return;
  }
  if (req.method === "POST" && url.pathname === "/api/chat") {
    await handleApiChat(req, res);
    return;
  }
  if (req.method === "POST" && url.pathname === "/api/generate") {
    await handleApiGenerate(req, res);
    return;
  }
  if (url.pathname.startsWith("/v1/")) {
    await proxyV1Request(req, res, url);
    return;
  }

  sendJson(res, 404, { error: `Unsupported path: ${req.method} ${url.pathname}` });
}

function startIntegratedServer(onShutdown?: () => void): void {
  // Load model specs passed from the parent `start` process.
  const specsEnv = process.env.BEDLLAMA_MODEL_SPECS;
  if (specsEnv) {
    try {
      dynamicModelSpecs = JSON.parse(specsEnv) as ModelSpec[];
    } catch {
      serverLog("warning: failed to parse BEDLLAMA_MODEL_SPECS");
    }
  }

  const frontServer = http.createServer((req, res) => {
    void handleFrontRequest(req, res).catch((error) => {
      sendJson(res, 502, { error: String(error) });
    });
  });

  const ollamaServer = http.createServer((req, res) => {
    void handleOllamaRequest(req, res).catch((error) => {
      sendJson(res, 502, { error: String(error) });
    });
  });

  const closeAll = (): void => {
    onShutdown?.();
    frontServer.close();
    ollamaServer.close();
    setTimeout(() => process.exit(0), 50);
  };

  process.on("SIGINT", closeAll);
  process.on("SIGTERM", closeAll);

  let listeningCount = 0;
  const onListening = (): void => {
    listeningCount += 1;
    if (listeningCount === 2) {
      serverLog(
        `serving front=http://${config.hosts.front}:${config.ports.front} ollama=http://${config.hosts.ollama}:${config.ports.ollama}`
      );
    }
  };

  frontServer.listen(config.ports.front, config.hosts.front, onListening);
  ollamaServer.listen(config.ports.ollama, config.hosts.ollama, onListening);
}

async function runStack(): Promise<void> {
  requireDependencies();

  process.stdout.write("Fetching available Bedrock model capabilities...\n");
  const capabilityMap = fetchFoundationModelCapabilities();

  process.stdout.write("Fetching available Bedrock inference profiles...\n");
  const allProfileIds = fetchBedrockInferenceProfiles(capabilityMap);

  const profileIds = config.models.length > 0
    ? (() => {
        const allowed = new Set(
          config.models.map((m) => (m.endsWith(":latest") ? m.slice(0, -7) : m))
        );
        return allProfileIds.filter((id) => allowed.has(deriveModelNameFromProfileId(id)));
      })()
    : allProfileIds;

  if (profileIds.length === 0) {
    fail("No Bedrock inference profiles found. Check your AWS credentials and region.");
  }

  process.stdout.write(`Probing output limits for ${profileIds.length} model(s) in parallel...\n`);
  const maxOutputMap = await probeAllMaxOutputTokens(profileIds);

  const { modelSpecs, litellmModelMap } = buildDynamicModels(profileIds, capabilityMap, maxOutputMap);
  dynamicModelSpecs = modelSpecs;
  dynamicLitellmModelMap = litellmModelMap;

  if (config.adminUi) {
    startPostgres();
    process.stdout.write("Waiting for PostgreSQL to be ready...\n");
    await waitForTcp("127.0.0.1", config.postgresPort);
    await waitForPostgresReady();
    prismaDbPush();
  }

  writeLitellmConfig();

  // Start LiteLLM as a tracked child so systemd sees the full process tree.
  const litellmChild = spawn(config.litellmBin, litellmArgs(), {
    env: litellmEnv(),
    stdio: "inherit",
    detached: false,
  });

  litellmChild.on("exit", (code, signal) => {
    process.stderr.write(`LiteLLM exited unexpectedly (code=${code} signal=${signal}), shutting down\n`);
    if (config.adminUi) stopPostgres();
    process.exit(1); // non-zero so systemd Restart=on-failure triggers
  });

  try {
    await waitForHttp(`http://127.0.0.1:${config.ports.litellm}/v1/models`, {
      Authorization: `Bearer ${config.apiKey}`,
    });
  } catch (error) {
    const litellmLog = tailLines(logFilePath(config.processNames.litellm), 40);
    litellmChild.kill("SIGTERM");
    if (config.adminUi) stopPostgres();
    if (litellmLog) {
      process.stderr.write("\nRecent LiteLLM log output:\n");
      process.stderr.write(`${litellmLog}\n`);
    }
    fail(String(error));
  }

  process.stdout.write(`bedllama ready\n`);
  process.stdout.write(`  OpenAI: http://${config.hosts.front}:${config.ports.front}/v1\n`);
  process.stdout.write(`  Ollama: http://${config.hosts.ollama}:${config.ports.ollama}\n`);
  for (const spec of modelSpecs) {
    process.stdout.write(`  model: ${spec.shimModel}\n`);
  }

  const onShutdown = (): void => {
    // Detach the unexpected-exit handler so killing litellm intentionally
    // doesn't trigger a non-zero exit from this process.
    litellmChild.removeAllListeners("exit");
    litellmChild.kill("SIGTERM");
    if (config.adminUi) stopPostgres();
  };

  startIntegratedServer(onShutdown);
}

function installService(): void {
  if (process.platform === "win32") {
    fail("systemd is not available on Windows. Use WSL2 or run bedllama start manually.");
  }

  const hasSystemctl = run("which", ["systemctl"]).status === 0;
  if (!hasSystemctl) {
    fail("systemctl not found — is systemd running on this system?");
  }

  // Prefer the linked bedllama bin from PATH; fall back to node + this script.
  const whichResult = run("which", ["bedllama"]);
  const execStart = whichResult.status === 0 && whichResult.stdout.trim()
    ? `${whichResult.stdout.trim()} run`
    : `${process.execPath} ${__filename} run`;

  const serviceDir = path.join(home, ".config", "systemd", "user");
  const servicePath = path.join(serviceDir, "bedllama.service");

  const unit = [
    "[Unit]",
    "Description=bedllama — AWS Bedrock LLM proxy (OpenAI + Ollama compatible)",
    "After=network-online.target",
    "Wants=network-online.target",
    "",
    "[Service]",
    "Type=simple",
    `ExecStart=${execStart}`,
    "Restart=on-failure",
    "RestartSec=10",
    "TimeoutStartSec=120",
    "StandardOutput=journal",
    "StandardError=journal",
    "SyslogIdentifier=bedllama",
    "",
    "[Install]",
    "WantedBy=default.target",
    "",
  ].join("\n");

  mkdirSync(serviceDir, { recursive: true });
  writeFileSync(servicePath, unit, { encoding: "utf8", mode: 0o644 });

  run("systemctl", ["--user", "daemon-reload"]);

  process.stdout.write(`Installed: ${servicePath}\n`);
  process.stdout.write(`ExecStart: ${execStart}\n`);
  process.stdout.write("\nNext steps:\n");
  process.stdout.write("  systemctl --user enable --now bedllama   # start now + on login\n");
  process.stdout.write("  systemctl --user status bedllama          # check state\n");
  process.stdout.write("  journalctl --user -fu bedllama            # follow logs\n");
}

function uninstallService(): void {
  const servicePath = path.join(home, ".config", "systemd", "user", "bedllama.service");

  // Stop and disable — ignore errors (service may not be enabled/running).
  run("systemctl", ["--user", "disable", "--now", "bedllama"]);

  if (existsSync(servicePath)) {
    unlinkSync(servicePath);
    process.stdout.write(`Removed: ${servicePath}\n`);
  } else {
    process.stdout.write("Service file not found (already uninstalled?)\n");
  }

  run("systemctl", ["--user", "daemon-reload"]);
  process.stdout.write("bedllama service uninstalled\n");
}

// ---------------------------------------------------------------------------
// VS Code integration
// ---------------------------------------------------------------------------

function findVSCodeUserDirs(): string[] {
  const dirs: string[] = [];
  const appNames = ["Code", "Code - Insiders"];
  if (process.platform === "darwin") {
    const appSupport = path.join(home, "Library", "Application Support");
    for (const name of appNames) {
      const d = path.join(appSupport, name, "User");
      if (existsSync(d)) dirs.push(d);
    }
  } else {
    const xdgConfig = process.env.XDG_CONFIG_HOME ?? path.join(home, ".config");
    for (const name of appNames) {
      const d = path.join(xdgConfig, name, "User");
      if (existsSync(d)) dirs.push(d);
    }
  }
  return dirs;
}

/** Extract the minor version from a Claude basename, e.g. "claude-opus-4-6" → 6, "claude-sonnet-4" → 0. */
function claudeMinorVersion(basename: string): number {
  const m = basename.replace(/-1m$/, "").match(/(\d+)-(\d+)$/);
  return m ? parseInt(m[2], 10) : 0;
}

function formatModelDisplayName(basename: string, contextLength: number): string {
  // claude-sonnet-4-6  → "Claude Sonnet 4.6"
  // claude-opus-4-7-1m → "Claude Opus 4.7 · 1M"
  const m = basename.match(/^claude-([a-z]+)-(\d+)-(\d+)(?:-1m)?$/);
  if (m) {
    const tierCap = m[1].charAt(0).toUpperCase() + m[1].slice(1);
    const suffix = contextLength >= 1_000_000 ? " \xb7 1M" : "";
    return `Claude ${tierCap} ${m[2]}.${m[3]}${suffix}`;
  }
  // Generic: capitalise each hyphen-separated word
  return basename.split("-").map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
}

function maxOutputTokensForModel(basename: string): number {
  if (basename.includes("opus")) return 32_000;
  if (basename.startsWith("claude")) return 64_000;
  return 32_000;
}

// specFromModelId is kept for backward-compat but now populates maxOutputTokens.
function specFromModelId(modelId: string): ModelSpec {
  const basename = modelId.endsWith(":latest") ? modelId.slice(0, -7) : modelId;
  const is1M = basename.endsWith("-1m");
  return {
    upstreamModel: basename,
    shimModel: `${basename}:latest`,
    basename,
    family: "unknown",
    parameterSize: basename,
    contextLength: is1M ? 1_000_000 : 200_000,
    capabilities: NO_TOOLS_MODELS.some((f) => basename.includes(f)) ? [] : ["tools", "vision"],
    size: 0,
    maxOutputTokens: 32_000,
  };
}

async function configureVSCode(): Promise<void> {
  requireDependencies();

  const frontUrl = `http://${config.hosts.front}:${config.ports.front}`;

  // Always fetch inference profiles — needed for probing regardless of whether
  // bedllama is running. This is a fast list call (~1s).
  process.stdout.write("Fetching Bedrock inference profiles...\n");
  const capabilityMap = fetchFoundationModelCapabilities();
  const allProfileIds = fetchBedrockInferenceProfiles(capabilityMap);
  const profileIds = config.models.length > 0
    ? (() => {
        const allowed = new Set(
          config.models.map((m) => (m.endsWith(":latest") ? m.slice(0, -7) : m))
        );
        return allProfileIds.filter((id) => allowed.has(deriveModelNameFromProfileId(id)));
      })()
    : allProfileIds;
  if (profileIds.length === 0) {
    fail("No Bedrock inference profiles found. Check your AWS credentials and region.");
  }

  // Probe max output tokens for all models in parallel.
  process.stdout.write(`Probing output limits for ${profileIds.length} model(s)...\n`);
  const maxOutputMap = await probeAllMaxOutputTokens(profileIds);

  // Build full model specs with probe data (includes 1M variants).
  const { modelSpecs } = buildDynamicModels(profileIds, capabilityMap, maxOutputMap);

  // If bedllama is running, filter to only the models it currently serves
  // (respects config.models filter and shows the same set VS Code will use).
  let finalSpecs = modelSpecs;
  try {
    const response = await fetch(`${frontUrl}/v1/models`, {
      headers: { Authorization: `Bearer ${config.apiKey}` },
      signal: AbortSignal.timeout(3000),
    });
    if (response.ok) {
      const payload = await response.json() as { data?: Array<{ id: string }> };
      const liveIds = new Set(
        (payload.data ?? []).map((m) => m.id.replace(/:latest$/, ""))
      );
      // Keep specs whose basename is in the live set; also keep specs whose
      // upstreamModel is live (catches 1M variants whose upstreamModel is served).
      finalSpecs = modelSpecs.filter(
        (s) => liveIds.has(s.basename) || liveIds.has(s.upstreamModel)
      );
      process.stdout.write(`Filtered to ${finalSpecs.length} model(s) from running bedllama\n`);
    }
  } catch {
    // bedllama not running — use all discovered specs.
  }

  for (const spec of finalSpecs) {
    const ctx = spec.contextLength >= 1_000_000 ? "1M" : "200K";
    const out = spec.maxOutputTokens.toLocaleString();
    process.stdout.write(`  ${spec.basename.padEnd(28)} ctx=${ctx}  out=${out}\n`);
  }

  const bedllamaEntry = {
    name: "bedllama",
    vendor: "customendpoint",
    apiKey: config.apiKey,
    apiType: "chat-completions",
    models: finalSpecs.map((spec) => {
      const entry: Record<string, unknown> = {
        id: spec.basename,
        name: formatModelDisplayName(spec.basename, spec.contextLength),
        url: `${frontUrl}/v1/chat/completions`,
        toolCalling: spec.capabilities.includes("tools"),
        vision: spec.capabilities.includes("vision"),
        maxInputTokens: (() => {
          // VS Code displays contextSize = maxInputTokens + maxOutputTokens.
          // Set maxInputTokens = targetWindow - maxOutputTokens so the sum
          // equals the intended context window.
          let targetWindow: number;
          if (spec.contextLength >= 1_000_000) {
            // Already a 1M variant.
            targetWindow = 1_000_000;
          } else {
            const has1MSibling = finalSpecs.some(
              (s) => s.basename === `${spec.basename}-1m`
            );
            if (has1MSibling) {
              // Base entry of a 200K/1M pair — keep this one at 200K.
              targetWindow = 200_000;
            } else if (spec.family === "claude" && claudeMinorVersion(spec.basename) >= 6) {
              // Claude 4.6+ natively supports 1M on Bedrock even without a
              // separate -1m variant in the registry.
              targetWindow = 1_000_000;
            } else {
              targetWindow = spec.contextLength; // 200K default
            }
          }
          return Math.max(1, targetWindow - spec.maxOutputTokens);
        })(),
        maxOutputTokens: spec.maxOutputTokens,
      };
      // Expose thinking + reasoning effort for high-output Anthropic models.
      // LiteLLM translates reasoning_effort → thinking.budget_tokens (4.5) or
      // thinking.type="adaptive" (4.6+) before forwarding to Bedrock Converse.
      if (spec.family === "claude" && spec.maxOutputTokens >= LONG_CONTEXT_OUTPUT_THRESHOLD) {
        entry.thinking = true;
        entry.supportsReasoningEffort = ["low", "medium", "high"];
        entry.reasoningEffortFormat = "chat-completions";
      }
      return entry;
    }),
  };

  const userDirs = findVSCodeUserDirs();
  if (userDirs.length === 0) {
    process.stderr.write("No VS Code User directories found. Is VS Code installed?\n");
    process.exit(1);
  }

  for (const userDir of userDirs) {
    const configPath = path.join(userDir, "chatLanguageModels.json");
    let existing: Record<string, unknown>[] = [];
    if (existsSync(configPath)) {
      try {
        const parsed = JSON.parse(readFileSync(configPath, "utf8")) as unknown;
        existing = Array.isArray(parsed) ? (parsed as Record<string, unknown>[]) : [];
      } catch {
        existing = [];
      }
    }
    // Replace the bedllama customendpoint entry; preserve everything else.
    const filtered = existing.filter(
      (e) => !(e.vendor === "customendpoint" && e.name === "bedllama")
    );
    filtered.push(bedllamaEntry as unknown as Record<string, unknown>);
    writeFileSync(configPath, JSON.stringify(filtered, null, "\t") + "\n", "utf8");
    process.stdout.write(`Updated: ${configPath}\n`);
  }
}

function legend(): void {
  process.stdout.write(`bedllama log field reference

request phases:
  read=Xms    time to receive the full request body from the client
  ttfb=Xms    time-to-first-byte: from sending to LiteLLM until the upstream
              response headers arrive (includes LiteLLM overhead + Bedrock latency)
  stream=Xms  time to drain all response chunks back to the client
  proc=Xms    pure bedllama processing time (read + stream, excludes waiting for upstream)
              baseline ~2-20ms at home; spikes here on corp = packet inspection / proxy
  total=Xms   wall-clock time for the entire request

request fields:
  model=      Ollama shim model name (e.g. claude-sonnet-4-6:latest)
  stream=     whether the response was streamed (true/false)
  status=     HTTP status code returned by LiteLLM

token counts (on completion):
  prompt=     number of input tokens billed
  completion= number of output tokens generated

log prefixes:
  proxy       OpenAI-compatible request via the front server (:4000/v1/...)
  ollama      Ollama-compatible request (:11434/api/...)
  [bedllama]  internal server lifecycle messages

overhead chain reference (normal home conditions):
  client -> bedllama (proc ~2-20ms)
          -> LiteLLM local loopback (~5-20ms, included in ttfb)
          -> AWS Bedrock eu-west-3 (ttfb ~2000-4500ms for 98K context)
  total overhead of bedllama+LiteLLM: <1% of total request time
  if proc spikes above ~100ms on corp: a proxy/DPI is inspecting the payload
`);
}

function usage(exitCode = 0): never {
  const text = `usage: bedllama <command>

commands:
  init      create a default config.jsonc in the current directory
  start     start the Bedrock -> LiteLLM -> compatibility stack in background
  stop      stop related helper processes
  restart   stop then start the stack
  status    show process state and endpoint URLs
  logs      show recent logs; optional target: litellm|front|ollama|server|all
            use -f or --follow to tail and follow new output
  legend    explain log field acronyms and prefixes

systemd:
  run       run the full stack in the foreground (used by systemd ExecStart)
  install   install ~/.config/systemd/user/bedllama.service and reload daemon
  uninstall disable and remove the systemd user service

integrations:
  vscode    update VS Code chatLanguageModels.json with bedllama custom endpoint
            fetches models from a running bedllama, or discovers from Bedrock
`;
  if (exitCode === 0) {
    process.stdout.write(text);
  } else {
    process.stderr.write(text);
  }
  process.exit(exitCode);
}

function initConfig(): void {
  const dest = path.join(process.cwd(), "config.jsonc");
  if (existsSync(dest)) {
    process.stderr.write(`config.jsonc already exists at ${dest}\n`);
    process.exit(1);
  }
  const content = `{
  // bedllama configuration file
  // All fields are optional — values here override built-in defaults.
  // Environment variables still take precedence over this file.

  // ── LiteLLM admin UI ────────────────────────────────────────────────────────
  // Set adminUi to true to enable the LiteLLM web dashboard.
  // The UI is served by LiteLLM itself on the litellm port (default 4001).
  // Access it at: http://127.0.0.1:<litellmPort>/ui
  // Requires a PostgreSQL container — bedllama will start one automatically
  // using docker or podman (whichever is available on your PATH).
  "adminUi": false,

  // Username and password for the LiteLLM admin UI login.
  "adminUiUsername": "admin",
  "adminUiPassword": "bedllama-admin"

  // ── PostgreSQL (required when adminUi is true) ───────────────────────────────
  // Host port the postgres container is published on (default 5432).
  // "postgresPort": 5432,
  // Password for the litellm postgres user (default "bedllama").
  // "postgresPassword": "bedllama",

  // ── Ports ───────────────────────────────────────────────────────────────────
  // "litellmPort": 4001,
  // "frontPort": 4000,
  // "ollamaPort": 11434,

  // ── AWS ─────────────────────────────────────────────────────────────────────
  // "awsProfile": "bedrock",
  // "awsRegion": "eu-west-3",

  // ── API key ─────────────────────────────────────────────────────────────────
  // "apiKey": "sk-local",

  // ── Models ──────────────────────────────────────────────────────────────────
  // Comma-separated list of Ollama model tags to expose.
  // "models": "claude-sonnet-4-6:latest,claude-opus-4-7:latest"
}
`;
  writeFileSync(dest, content, { encoding: "utf8", mode: 0o644 });
  process.stdout.write(`Created ${dest}\n`);
}

const command = process.argv[2] || "start";

switch (command) {
  case "start":
    void startStack().catch((error) => fail(String(error)));
    break;
  case "stop":
    stopStack();
    break;
  case "restart":
    stopStack({ quiet: true });
    void startStack().catch((error) => fail(String(error)));
    break;
  case "status":
    statusStack();
    break;
  case "logs": {
    const logsArgs = process.argv.slice(3);
    const follow = logsArgs.includes("-f") || logsArgs.includes("--follow");
    const target = logsArgs.find((a) => !a.startsWith("-")) as LogTarget | undefined;
    logsStack(target || "all", follow);
    break;
  }
  case "legend":
    legend();
    break;
  case "init":
    initConfig();
    break;
  case "serve":
    startIntegratedServer();
    break;
  case "run":
    void runStack().catch((error) => fail(String(error)));
    break;
  case "install":
    installService();
    break;
  case "uninstall":
    uninstallService();
    break;
  case "vscode":
    void configureVSCode().catch((error) => fail(String(error)));
    break;
  case "help":
  case "--help":
  case "-h":
    usage(0);
    break;
  default:
    usage(1);
}
