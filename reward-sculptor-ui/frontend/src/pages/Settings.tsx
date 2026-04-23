import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Database,
  ExternalLink,
  Folder,
  Key,
  Moon,
  Palette,
  RefreshCw,
  Sun,
  SunMoon,
  XCircle,
} from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useSystemInfo } from "@/hooks/useDashboard";
import { useSharedKgStats, useSystemGpu } from "@/hooks/useLibrary";
import { useTheme, type Theme } from "@/hooks/useTheme";
import { cn } from "@/lib/utils";
import type { GpuDevice } from "@/lib/types";

export default function Settings() {
  const { data, isLoading, error } = useSystemInfo();
  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col gap-4 px-6 pb-16 pt-6">
      <header>
        <h1 className="text-lg font-semibold tracking-tight">Settings</h1>
        <p className="text-xs text-muted-foreground">
          Environment + theme + local cache controls
        </p>
      </header>

      {isLoading && <p className="text-sm text-muted-foreground">Loading…</p>}
      {error && (
        <Card className="border-destructive/40 bg-destructive/5">
          <CardContent className="p-4 text-sm text-destructive">
            Could not load system info: {(error as Error).message}
          </CardContent>
        </Card>
      )}
      {data && (
        <>
          <ApiKeyCard
            isSet={data.anthropic_api_key_set}
            masked={data.anthropic_api_key_masked}
          />
          <GpuCard />
          <SharedKgCard />
          <PathsCard
            projectsRoot={data.projects_root}
            sculptorPath={data.sculptor_path}
            sculptorOk={data.sculptor_ok}
            sculptorError={data.sculptor_error}
          />
        </>
      )}

      <ThemeCard />
      <ResetCacheCard />
    </div>
  );
}

// ── API key ──────────────────────────────────────────────────────────
function ApiKeyCard({
  isSet,
  masked,
}: {
  isSet: boolean;
  masked: string | null;
}) {
  return (
    <Card>
      <CardHeader className="py-3">
        <CardTitle className="flex items-center gap-2 text-sm">
          <Key className="h-3.5 w-3.5" />
          Anthropic API key
        </CardTitle>
        <CardDescription className="text-[11px]">
          Required for live sculpt runs. <code>--dry-run</code> works
          without it.
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-0">
        {isSet ? (
          <div className="flex items-center gap-2 text-xs">
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
            <span>Set</span>
            <code className="ml-auto font-mono text-[11px] text-muted-foreground">
              {masked}
            </code>
          </div>
        ) : (
          <div className="flex flex-col gap-2 text-xs">
            <div className="flex items-center gap-2">
              <XCircle className="h-3.5 w-3.5 text-rose-600" />
              <span>Not set — only dry-run sculpt is available.</span>
            </div>
            <p className="text-muted-foreground">
              Set <code>ANTHROPIC_API_KEY</code> in your shell
              environment OR in <code>../RewardSculptor/.env</code>,
              then restart the backend.
            </p>
            <a
              href="https://console.anthropic.com/settings/keys"
              target="_blank"
              rel="noreferrer noopener"
              className="inline-flex items-center gap-1 self-start text-foreground underline-offset-2 hover:underline"
            >
              Get a key at console.anthropic.com
              <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ── GPU ──────────────────────────────────────────────────────────────
/** Live GPU panel (M4). Refreshes every 5s while Settings is mounted.
 *  Data source: GET /system/gpu, which is pynvml-backed + 2s-cached. */
function GpuCard() {
  const { data, isLoading } = useSystemGpu({ refetchIntervalMs: 5000 });

  if (isLoading && !data) {
    return (
      <Card>
        <CardHeader className="py-3">
          <CardTitle className="flex items-center gap-2 text-sm">
            <Cpu className="h-3.5 w-3.5" /> GPU & training
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-0 text-xs text-muted-foreground">
          Loading GPU info…
        </CardContent>
      </Card>
    );
  }

  const cudaOk = !!data?.cuda_available;
  const versionOk = !!data?.cuda_version_ok;
  const banner = !cudaOk
    ? {
        tone: "warning" as const,
        title: "No NVIDIA GPU detected",
        body:
          "mjlab adapter unavailable. Gymnasium adapter still works on CPU.",
      }
    : !versionOk
      ? {
          tone: "warning" as const,
          title: "CUDA version too old for mjlab",
          body: `Detected CUDA ${data?.cuda_version ?? "unknown"}. mjlab requires 12.4+.`,
        }
      : null;

  return (
    <Card>
      <CardHeader className="py-3">
        <CardTitle className="flex items-center gap-2 text-sm">
          <Cpu className="h-3.5 w-3.5" /> GPU & training
        </CardTitle>
        <CardDescription className="text-[11px]">
          Refreshes every 5 s. Telemetry via{" "}
          {data?.pynvml_available ? "pynvml" : "torch.cuda (no utilization)"}.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-2 pt-0 text-xs">
        {banner && (
          <div
            className={cn(
              "flex items-start gap-2 rounded-md border p-2",
              banner.tone === "warning"
                ? "border-amber-500/40 bg-amber-500/5"
                : "border-destructive/40 bg-destructive/5",
            )}
          >
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-600" />
            <div>
              <div className="font-medium">{banner.title}</div>
              <p className="text-muted-foreground">{banner.body}</p>
            </div>
          </div>
        )}

        <KV k="torch" v={data?.torch_version ?? "—"} />
        <KV
          k="cuda runtime"
          v={
            data?.cuda_version
              ? `${data.cuda_version} ${versionOk ? "✓" : "(needs 12.4+)"}`
              : "—"
          }
        />
        {data?.driver_version && (
          <KV k="driver" v={data.driver_version} />
        )}
        <div className="flex items-center gap-3 pt-1">
          <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
            adapters
          </span>
          <Status label="mjlab" ok={!!data?.mjlab_available} />
          <Status label="mujoco_warp" ok={!!data?.mujoco_warp_available} />
          <Status label="rsl_rl" ok={!!data?.rsl_rl_available} />
        </div>

        {(data?.devices ?? []).length > 0 && (
          <div className="mt-2 flex flex-col gap-2">
            {(data?.devices ?? []).map((d) => (
              <GpuDeviceRow key={d.index} device={d} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function Status({ label, ok }: { label: string; ok: boolean }) {
  return (
    <span className="inline-flex items-center gap-1 text-[11px]">
      <span
        className={cn(
          "inline-block h-1.5 w-1.5 rounded-full",
          ok ? "bg-emerald-500" : "bg-muted-foreground/40",
        )}
      />
      {label}
    </span>
  );
}

function GpuDeviceRow({ device }: { device: GpuDevice }) {
  const total = device.total_memory_bytes;
  const free = device.free_memory_bytes;
  const used = device.used_memory_bytes ?? total - free;
  const usedGb = used / (1024 ** 3);
  const totalGb = total / (1024 ** 3);
  const memPct = total > 0 ? (used / total) * 100 : 0;
  const utilPct = device.utilization_percent ?? 0;

  return (
    <div className="rounded-md border bg-muted/20 p-2">
      <div className="flex items-baseline justify-between gap-2 pb-1.5">
        <span className="text-sm font-medium">{device.name}</span>
        <span className="font-mono text-[10px] text-muted-foreground">
          sm_{device.major}{device.minor}
          {device.temperature_c != null && ` · ${Math.round(device.temperature_c)}°C`}
        </span>
      </div>
      <div className="flex items-center gap-2">
        <span className="w-20 text-[10px] uppercase tracking-wider text-muted-foreground">
          memory
        </span>
        <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
          <div
            className={cn(
              "absolute inset-y-0 left-0 transition-[width]",
              memPct > 85 ? "bg-rose-500" : memPct > 70 ? "bg-amber-500" : "bg-emerald-500",
            )}
            style={{ width: `${memPct}%` }}
          />
        </div>
        <span className="w-28 text-right font-mono text-[10px]">
          {usedGb.toFixed(1)} / {totalGb.toFixed(1)} GiB
        </span>
      </div>
      {device.utilization_percent != null && (
        <div className="mt-1 flex items-center gap-2">
          <span className="w-20 text-[10px] uppercase tracking-wider text-muted-foreground">
            utilization
          </span>
          <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
            <div
              className="absolute inset-y-0 left-0 bg-sky-500 transition-[width]"
              style={{ width: `${utilPct}%` }}
            />
          </div>
          <span className="w-28 text-right font-mono text-[10px]">
            {utilPct.toFixed(0)} %
          </span>
        </div>
      )}
    </div>
  );
}

// ── shared knowledge graph ───────────────────────────────────────────
function SharedKgCard() {
  const { data, isLoading, error } = useSharedKgStats({
    refetchIntervalMs: 30_000,
  });

  return (
    <Card>
      <CardHeader className="py-3">
        <CardTitle className="flex items-center gap-2 text-sm">
          <Database className="h-3.5 w-3.5" />
          Knowledge graph (shared)
        </CardTitle>
        <CardDescription className="text-[11px]">
          One graph across all your projects. Seeded from the bundled
          pre-extracted papers and grown on demand via the KG tab's
          Research button.
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-0 text-xs">
        {isLoading && (
          <p className="text-muted-foreground">Loading KG stats…</p>
        )}
        {error && (
          <p className="text-destructive">
            Could not load KG stats: {(error as Error).message}
          </p>
        )}
        {data && (
          <div className="space-y-2">
            {!data.db_exists && (
              <div className="flex items-center gap-1.5">
                <AlertTriangle className="h-3.5 w-3.5 text-amber-600" />
                <span>
                  Empty — no DB yet. Restart the backend or run
                  <code className="mx-1 font-mono">
                    RewardSculptor/scripts/regenerate_kg_preextracted.sh
                  </code>
                  to seed.
                </span>
              </div>
            )}
            <div className="grid grid-cols-3 gap-x-3 gap-y-1 font-mono text-[11px]">
              <KgStat label="Papers" value={data.papers} />
              <KgStat label="Techniques" value={data.techniques} />
              <KgStat label="Failure modes" value={data.failure_modes} />
              <KgStat label="Reward comps" value={data.reward_components} />
              <KgStat label="Environments" value={data.environments} />
              <KgStat label="Edges" value={data.edges} />
            </div>
            <KV k="path" v={data.db_path} mono truncate />
            {data.db_size_bytes > 0 && (
              <KV
                k="size"
                v={`${(data.db_size_bytes / 1024).toFixed(1)} KB`}
                mono
              />
            )}
            {data.last_modified && (
              <KV
                k="last modified"
                v={new Date(data.last_modified).toLocaleString()}
              />
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function KgStat({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-baseline justify-between gap-1 border-b border-border/50 py-0.5">
      <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
        {label}
      </span>
      <span className="font-semibold text-foreground">{value}</span>
    </div>
  );
}

// ── paths ────────────────────────────────────────────────────────────
function PathsCard({
  projectsRoot,
  sculptorPath,
  sculptorOk,
  sculptorError,
}: {
  projectsRoot: string;
  sculptorPath: string | null;
  sculptorOk: boolean;
  sculptorError: string | null;
}) {
  return (
    <Card>
      <CardHeader className="py-3">
        <CardTitle className="flex items-center gap-2 text-sm">
          <Folder className="h-3.5 w-3.5" />
          Paths
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-1 pt-0 text-xs">
        <KV k="$RS_PROJECTS_ROOT" v={projectsRoot} mono />
        {sculptorPath && (
          <KV k="sculptor module" v={sculptorPath} mono truncate />
        )}
        <div className="flex items-center gap-1.5 pt-2">
          {sculptorOk ? (
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
          ) : (
            <AlertTriangle className="h-3.5 w-3.5 text-amber-600" />
          )}
          <span>
            sculptor: {sculptorOk ? "importable" : sculptorError ?? "error"}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

// ── theme ────────────────────────────────────────────────────────────
function ThemeCard() {
  const { theme, setTheme } = useTheme();
  const items: Array<{ key: Theme; label: string; icon: React.ComponentType<{ className?: string }> }> = [
    { key: "light", label: "Light", icon: Sun },
    { key: "dark", label: "Dark", icon: Moon },
    { key: "system", label: "System", icon: SunMoon },
  ];
  return (
    <Card>
      <CardHeader className="py-3">
        <CardTitle className="flex items-center gap-2 text-sm">
          <Palette className="h-3.5 w-3.5" />
          Theme
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="inline-flex rounded-md border bg-background p-0.5">
          {items.map((it) => {
            const Icon = it.icon;
            return (
              <button
                key={it.key}
                type="button"
                onClick={() => setTheme(it.key)}
                className={cn(
                  "inline-flex items-center gap-1 rounded-sm px-2.5 py-1 text-xs transition-colors",
                  theme === it.key
                    ? "bg-foreground text-background"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                <Icon className="h-3 w-3" />
                {it.label}
              </button>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

// ── reset cache ──────────────────────────────────────────────────────
function ResetCacheCard() {
  const qc = useQueryClient();
  const [confirming, setConfirming] = useState(false);
  useEffect(() => {
    if (!confirming) return;
    const t = setTimeout(() => setConfirming(false), 5000);
    return () => clearTimeout(t);
  }, [confirming]);

  const reset = () => {
    qc.clear();
    try {
      localStorage.clear();
    } catch {
      /* private mode, ignore */
    }
    toast.success("UI state reset", {
      description: "React Query cache + localStorage cleared.",
    });
    setConfirming(false);
  };

  return (
    <Card>
      <CardHeader className="py-3">
        <CardTitle className="flex items-center gap-2 text-sm">
          <RefreshCw className="h-3.5 w-3.5" />
          Reset UI state
        </CardTitle>
        <CardDescription className="text-[11px]">
          Clears the React Query cache + localStorage. Doesn't touch
          any on-disk project data.
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-0">
        {confirming ? (
          <div className="flex items-center gap-2">
            <Button variant="destructive" size="sm" onClick={reset}>
              Confirm reset
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setConfirming(false)}
            >
              Cancel
            </Button>
          </div>
        ) : (
          <Button
            variant="outline"
            size="sm"
            onClick={() => setConfirming(true)}
          >
            Reset
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

// ── utility ──────────────────────────────────────────────────────────
function KV({
  k,
  v,
  mono,
  truncate,
}: {
  k: string;
  v: string;
  mono?: boolean;
  truncate?: boolean;
}) {
  return (
    <div className="flex items-baseline gap-3">
      <span className="shrink-0 text-[10px] uppercase tracking-wider text-muted-foreground">
        {k}
      </span>
      <span
        className={cn(
          "min-w-0 flex-1",
          mono && "font-mono text-[11px]",
          truncate && "truncate",
        )}
        title={truncate ? v : undefined}
      >
        {v}
      </span>
    </div>
  );
}
