import { useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { Icon } from "@/components/rs/icon";
import { Badge, Btn, Field, Segmented, ToggleRow } from "@/components/rs/primitives";
import { useSystemInfo } from "@/hooks/useDashboard";
import { useRemoteSettings, useSharedKgStats, useSystemGpu } from "@/hooks/useLibrary";
import { useTheme } from "@/hooks/useTheme";
import { putRemoteSettings, runRemoteDoctor } from "@/lib/api";
import type { GpuDevice, RemoteDoctorResponse, RemoteSettings } from "@/lib/types";

export default function Settings() {
  const { data, isLoading, error } = useSystemInfo();
  return (
    <div className="rs-scroll">
      <div className="rs-pad rs-vgap-24" style={{ maxWidth: 760 }}>
        <div>
          <div className="rs-eyebrow">Localhost configuration</div>
          <h2 className="rs-h2" style={{ marginTop: 6 }}>Settings</h2>
        </div>

        {isLoading && <p className="rs-sub">Loading…</p>}
        {error && (
          <div className="rs-banner err">
            <Icon name="alert-triangle" size={17} />
            <span className="rs-grow">Could not load system info: {(error as Error).message}</span>
          </div>
        )}
        {data && (
          <>
            <ApiKeyCard isSet={data.anthropic_api_key_set} masked={data.anthropic_api_key_masked} />
            <GpuCard />
            <RemoteGpuCard />
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
    </div>
  );
}

function KvRow({ k, v, mono, truncate }: { k: string; v: string; mono?: boolean; truncate?: boolean }) {
  return (
    <>
      <div className="k">{k}</div>
      <div
        className="v"
        style={{
          fontFamily: mono ? "var(--font-mono)" : undefined,
          overflow: truncate ? "hidden" : undefined,
          textOverflow: truncate ? "ellipsis" : undefined,
          whiteSpace: truncate ? "nowrap" : undefined,
        }}
        title={truncate ? v : undefined}
      >
        {v}
      </div>
    </>
  );
}

// ── API key (read-only — no write endpoint; honest per Finding B) ─────
function ApiKeyCard({ isSet, masked }: { isSet: boolean; masked: string | null }) {
  return (
    <div className="rs-card">
      <div className="rs-card-head">
        <div className="rs-card-title"><Icon name="key" size={16} />Anthropic API</div>
        <Badge status={isSet ? "completed" : "errored"} label={isSet ? "Connected" : "Not set"} />
      </div>
      <div className="rs-card-pad rs-vgap-8">
        {isSet ? (
          <div className="rs-kv">
            <KvRow k="api key" v={masked ?? "set"} mono />
            <KvRow k="source" v="ANTHROPIC_API_KEY" />
          </div>
        ) : (
          <div className="rs-vgap-8">
            <p className="rs-sub" style={{ margin: 0 }}>
              No key set — only <code className="mono">--dry-run</code> sculpt is available. Set{" "}
              <code className="mono">ANTHROPIC_API_KEY</code> in your shell or in{" "}
              <code className="mono">../RewardSculptor/.env</code>, then restart the backend.
            </p>
            <a
              href="https://console.anthropic.com/settings/keys"
              target="_blank"
              rel="noreferrer noopener"
              className="rs-flex rs-gap-6"
              style={{ fontSize: 13, color: "var(--ink)", width: "fit-content" }}
            >
              Get a key at console.anthropic.com <Icon name="external" size={13} />
            </a>
          </div>
        )}
      </div>
    </div>
  );
}

// ── GPU ──────────────────────────────────────────────────────────────
function GpuCard() {
  const { data, isLoading } = useSystemGpu({ refetchIntervalMs: 5000 });
  if (isLoading && !data) {
    return (
      <div className="rs-card">
        <div className="rs-card-head"><div className="rs-card-title"><Icon name="cpu" size={16} />GPU &amp; training</div></div>
        <div className="rs-card-pad"><p className="rs-sub">Loading GPU info…</p></div>
      </div>
    );
  }
  const cudaOk = !!data?.cuda_available;
  const versionOk = !!data?.cuda_version_ok;
  const banner = !cudaOk
    ? { title: "No NVIDIA GPU detected", body: "mjlab adapter unavailable. Gymnasium adapter still works on CPU." }
    : !versionOk
      ? { title: "CUDA version too old for mjlab", body: `Detected CUDA ${data?.cuda_version ?? "unknown"}. mjlab requires 12.4+.` }
      : null;
  return (
    <div className="rs-card">
      <div className="rs-card-head">
        <div className="rs-card-title"><Icon name="cpu" size={16} />GPU &amp; training</div>
        <span className="rs-sub" style={{ fontSize: 12 }}>
          {data?.pynvml_available ? "pynvml · 5s" : "torch.cuda · 5s"}
        </span>
      </div>
      <div className="rs-card-pad rs-vgap-8">
        {banner && (
          <div className="rs-banner warn">
            <Icon name="alert-triangle" size={17} />
            <span className="rs-grow"><b>{banner.title}.</b> {banner.body}</span>
          </div>
        )}
        <div className="rs-kv">
          <KvRow k="torch" v={data?.torch_version ?? "—"} mono />
          <KvRow k="cuda runtime" v={data?.cuda_version ? `${data.cuda_version}${versionOk ? " ✓" : " (needs 12.4+)"}` : "—"} mono />
          {data?.devices?.[0]?.driver_version && <KvRow k="driver" v={data.devices[0].driver_version!} mono />}
        </div>
        <div className="rs-flex rs-gap-12 rs-wrap" style={{ paddingTop: 4 }}>
          <span className="rs-eyebrow">Adapters</span>
          <StatusDot label="mjlab" ok={!!data?.mjlab_available} />
          <StatusDot label="mujoco_warp" ok={!!data?.mujoco_warp_available} />
          <StatusDot label="rsl_rl" ok={!!data?.rsl_rl_available} />
        </div>
        {(data?.devices ?? []).map((d) => <GpuDeviceRow key={d.index} device={d} />)}
      </div>
    </div>
  );
}

function StatusDot({ label, ok }: { label: string; ok: boolean }) {
  return (
    <span className="rs-flex rs-gap-6" style={{ fontSize: 12.5, color: "var(--body)" }}>
      <span className="rs-dot" style={{ background: ok ? "var(--st-emerald)" : "var(--rs-muted-soft)" }} />
      {label}
    </span>
  );
}

function GpuDeviceRow({ device }: { device: GpuDevice }) {
  const total = device.total_memory_bytes;
  const used = device.used_memory_bytes ?? total - device.free_memory_bytes;
  const memPct = total > 0 ? (used / total) * 100 : 0;
  const utilPct = device.utilization_percent ?? 0;
  return (
    <div style={{ border: "1px solid var(--hairline)", borderRadius: "var(--radius-md)", padding: 12, background: "var(--canvas-soft)" }}>
      <div className="rs-flex-between" style={{ marginBottom: 8 }}>
        <span style={{ fontSize: 13, fontWeight: 500 }}>{device.name}</span>
        <span className="rs-num" style={{ fontSize: 11, color: "var(--rs-muted)" }}>
          sm_{device.major}{device.minor}
          {device.temperature_c != null ? ` · ${Math.round(device.temperature_c)}°C` : ""}
        </span>
      </div>
      <div className="rs-flex rs-gap-8" style={{ marginBottom: 6 }}>
        <span className="rs-eyebrow" style={{ width: 64 }}>Memory</span>
        <div className="rs-vram rs-grow"><i style={{ width: `${memPct}%`, background: memPct > 85 ? "var(--st-rose)" : memPct > 70 ? "var(--st-amber)" : "var(--st-emerald)" }} /></div>
        <span className="rs-num" style={{ fontSize: 11, width: 96, textAlign: "right" }}>
          {(used / 1024 ** 3).toFixed(1)} / {(total / 1024 ** 3).toFixed(1)} GiB
        </span>
      </div>
      {device.utilization_percent != null && (
        <div className="rs-flex rs-gap-8">
          <span className="rs-eyebrow" style={{ width: 64 }}>Util</span>
          <div className="rs-vram rs-grow"><i style={{ width: `${utilPct}%`, background: "var(--st-blue)" }} /></div>
          <span className="rs-num" style={{ fontSize: 11, width: 96, textAlign: "right" }}>{utilPct.toFixed(0)} %</span>
        </div>
      )}
    </div>
  );
}

// ── remote GPU dispatch (§Ship 23d) ──────────────────────────────────
const EMPTY_REMOTE: RemoteSettings = {
  enabled: false,
  host: "",
  port: 22,
  user: "",
  key_path: "",
  remote_workdir: "~/.sculptor_remote",
  remote_python: "~/.sculptor_remote/venv/bin/python",
  device: "",
  rollout_remote: false,
};

function clampPort(raw: string): number {
  const n = Math.floor(Number(raw));
  if (!Number.isFinite(n) || n < 1) return 22;
  return Math.min(65535, n);
}

function RemoteGpuCard() {
  const qc = useQueryClient();
  const { data, isLoading, error } = useRemoteSettings();
  const [form, setForm] = useState<RemoteSettings | null>(null);
  const [doctor, setDoctor] = useState<RemoteDoctorResponse | null>(null);
  // Seed the form once from the server copy; afterwards local edits win
  // until Save. Inputs are disabled until the server copy arrives so a
  // fast typist can't fork the form off the defaults and clobber saved
  // fields.
  useEffect(() => {
    if (data && form === null) setForm(data);
  }, [data, form]);

  const save = useMutation({
    mutationFn: putRemoteSettings,
    onSuccess: (saved) => {
      qc.setQueryData(["system", "remote"], saved);
      setForm(saved);
      toast.success("Remote settings saved");
    },
    onError: (e: Error) => toast.error("Could not save remote settings", { description: e.message }),
  });
  const test = useMutation({
    // Doctor checks the SAVED settings — persist the form first so the
    // button tests what the user sees, not a stale file. The button is
    // labelled "Save & test" so the save isn't a silent side effect.
    mutationFn: async (f: RemoteSettings) => {
      const saved = await putRemoteSettings(f);
      qc.setQueryData(["system", "remote"], saved);
      setForm(saved);
      return runRemoteDoctor();
    },
    onSuccess: (report) => setDoctor(report),
    onError: (e: Error) => toast.error("Connection test failed to run", { description: e.message }),
  });

  const ready = data !== undefined;
  const busy = save.isPending || test.isPending;
  const f = form ?? data ?? EMPTY_REMOTE;
  const set = (patch: Partial<RemoteSettings>) => {
    setDoctor(null); // a result for edited settings would be misleading
    setForm({ ...f, ...patch });
  };
  const dirty = form !== null && ready && JSON.stringify(form) !== JSON.stringify(data);

  return (
    <div className="rs-card">
      <div className="rs-card-head">
        <div className="rs-card-title"><Icon name="upload" size={16} />Remote GPU (rented pod)</div>
        {/* Persisted truth, not the unsaved form. */}
        <Badge
          status={data?.enabled ? "completed" : "stopped"}
          label={data?.enabled ? "Enabled" : "Off"}
        />
      </div>
      <div className="rs-card-pad rs-vgap-8">
        {isLoading && !data && <p className="rs-sub">Loading remote settings…</p>}
        {error && (
          <div className="rs-banner err">
            <Icon name="alert-triangle" size={17} />
            <span className="rs-grow">Could not load remote settings: {(error as Error).message}</span>
          </div>
        )}
        <p className="rs-sub" style={{ margin: 0 }}>
          Dispatch training to a rented GPU over SSH (e.g. a RunPod 5090) — diagnose,
          edits and the KG stay local. Provision the pod first:{" "}
          <code className="mono">scripts/provision_remote.sh</code> (see{" "}
          <code className="mono">RewardSculptor/docs/remote.md</code>).
        </p>
        <fieldset disabled={!ready || busy} style={{ border: 0, padding: 0, margin: 0, minWidth: 0 }} className="rs-vgap-8">
          <ToggleRow
            on={f.enabled}
            onChange={(v) => set({ enabled: v })}
            title="Dispatch training remotely"
            desc="Applies to new runs and mission stages; overrides any [remote] table in project config.toml. Rollouts stay local unless opted in below."
            label="Dispatch training remotely"
          />
          <div className="rs-flex rs-gap-8 rs-wrap">
            <div style={{ flex: "2 1 220px" }}>
              <Field label="Host" htmlFor="rm-host">
                <input id="rm-host" className="rs-input" placeholder="203.0.113.7" value={f.host}
                  onChange={(e) => set({ host: e.target.value })} />
              </Field>
            </div>
            <div style={{ flex: "0 1 110px" }}>
              <Field label="Port" htmlFor="rm-port">
                <input id="rm-port" className="rs-input" type="number" min={1} max={65535} value={f.port}
                  onChange={(e) => set({ port: clampPort(e.target.value) })} />
              </Field>
            </div>
            <div style={{ flex: "1 1 120px" }}>
              <Field label="User" htmlFor="rm-user">
                <input id="rm-user" className="rs-input" placeholder="root" value={f.user}
                  onChange={(e) => set({ user: e.target.value })} />
              </Field>
            </div>
          </div>
          <div className="rs-flex rs-gap-8 rs-wrap">
            <div style={{ flex: "1 1 220px" }}>
              <Field label="SSH key path" htmlFor="rm-key">
                <input id="rm-key" className="rs-input" placeholder="~/.ssh/id_ed25519" value={f.key_path}
                  onChange={(e) => set({ key_path: e.target.value })} />
              </Field>
            </div>
            <div style={{ flex: "1 1 220px" }}>
              <Field label="Remote python" hint="printed by provision_remote.sh" htmlFor="rm-python">
                <input id="rm-python" className="rs-input" value={f.remote_python}
                  onChange={(e) => set({ remote_python: e.target.value })} />
              </Field>
            </div>
          </div>
          <div className="rs-flex rs-gap-8 rs-wrap">
            <div style={{ flex: "1 1 220px" }}>
              <Field label="Remote workdir" hint="/workspace/… on a network volume" htmlFor="rm-workdir">
                <input id="rm-workdir" className="rs-input" value={f.remote_workdir}
                  onChange={(e) => set({ remote_workdir: e.target.value })} />
              </Field>
            </div>
            <div style={{ flex: "0 1 140px" }}>
              <Field label="Pod device" hint="blank = cuda:0" htmlFor="rm-device">
                <input id="rm-device" className="rs-input" placeholder="cuda:0" value={f.device}
                  onChange={(e) => set({ device: e.target.value })} />
              </Field>
            </div>
          </div>
          <ToggleRow
            on={f.rollout_remote}
            onChange={(v) => set({ rollout_remote: v })}
            title="Also run rollouts remotely"
            desc="Off by default — local rollouts keep video preview working when the pod is flaky."
            label="Also run rollouts remotely"
          />
          <div className="rs-flex rs-gap-8">
            <Btn
              icon="check"
              disabled={busy || !dirty}
              onClick={() => form && save.mutate(form)}
            >
              {save.isPending ? "Saving…" : dirty || !ready ? "Save" : "Saved"}
            </Btn>
            <Btn
              kind="ghost"
              icon={test.isPending ? "loader" : "zap"}
              disabled={busy || !f.host.trim()}
              onClick={() => { setDoctor(null); test.mutate(f); }}
            >
              {test.isPending ? "Testing… (slow hosts can take minutes)" : "Save & test connection"}
            </Btn>
          </div>
        </fieldset>
        <div role="status" aria-live="polite">
          {doctor && (
            <div className="rs-vgap-8" style={{ border: "1px solid var(--hairline)", borderRadius: "var(--radius-md)", padding: 12, background: "var(--canvas-soft)" }}>
              <div className="rs-flex rs-gap-6" style={{ fontSize: 13, fontWeight: 500 }}>
                <Icon
                  name={doctor.ok ? "check-circle" : "alert-triangle"}
                  size={15}
                  color={doctor.ok ? "var(--st-emerald)" : "var(--st-rose)"}
                />
                {doctor.ok ? "All checks passed" : "Some checks failed"}
                <span className="rs-sub" style={{ fontSize: 12 }}>({doctor.host}:{doctor.port})</span>
              </div>
              {doctor.checks.map((c) => (
                <div key={c.name} className="rs-flex rs-gap-6" style={{ fontSize: 12.5 }}>
                  <Icon
                    name={c.ok ? "check-circle" : "alert-triangle"}
                    size={14}
                    color={c.ok ? "var(--st-emerald)" : "var(--st-rose)"}
                  />
                  <span className="sr-only">{c.ok ? "passed:" : "failed:"}</span>
                  <span style={{ fontWeight: 500, whiteSpace: "nowrap" }}>{c.name}</span>
                  <span className="rs-sub mono" style={{ fontSize: 11.5, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={c.detail}>
                    {c.detail}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── shared KG ─────────────────────────────────────────────────────────
function SharedKgCard() {
  const { data, isLoading, error } = useSharedKgStats({ refetchIntervalMs: 30_000 });
  return (
    <div className="rs-card">
      <div className="rs-card-head">
        <div className="rs-card-title"><Icon name="network" size={16} />Knowledge graph (shared)</div>
      </div>
      <div className="rs-card-pad rs-vgap-8">
        {isLoading && <p className="rs-sub">Loading KG stats…</p>}
        {error && <p className="rs-sub" style={{ color: "var(--st-rose)" }}>Could not load KG stats: {(error as Error).message}</p>}
        {data && (
          <>
            {!data.db_exists && (
              <div className="rs-banner warn">
                <Icon name="alert-triangle" size={17} />
                <span className="rs-grow">Empty — no DB yet. Restart the backend to seed the pre-extracted papers.</span>
              </div>
            )}
            <div className="rs-sysgrid">
              <KgStat label="Papers" value={data.papers} />
              <KgStat label="Techniques" value={data.techniques} />
              <KgStat label="Failure modes" value={data.failure_modes} />
              <KgStat label="Reward comps" value={data.reward_components} />
              <KgStat label="Environments" value={data.environments} />
              <KgStat label="Edges" value={data.edges} />
            </div>
            <div className="rs-kv">
              <KvRow k="path" v={data.db_path} mono truncate />
              {data.db_size_bytes > 0 && <KvRow k="size" v={`${(data.db_size_bytes / 1024).toFixed(1)} KB`} mono />}
              {data.last_modified && <KvRow k="last modified" v={new Date(data.last_modified).toLocaleString()} />}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function KgStat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rs-sysitem">
      <span className="lab">{label}</span>
      <span className="val">{value}</span>
    </div>
  );
}

// ── paths ─────────────────────────────────────────────────────────────
function PathsCard({
  projectsRoot, sculptorPath, sculptorOk, sculptorError,
}: { projectsRoot: string; sculptorPath: string | null; sculptorOk: boolean; sculptorError: string | null }) {
  return (
    <div className="rs-card">
      <div className="rs-card-head"><div className="rs-card-title"><Icon name="folder" size={16} />Paths</div></div>
      <div className="rs-card-pad rs-vgap-8">
        <div className="rs-kv">
          <KvRow k="$RS_PROJECTS_ROOT" v={projectsRoot} mono truncate />
          {sculptorPath && <KvRow k="sculptor module" v={sculptorPath} mono truncate />}
        </div>
        <div className="rs-flex rs-gap-6" style={{ fontSize: 13 }}>
          <Icon name={sculptorOk ? "check-circle" : "alert-triangle"} size={15} color={sculptorOk ? "var(--st-emerald)" : "var(--st-amber)"} />
          <span>sculptor: {sculptorOk ? "importable" : sculptorError ?? "error"}</span>
        </div>
      </div>
    </div>
  );
}

// ── appearance ────────────────────────────────────────────────────────
function ThemeCard() {
  const { theme, setTheme } = useTheme();
  return (
    <div className="rs-card">
      <div className="rs-card-head"><div className="rs-card-title"><Icon name="sun" size={16} />Appearance</div></div>
      <div className="rs-card-pad">
        <div className="rs-flex-between">
          <div>
            <div style={{ fontSize: 14, fontWeight: 500 }}>Theme</div>
            <div className="rs-sub" style={{ fontSize: 12.5 }}>Warm cream or warm dark</div>
          </div>
          <Segmented
            value={theme}
            onChange={(v) => setTheme(v as "light" | "dark" | "system")}
            options={[
              { value: "light", label: "Light", icon: "sun" },
              { value: "dark", label: "Dark", icon: "moon" },
              { value: "system", label: "System" },
            ]}
          />
        </div>
      </div>
    </div>
  );
}

// ── reset cache ───────────────────────────────────────────────────────
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
    toast.success("UI state reset", { description: "React Query cache + localStorage cleared." });
    setConfirming(false);
  };
  return (
    <div className="rs-card">
      <div className="rs-card-head"><div className="rs-card-title"><Icon name="refresh-cw" size={16} />Reset UI state</div></div>
      <div className="rs-card-pad">
        <p className="rs-sub" style={{ margin: "0 0 12px" }}>
          Clears the React Query cache + localStorage. Doesn't touch any on-disk project data.
        </p>
        {confirming ? (
          <div className="rs-flex rs-gap-8">
            <Btn kind="danger" icon="refresh-cw" onClick={reset}>Confirm reset</Btn>
            <Btn kind="quiet" onClick={() => setConfirming(false)}>Cancel</Btn>
          </div>
        ) : (
          <Btn kind="ghost" icon="refresh-cw" onClick={() => setConfirming(true)}>Reset</Btn>
        )}
      </div>
    </div>
  );
}
