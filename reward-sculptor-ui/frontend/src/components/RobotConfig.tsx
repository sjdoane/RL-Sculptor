import { useMemo, useRef, useState } from "react";
import { toast } from "sonner";

import { CreateProjectDialog } from "@/components/CreateProjectDialog";
import { Icon } from "@/components/rs/icon";
import { Btn, Modal } from "@/components/rs/primitives";
import { useLibraryRobots } from "@/hooks/useLibrary";
import { useUploadRobotModel } from "@/hooks/useRobot";
import { ApiError, libraryThumbnailUrl } from "@/lib/api";
import {
  DEFAULT_CATEGORIES,
  DEFAULT_TRAINING_SUPPORT,
  ROBOT_CATEGORIES,
  type LibraryRobot,
  type RobotCategory,
  type RobotStateResponse,
  type TrainingSupport,
} from "@/lib/types";

const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;
const ACCEPT_ATTR = ".urdf,.xml,.mjcf,.zip,application/xml,application/zip";

export function RobotConfig({ slug: _slug }: { slug: string }) {
  const [tab, setTab] = useState<"library" | "upload">("library");
  return (
    <div className="rs-card rs-card-pad">
      <div className="rs-card-title" style={{ marginBottom: 4 }}>
        <Icon name="cpu" size={16} />Configure robot
      </div>
      <p className="rs-sub" style={{ margin: "0 0 14px", lineHeight: 1.5 }}>
        Browse the Menagerie-seeded library to spin up a new project, or upload your own URDF / MJCF for the current one.
      </p>
      <div className="rs-mtabs" style={{ marginBottom: 16 }}>
        <button className={tab === "library" ? "on" : ""} onClick={() => setTab("library")}>Library</button>
        <button className={tab === "upload" ? "on" : ""} onClick={() => setTab("upload")}>Upload</button>
      </div>
      {tab === "library" ? <LibraryBrowser /> : <UploadPanel slug={_slug} />}
    </div>
  );
}

// ── Library browser ──────────────────────────────────────────────────
export function LibraryBrowser() {
  const [categories, setCategories] = useState<Set<RobotCategory>>(
    () => new Set(DEFAULT_CATEGORIES),
  );
  const [supports, setSupports] = useState<Set<TrainingSupport>>(
    () => new Set(DEFAULT_TRAINING_SUPPORT),
  );
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<LibraryRobot | null>(null);

  // We fetch the full list once and filter client-side. At ~65 entries
  // the payload is small and the UX benefits from instant local
  // filtering across multiple chips.
  const { data, isLoading, error } = useLibraryRobots();

  const filtered = useMemo(() => {
    const all = data?.robots ?? [];
    const needle = search.trim().toLowerCase();
    return all.filter((r) => {
      if (categories.size > 0 && !categories.has(r.category as RobotCategory))
        return false;
      if (supports.size > 0 && !supports.has(r.training_support as TrainingSupport))
        return false;
      if (needle) {
        const hay = (r.slug + " " + r.display_name + " " + r.description).toLowerCase();
        if (!hay.includes(needle)) return false;
      }
      return true;
    });
  }, [data, categories, supports, search]);

  if (isLoading) {
    return (
      <div className="rs-flex rs-gap-8" style={{ padding: 24, fontSize: 13, color: "var(--rs-muted)" }}>
        <Icon name="loader" size={16} className="rs-spin" /> Loading robot library…
      </div>
    );
  }
  if (error) {
    return (
      <div style={{ borderRadius: "var(--radius-md)", border: "1px solid var(--st-rose-bg)", background: "var(--st-rose-bg)", color: "var(--st-rose-fg)", padding: 16, fontSize: 13 }}>
        Could not load library: {(error as Error).message}
      </div>
    );
  }

  return (
    <div style={{ display: "grid", gridTemplateColumns: "210px 1fr", gap: 18, alignItems: "start" }}>
      {/* Sidebar filters */}
      <div style={{ display: "flex", flexDirection: "column", gap: 16, position: "sticky", top: 8 }}>
        <div>
          <div className="rs-caption">Category</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {ROBOT_CATEGORIES.map((c) => (
              <FilterChip
                key={c}
                active={categories.has(c)}
                onClick={() =>
                  setCategories((prev) => {
                    const next = new Set(prev);
                    if (next.has(c)) next.delete(c);
                    else next.add(c);
                    return next;
                  })
                }
              >
                {c.replace(/_/g, " ")}
              </FilterChip>
            ))}
          </div>
        </div>
        <div>
          <div className="rs-caption">Training support</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {(["mjlab_ready", "gymnasium_compatible", "preview_only"] as TrainingSupport[]).map((t) => (
              <FilterChip
                key={t}
                active={supports.has(t)}
                onClick={() =>
                  setSupports((prev) => {
                    const next = new Set(prev);
                    if (next.has(t)) next.delete(t);
                    else next.add(t);
                    return next;
                  })
                }
              >
                {t === "mjlab_ready" ? "mjlab ready" : t === "gymnasium_compatible" ? "Gymnasium" : "Preview only"}
              </FilterChip>
            ))}
          </div>
        </div>
        <input
          className="rs-input"
          placeholder="Search…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          aria-label="Search robots"
        />
        <div className="rs-sub" style={{ fontSize: 12 }}>{filtered.length} / {data?.total ?? 0} robots</div>
      </div>

      {/* Card grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 12 }}>
        {filtered.map((r) => (
          <RobotCard key={r.slug} robot={r} onOpen={() => setSelected(r)} />
        ))}
        {filtered.length === 0 && (
          <div style={{ gridColumn: "1 / -1", borderRadius: "var(--radius-md)", border: "1px dashed var(--hairline-strong)", padding: 32, textAlign: "center", fontSize: 13, color: "var(--rs-muted)" }}>
            No robots match the current filter. Try adding a category chip or clearing the search.
          </div>
        )}
      </div>

      <RobotDetailModal robot={selected} onClose={() => setSelected(null)} />
    </div>
  );
}

function FilterChip({
  active,
  children,
  onClick,
}: {
  active: boolean;
  children: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button type="button" onClick={onClick} aria-pressed={active} className={"rs-fchip" + (active ? " on" : "")}>
      {children}
    </button>
  );
}

function RobotCard({
  robot,
  onOpen,
}: {
  robot: LibraryRobot;
  onOpen: () => void;
}) {
  return (
    <button type="button" onClick={onOpen} className="rs-robotcard" aria-label={`Open ${robot.display_name}`}>
      <img
        src={libraryThumbnailUrl(robot.slug)}
        alt=""
        loading="lazy"
        className="rs-robotcard-thumb"
        onError={(e) => {
          (e.currentTarget as HTMLImageElement).style.display = "none";
        }}
      />
      <div style={{ display: "flex", flexDirection: "column", gap: 6, padding: 12 }}>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 8 }}>
          <span style={{ fontSize: 13.5, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{robot.display_name}</span>
          <TrainingBadge support={robot.training_support} />
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 10, textTransform: "uppercase", letterSpacing: 0.4, color: "var(--rs-muted)" }}>
          <span>{robot.category.replace(/_/g, " ")}</span>
          {robot.references.length > 0 && (
            <span
              style={{ display: "inline-flex", alignItems: "center", gap: 2 }}
              title={`${robot.references.length} reference paper${robot.references.length === 1 ? "" : "s"} / repo${robot.references.length === 1 ? "" : "s"}`}
            >
              <Icon name="book" size={12} />{robot.references.length}
            </span>
          )}
        </div>
        <p style={{ margin: 0, fontSize: 12, lineHeight: 1.45, color: "var(--rs-muted)", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
          {robot.description}
        </p>
      </div>
    </button>
  );
}

function TrainingBadge({ support }: { support: string }) {
  if (support === "mjlab_ready") {
    return <span className="rs-badge emerald"><Icon name="sparkles" size={11} /> Ready to train</span>;
  }
  if (support === "gymnasium_compatible") {
    return <span className="rs-badge blue">Gymnasium</span>;
  }
  return <span className="rs-badge amber">Preview only</span>;
}

// ── Detail modal ────────────────────────────────────────────────────
function RobotDetailModal({
  robot,
  onClose,
}: {
  robot: LibraryRobot | null;
  onClose: () => void;
}) {
  const [creating, setCreating] = useState(false);

  if (!robot) return null;

  const canTrain =
    robot.training_support === "mjlab_ready" ||
    robot.training_support === "gymnasium_compatible";
  const preview = robot.training_support === "preview_only";

  // M4 §4 / M3 deferral: clicking "Create project" opens the
  // pre-create dialog with device/num_envs pickers (mjlab) or a
  // minimal name-only form (gym / preview).
  if (creating) {
    return (
      <CreateProjectDialog
        robot={robot}
        onClose={() => {
          setCreating(false);
          onClose();
        }}
      />
    );
  }

  const sourceNote =
    robot.source === "menagerie" ? " · MuJoCo Menagerie"
    : robot.source === "gymnasium_builtin" ? " · Gymnasium built-in"
    : robot.source === "mjlab_builtin" ? " · mjlab built-in"
    : "";

  return (
    <Modal
      wide
      icon="bot"
      title={robot.display_name}
      subtitle={<span style={{ display: "inline-flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}><span>{robot.category.replace(/_/g, " ")}{sourceNote}</span><TrainingBadge support={robot.training_support} /></span>}
      onClose={onClose}
      footer={
        <>
          <Btn kind="quiet" onClick={onClose}>Close</Btn>
          <Btn kind="primary" iconRight="arrow-right" disabled={!canTrain && !preview} onClick={() => setCreating(true)}>
            Create project with this robot
          </Btn>
        </>
      }
    >
      <div style={{ borderRadius: "var(--radius-md)", border: "1px solid var(--hairline)", background: "var(--surface-strong)", overflow: "hidden" }}>
        <img
          src={libraryThumbnailUrl(robot.slug)}
          alt=""
          style={{ aspectRatio: "4 / 3", width: "100%", objectFit: "contain", display: "block" }}
          onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = "none"; }}
        />
      </div>

      <p className="rs-sub" style={{ fontSize: 13.5, lineHeight: 1.6, margin: 0 }}>
        {robot.description || "(No description available.)"}
      </p>

      {preview && (
        <div style={{ display: "flex", alignItems: "flex-start", gap: 8, borderRadius: "var(--radius-md)", border: "1px solid color-mix(in srgb, var(--st-amber) 40%, transparent)", background: "var(--st-amber-bg)", color: "var(--st-amber-fg)", padding: 12, fontSize: 13 }}>
          <span style={{ marginTop: 1, flexShrink: 0 }}><Icon name="alert-circle" size={16} /></span>
          <div>
            <div style={{ fontWeight: 600 }}>Preview only — training not yet wired up</div>
            <p style={{ margin: "2px 0 0", fontSize: 12, color: "var(--rs-muted)" }}>
              No mjlab task is registered for this robot in the running mjlab install. You can still create the project to render the robot in the preview panel; when a task is contributed upstream, the training button will light up automatically.
            </p>
          </div>
        </div>
      )}

      {robot.preconfigured_tasks.length > 0 && (
        <div>
          <div className="rs-caption" style={{ display: "flex", alignItems: "center", gap: 5 }}><Icon name="layers" size={12} /> Pre-configured tasks</div>
          <ul style={{ display: "flex", flexDirection: "column", gap: 6, margin: 0, padding: 0, listStyle: "none" }}>
            {robot.preconfigured_tasks.map((t) => (
              <li key={t.task_id} style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 8, borderRadius: "var(--radius-md)", border: "1px solid var(--hairline)", background: "var(--surface-strong)", padding: 9, fontSize: 13 }}>
                <div>
                  <div style={{ fontWeight: 500 }}>{t.display_name}</div>
                  <div className="mono" style={{ fontSize: 10, color: "var(--rs-muted)" }}>{t.task_id}</div>
                </div>
                <span style={{ flexShrink: 0, fontSize: 12, color: "var(--rs-muted)" }}>{t.recommended_num_envs} envs</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {robot.references.length > 0 && (
        <div>
          <div className="rs-caption" style={{ display: "flex", alignItems: "center", gap: 5 }}><Icon name="book" size={12} /> References</div>
          <ul style={{ display: "flex", flexDirection: "column", gap: 5, margin: 0, padding: 0, listStyle: "none" }}>
            {robot.references.map((ref) => (
              <li key={ref.url} style={{ display: "flex", alignItems: "flex-start", gap: 8, fontSize: 13 }}>
                <span style={{ marginTop: 2, flexShrink: 0, color: "var(--rs-muted)" }}><Icon name={ref.kind === "repo" ? "git-branch" : "book"} size={13} /></span>
                <a href={ref.url} target="_blank" rel="noopener noreferrer" style={{ display: "inline-flex", alignItems: "center", gap: 4, color: "var(--ink)" }}>
                  {ref.citation || ref.url}
                  <Icon name="external" size={12} />
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}

      {robot.demote_note && (
        <div style={{ borderRadius: "var(--radius-md)", border: "1px solid color-mix(in srgb, var(--st-amber) 40%, transparent)", background: "var(--st-amber-bg)", padding: 9, fontSize: 12, color: "var(--rs-muted)" }}>
          {robot.demote_note}
        </div>
      )}
    </Modal>
  );
}

// ── Upload tab (unchanged from M0; preserved for backward compat) ────
function UploadPanel({ slug }: { slug: string }) {
  const upload = useUploadRobotModel(slug);
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const [lastSuccess, setLastSuccess] = useState<RobotStateResponse | null>(null);

  const submit = (file: File) => {
    setLocalError(null);
    const ext = file.name.toLowerCase().slice(file.name.lastIndexOf("."));
    if (![".urdf", ".xml", ".mjcf", ".zip"].includes(ext)) {
      setLocalError(
        `Extension ${ext || "(none)"} not supported — expected .urdf, .xml, .mjcf, or .zip.`,
      );
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setLocalError(`File is ${(file.size / (1024 * 1024)).toFixed(1)} MB — limit is 50 MB.`);
      return;
    }
    const isZip = ext === ".zip";
    upload.mutate(
      {
        modelFile: isZip ? new File([], "model.xml") : file,
        meshesZip: isZip ? file : undefined,
      },
      {
        onSuccess: (robot) => {
          setLastSuccess(robot);
          toast.success(`Uploaded ${robot.original_filename ?? "model"}`);
        },
        onError: (err) => {
          const msg =
            err instanceof ApiError
              ? err.problem.detail ?? err.problem.title
              : (err as Error).message;
          setLocalError(msg);
        },
      },
    );
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          const f = e.dataTransfer.files?.[0];
          if (f) submit(f);
        }}
        className={"rs-drop" + (dragOver ? " over" : "")}
        style={upload.isPending ? { opacity: 0.6 } : undefined}
      >
        <Icon name="upload" size={30} color="var(--rs-muted)" />
        <div style={{ fontSize: 13.5 }}><strong>Drop</strong> a .urdf / .xml / .mjcf / .zip file here</div>
        <div style={{ fontSize: 12, color: "var(--rs-muted)" }}>
          or click to browse. 50 MB combined cap. Mesh zips are extracted into <code className="mono">uploads/robot/</code>.
        </div>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT_ATTR}
          style={{ display: "none" }}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) submit(f);
            e.currentTarget.value = "";
          }}
        />
        <Btn kind="ghost" size="sm" icon="upload" onClick={() => inputRef.current?.click()} disabled={upload.isPending} style={{ marginTop: 8 }}>
          Browse
        </Btn>
      </div>

      {upload.isPending && (
        <div className="rs-flex rs-gap-8" style={{ borderRadius: "var(--radius-md)", border: "1px solid var(--hairline)", background: "var(--surface-strong)", padding: 12, fontSize: 13 }}>
          <Icon name="loader" size={16} className="rs-spin" />
          <span>Uploading + validating via MuJoCo…</span>
        </div>
      )}

      {localError && (
        <div style={{ display: "flex", alignItems: "flex-start", gap: 8, borderRadius: "var(--radius-md)", border: "1px solid var(--st-rose-bg)", background: "var(--st-rose-bg)", color: "var(--st-rose-fg)", padding: 12, fontSize: 13 }}>
          <span style={{ marginTop: 1, flexShrink: 0 }}><Icon name="alert-circle" size={16} /></span>
          <div style={{ minWidth: 0 }}>
            <div style={{ fontWeight: 600 }}>Could not upload</div>
            <p className="mono" style={{ margin: "2px 0 0", fontSize: 12, color: "var(--rs-muted)", wordBreak: "break-word" }}>{localError}</p>
          </div>
        </div>
      )}

      {lastSuccess && !upload.isPending && !localError && (
        <div className="rs-flex rs-gap-8" style={{ borderRadius: "var(--radius-md)", border: "1px solid color-mix(in srgb, var(--st-emerald) 35%, transparent)", background: "var(--st-emerald-bg)", color: "var(--st-emerald-fg)", padding: 12, fontSize: 13 }}>
          <Icon name="check" size={16} />
          <span>
            Uploaded <b>{lastSuccess.original_filename}</b> as <span className="mono" style={{ fontSize: 12 }}>{lastSuccess.kind}</span>
            {lastSuccess.mesh_paths.length > 0 && <> · {lastSuccess.mesh_paths.length} mesh{lastSuccess.mesh_paths.length === 1 ? "" : "es"}</>}
          </span>
        </div>
      )}
    </div>
  );
}
