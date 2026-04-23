import { useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  BookOpen,
  Check,
  Cpu,
  ExternalLink,
  FileUp,
  Github,
  Layers,
  Loader2,
  Sparkles,
  Upload,
} from "lucide-react";
import { toast } from "sonner";

import { CreateProjectDialog } from "@/components/CreateProjectDialog";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useLibraryRobots } from "@/hooks/useLibrary";
import { useUploadRobotModel } from "@/hooks/useRobot";
import { ApiError, libraryThumbnailUrl } from "@/lib/api";
import { cn } from "@/lib/utils";
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
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Cpu className="h-4 w-4" />
          Configure robot
        </CardTitle>
        <CardDescription>
          Browse the Menagerie-seeded library to spin up a new project, or
          upload your own URDF / MJCF for the current one.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="library">
          <TabsList>
            <TabsTrigger value="library">Library</TabsTrigger>
            <TabsTrigger value="upload">Upload</TabsTrigger>
          </TabsList>
          <TabsContent value="library">
            <LibraryBrowser />
          </TabsContent>
          <TabsContent value="upload">
            <UploadPanel slug={_slug} />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
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
      <div className="flex items-center gap-2 p-6 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        Loading robot library…
      </div>
    );
  }
  if (error) {
    return (
      <div className="rounded-md border border-destructive/40 bg-destructive/5 p-4 text-sm text-destructive">
        Could not load library: {(error as Error).message}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-[220px_1fr]">
      {/* Sidebar filters */}
      <div className="flex flex-col gap-4 md:sticky md:top-2 md:self-start">
        <div>
          <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Category
          </div>
          <div className="flex flex-wrap gap-1.5 md:flex-col md:items-start">
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
          <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Training support
          </div>
          <div className="flex flex-wrap gap-1.5 md:flex-col md:items-start">
            {(["mjlab_ready", "gymnasium_compatible", "preview_only"] as TrainingSupport[]).map(
              (t) => (
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
              ),
            )}
          </div>
        </div>
        <div>
          <Input
            placeholder="Search…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            aria-label="Search robots"
          />
        </div>
        <div className="text-xs text-muted-foreground">
          {filtered.length} / {data?.total ?? 0} robots
        </div>
      </div>

      {/* Card grid */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {filtered.map((r) => (
          <RobotCard
            key={r.slug}
            robot={r}
            onOpen={() => setSelected(r)}
          />
        ))}
        {filtered.length === 0 && (
          <div className="col-span-full rounded-md border border-dashed p-8 text-center text-sm text-muted-foreground">
            No robots match the current filter. Try adding a category chip
            or clearing the search.
          </div>
        )}
      </div>

      <RobotDetailModal
        robot={selected}
        onClose={() => setSelected(null)}
      />
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
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium transition-colors",
        active
          ? "border-foreground bg-foreground text-background"
          : "border-border bg-background hover:bg-accent",
      )}
    >
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
    <button
      type="button"
      onClick={onOpen}
      className={cn(
        "group relative flex w-full flex-col overflow-hidden rounded-lg border bg-card text-left transition-all",
        "hover:-translate-y-0.5 hover:border-foreground/30 hover:shadow-md",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
      )}
      aria-label={`Open ${robot.display_name}`}
    >
      <div className="relative aspect-[4/3] w-full bg-muted/40">
        <img
          src={libraryThumbnailUrl(robot.slug)}
          alt=""
          loading="lazy"
          className="h-full w-full object-cover"
          onError={(e) => {
            const el = e.currentTarget as HTMLImageElement;
            el.style.display = "none";
          }}
        />
      </div>
      <div className="flex flex-col gap-1.5 p-3">
        <div className="flex items-start justify-between gap-2">
          <span className="text-sm font-semibold line-clamp-1">
            {robot.display_name}
          </span>
          <TrainingBadge support={robot.training_support} />
        </div>
        <div className="flex items-center gap-2 text-[10px] uppercase tracking-wide text-muted-foreground">
          <span>{robot.category.replace(/_/g, " ")}</span>
          {robot.references.length > 0 && (
            <span className="inline-flex items-center gap-0.5">
              <BookOpen className="h-3 w-3" />
              {robot.references.length}
            </span>
          )}
        </div>
        <p className="line-clamp-2 text-xs text-muted-foreground">
          {robot.description}
        </p>
      </div>
    </button>
  );
}

function TrainingBadge({ support }: { support: string }) {
  if (support === "mjlab_ready") {
    return (
      <Badge
        variant="outline"
        className="border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
      >
        <Sparkles className="mr-1 h-3 w-3" /> Ready to train
      </Badge>
    );
  }
  if (support === "gymnasium_compatible") {
    return (
      <Badge
        variant="outline"
        className="border-sky-500/40 bg-sky-500/10 text-sky-700 dark:text-sky-300"
      >
        Gymnasium
      </Badge>
    );
  }
  return (
    <Badge
      variant="outline"
      className="border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300"
    >
      Preview only
    </Badge>
  );
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

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-h-[90vh] max-w-2xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {robot.display_name}
            <TrainingBadge support={robot.training_support} />
          </DialogTitle>
          <DialogDescription>
            {robot.category.replace(/_/g, " ")}
            {robot.source === "menagerie" && " · MuJoCo Menagerie"}
            {robot.source === "gymnasium_builtin" && " · Gymnasium built-in"}
            {robot.source === "mjlab_builtin" && " · mjlab built-in"}
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-4">
          <div className="rounded-md border bg-muted/30">
            <img
              src={libraryThumbnailUrl(robot.slug)}
              alt=""
              className="aspect-[4/3] w-full rounded-md object-contain"
              onError={(e) => {
                (e.currentTarget as HTMLImageElement).style.display = "none";
              }}
            />
          </div>

          <p className="text-sm text-muted-foreground">
            {robot.description || "(No description available.)"}
          </p>

          {preview && (
            <div className="flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/5 p-3 text-sm">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
              <div>
                <div className="font-medium text-amber-700 dark:text-amber-300">
                  Preview only — training not yet wired up
                </div>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  No mjlab task is registered for this robot in the running
                  mjlab install. You can still create the project to
                  render the robot in the preview panel; when a task is
                  contributed upstream, the training button will light up
                  automatically.
                </p>
              </div>
            </div>
          )}

          {robot.preconfigured_tasks.length > 0 && (
            <div>
              <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                <Layers className="mr-1 inline h-3 w-3" /> Pre-configured tasks
              </div>
              <ul className="flex flex-col gap-1 text-sm">
                {robot.preconfigured_tasks.map((t) => (
                  <li
                    key={t.task_id}
                    className="flex items-start justify-between gap-2 rounded-md border bg-muted/20 p-2"
                  >
                    <div>
                      <div className="font-medium">{t.display_name}</div>
                      <div className="font-mono text-[10px] text-muted-foreground">
                        {t.task_id}
                      </div>
                    </div>
                    <span className="shrink-0 text-xs text-muted-foreground">
                      {t.recommended_num_envs} envs
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {robot.references.length > 0 && (
            <div>
              <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                <BookOpen className="mr-1 inline h-3 w-3" /> References
              </div>
              <ul className="flex flex-col gap-1 text-sm">
                {robot.references.map((ref) => (
                  <li key={ref.url} className="flex items-start gap-2">
                    {ref.kind === "repo" ? (
                      <Github className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                    ) : (
                      <BookOpen className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                    )}
                    <a
                      href={ref.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-sm text-foreground hover:underline"
                    >
                      {ref.citation || ref.url}
                      <ExternalLink className="h-3 w-3" />
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {robot.demote_note && (
            <div className="rounded-md border border-amber-500/40 bg-amber-500/5 p-2 text-xs text-muted-foreground">
              {robot.demote_note}
            </div>
          )}
        </div>

        <DialogFooter>
          <Button type="button" variant="outline" onClick={onClose}>
            Close
          </Button>
          <Button
            type="button"
            disabled={!canTrain && !preview}
            onClick={() => setCreating(true)}
          >
            Create project with this robot
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
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
    <div className="flex flex-col gap-3">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          const f = e.dataTransfer.files?.[0];
          if (f) submit(f);
        }}
        className={cn(
          "flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-10 text-center transition-colors",
          dragOver
            ? "border-foreground/60 bg-accent/50"
            : "border-border bg-muted/20",
          upload.isPending && "opacity-60",
        )}
      >
        <FileUp className="h-8 w-8 text-muted-foreground" />
        <div className="text-sm">
          <span className="font-medium">Drop</span> a .urdf / .xml / .mjcf / .zip
          file here
        </div>
        <div className="text-xs text-muted-foreground">
          or click to browse. 50 MB combined cap. Mesh zips are extracted into
          <code className="mx-1">uploads/robot/</code>.
        </div>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT_ATTR}
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) submit(f);
            e.currentTarget.value = "";
          }}
        />
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="mt-2"
          onClick={() => inputRef.current?.click()}
          disabled={upload.isPending}
        >
          <Upload className="h-4 w-4" />
          Browse
        </Button>
      </div>

      {upload.isPending && (
        <div className="flex items-center gap-2 rounded-md border bg-muted/30 p-3 text-sm">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span>Uploading + validating via MuJoCo…</span>
        </div>
      )}

      {localError && (
        <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/5 p-3 text-sm">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />
          <div className="min-w-0">
            <div className="font-medium text-destructive">Could not upload</div>
            <p className="font-mono text-xs text-muted-foreground break-words">
              {localError}
            </p>
          </div>
        </div>
      )}

      {lastSuccess && !upload.isPending && !localError && (
        <div className="flex items-center gap-2 rounded-md border border-emerald-300/50 bg-emerald-50 p-3 text-sm text-emerald-800">
          <Check className="h-4 w-4" />
          <span>
            Uploaded <b>{lastSuccess.original_filename}</b> as{" "}
            <span className="font-mono text-xs">{lastSuccess.kind}</span>
            {lastSuccess.mesh_paths.length > 0 && (
              <>
                {" "}
                · {lastSuccess.mesh_paths.length} mesh
                {lastSuccess.mesh_paths.length === 1 ? "" : "es"}
              </>
            )}
          </span>
        </div>
      )}
    </div>
  );
}
