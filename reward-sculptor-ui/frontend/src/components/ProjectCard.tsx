import { Link } from "react-router-dom";
import { AlertTriangle, ImageOff, MoreVertical, Trash2 } from "lucide-react";
import { useState } from "react";

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { StatusBadge } from "@/components/StatusBadge";
import { useProjectPreview } from "@/hooks/useProjectPreview";
import { useDeleteProject } from "@/hooks/useProjects";
import { formatRelative } from "@/lib/utils";
import type { ProjectSummary } from "@/lib/types";

export function ProjectCard({ project }: { project: ProjectSummary }) {
  const [confirmOpen, setConfirmOpen] = useState(false);
  return (
    <Card className="flex flex-col overflow-hidden transition-colors hover:border-foreground/20">
      <Link to={`/projects/${project.slug}`} className="block">
        <PreviewPane slug={project.slug} />
      </Link>

      <CardHeader className="flex-row items-start justify-between gap-2 space-y-0">
        <div className="min-w-0 flex-1">
          <Link to={`/projects/${project.slug}`} className="block min-w-0">
            <CardTitle className="truncate">{project.display_name}</CardTitle>
            <CardDescription className="mt-1 truncate font-mono text-[11px]">
              {project.slug}
            </CardDescription>
          </Link>
        </div>
        <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
          <DialogTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="-mr-2 h-7 w-7 shrink-0"
              aria-label="Project menu"
              onClick={(e) => e.stopPropagation()}
            >
              <MoreVertical className="h-4 w-4" />
            </Button>
          </DialogTrigger>
          <DeleteConfirm
            project={project}
            onDone={() => setConfirmOpen(false)}
          />
        </Dialog>
      </CardHeader>

      <CardContent className="flex flex-col gap-2 text-xs">
        {project.migration_warning && (
          <div
            className="flex items-start gap-1.5 rounded-md border border-amber-500/40 bg-amber-500/5 p-2 text-[11px]"
            title={project.migration_warning}
          >
            <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0 text-amber-600" />
            <span className="line-clamp-2 text-amber-700 dark:text-amber-300">
              Uses a deprecated adapter. Fork from the library to upgrade —
              the original will not be modified.
            </span>
          </div>
        )}
        {project.description ? (
          <p className="line-clamp-2 text-muted-foreground">
            {project.description}
          </p>
        ) : (
          <p className="italic text-muted-foreground/70">no description</p>
        )}
        <div className="flex items-center gap-1 font-mono text-[10px] text-muted-foreground">
          <span>env:</span>
          <span className="text-foreground">
            {project.env_id && project.env_id !== "CHANGE_ME"
              ? project.env_id
              : "—"}
          </span>
        </div>
      </CardContent>

      <CardFooter className="justify-between gap-2 border-t pt-3">
        <div className="flex items-center gap-2">
          <StatusBadge status={project.status} />
          <span className="text-[10px] text-muted-foreground">
            {project.n_iterations_completed} iter
          </span>
        </div>
        <span className="text-[10px] text-muted-foreground">
          {formatRelative(project.created_at)}
        </span>
      </CardFooter>
    </Card>
  );
}

function PreviewPane({ slug }: { slug: string }) {
  const { data: previewUrl, error } = useProjectPreview(slug);
  return (
    <div className="flex aspect-video w-full items-center justify-center border-b bg-muted/40">
      {previewUrl ? (
        <img
          src={previewUrl}
          alt=""
          className="h-full w-full object-cover"
          loading="lazy"
        />
      ) : (
        <div className="flex flex-col items-center gap-1 text-muted-foreground">
          <ImageOff className="h-6 w-6 opacity-60" />
          <span className="text-[10px]">
            {error?.status === 409
              ? "no robot yet"
              : error
              ? "preview unavailable"
              : "…"}
          </span>
        </div>
      )}
    </div>
  );
}

function DeleteConfirm({
  project,
  onDone,
}: {
  project: ProjectSummary;
  onDone: () => void;
}) {
  const del = useDeleteProject();
  return (
    <DialogContent onClick={(e) => e.stopPropagation()}>
      <DialogHeader>
        <DialogTitle>Delete “{project.display_name}”?</DialogTitle>
        <DialogDescription>
          This removes the project directory and all its runs. The sculptor
          source is untouched. This action cannot be undone.
        </DialogDescription>
      </DialogHeader>
      <DialogFooter>
        <Button variant="outline" onClick={onDone} disabled={del.isPending}>
          Cancel
        </Button>
        <Button
          variant="destructive"
          onClick={() => del.mutate(project.slug, { onSuccess: onDone })}
          disabled={del.isPending}
        >
          {del.isPending ? (
            "Deleting…"
          ) : (
            <>
              <Trash2 className="h-4 w-4" />
              Delete
            </>
          )}
        </Button>
      </DialogFooter>
    </DialogContent>
  );
}
