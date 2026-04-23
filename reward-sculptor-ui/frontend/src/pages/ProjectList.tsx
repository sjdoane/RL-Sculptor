import { Link } from "react-router-dom";
import { Plus, FolderOpen, RefreshCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ProjectCard } from "@/components/ProjectCard";
import { useProjects } from "@/hooks/useProjects";

export default function ProjectList() {
  const { data: projects, isLoading, error, refetch, isFetching } = useProjects();

  return (
    <div className="flex h-full flex-col">
      <Header onRefresh={() => refetch()} refreshing={isFetching} />

      <div className="flex-1 overflow-y-auto scrollbar-thin px-6 pb-10 pt-4">
        {isLoading ? (
          <LoadingState />
        ) : error ? (
          <ErrorState error={error as Error} />
        ) : !projects || projects.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {projects.map((p) => (
              <ProjectCard key={p.slug} project={p} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function Header({
  onRefresh,
  refreshing,
}: {
  onRefresh: () => void;
  refreshing: boolean;
}) {
  return (
    <header className="sticky top-0 z-10 flex h-12 items-center justify-between border-b bg-background/80 px-6 backdrop-blur">
      <div className="flex items-baseline gap-2">
        <h1 className="text-sm font-semibold">Projects</h1>
        <span className="text-xs text-muted-foreground">
          Every project is a self-contained sculpt workspace.
        </span>
      </div>
      <div className="flex items-center gap-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={onRefresh}
          disabled={refreshing}
          aria-label="Refresh project list"
        >
          <RefreshCcw
            className={refreshing ? "animate-spin" : ""}
          />
        </Button>
        <Button asChild size="sm">
          <Link to="/library">
            <Plus />
            New from library
          </Link>
        </Button>
      </div>
    </header>
  );
}

function LoadingState() {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {Array.from({ length: 4 }).map((_, i) => (
        <div
          key={i}
          className="h-64 animate-pulse rounded-lg border bg-muted/30"
        />
      ))}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="mx-auto mt-24 max-w-md rounded-lg border border-dashed bg-muted/20 p-10 text-center">
      <FolderOpen className="mx-auto h-10 w-10 text-muted-foreground" />
      <h2 className="mt-4 text-base font-semibold">No projects yet</h2>
      <p className="mt-1 text-sm text-muted-foreground">
        Create a project to scaffold a sculpt workspace with its own
        rewards, runs, and knowledge graph.
      </p>
      <Button asChild className="mt-6">
        <Link to="/library">
          <Plus />
          New from library
        </Link>
      </Button>
    </div>
  );
}

function ErrorState({ error }: { error: Error }) {
  return (
    <div className="mx-auto mt-24 max-w-md rounded-lg border border-destructive/40 bg-destructive/5 p-6">
      <h2 className="text-sm font-semibold text-destructive">
        Could not load projects
      </h2>
      <p className="mt-1 font-mono text-xs text-muted-foreground">
        {error.message}
      </p>
      <p className="mt-3 text-xs text-muted-foreground">
        Check the backend is running at <code>localhost:8000</code>.
      </p>
    </div>
  );
}
