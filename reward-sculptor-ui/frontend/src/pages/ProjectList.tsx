import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { Icon } from "@/components/rs/icon";
import { Badge, Btn, IconBtn, Modal, Sparkline } from "@/components/rs/primitives";
import { useRunningSlugs } from "@/hooks/useDashboard";
import { useDeleteProject, useProjects } from "@/hooks/useProjects";
import { projectBadgeStatus } from "@/lib/projectStatus";
import { formatRelative } from "@/lib/utils";
import type { ProjectSummary } from "@/lib/types";

function humanizeSlug(s: string | null | undefined): string {
  if (!s) return "—";
  return s
    .split(/[_-]/)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

function adapterShort(cls: string | null | undefined): string {
  if (!cls) return "—";
  if (cls.includes("mjlab")) return "mjlab";
  if (cls.toLowerCase().includes("gym")) return "gym_sb3";
  return cls.split(".").pop() ?? cls;
}

export default function ProjectList() {
  const nav = useNavigate();
  const { data, isLoading, error } = useProjects();
  const del = useDeleteProject();
  const [confirm, setConfirm] = useState<ProjectSummary | null>(null);
  const projects = data ?? [];
  const runningSlugs = useRunningSlugs();

  return (
    <div className="rs-scroll">
      <div className="rs-pad rs-vgap-24">
        <div className="rs-flex-between rs-wrap rs-gap-12">
          <div>
            <div className="rs-eyebrow">{projects.length} projects</div>
            <h2 className="rs-h2" style={{ marginTop: 6 }}>Projects</h2>
          </div>
          <Btn kind="primary" icon="plus" onClick={() => nav("/library")}>
            New project
          </Btn>
        </div>

        {isLoading ? (
          <div className="rs-sub">Loading projects…</div>
        ) : error ? (
          <div className="rs-banner err">
            <Icon name="alert-triangle" size={17} />
            <span className="rs-grow">Failed to load projects: {(error as Error).message}</span>
          </div>
        ) : projects.length === 0 ? (
          <div className="rs-card">
            <div className="rs-empty">
              <span className="ic"><Icon name="folder" size={22} color="var(--rs-muted)" /></span>
              <div className="rs-h3" style={{ fontWeight: 500, color: "var(--ink)" }}>No projects yet</div>
              <div className="rs-sub" style={{ maxWidth: 360 }}>
                Pick a robot from the library to scaffold your first project.
              </div>
              <Btn kind="primary" icon="plus" onClick={() => nav("/library")}>New project</Btn>
            </div>
          </div>
        ) : (
          <div className="rs-card">
            <div style={{ overflowX: "auto" }}>
            <table className="rs-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Status</th>
                  <th>Adapter</th>
                  <th>Robot</th>
                  <th>Best</th>
                  <th>Trend</th>
                  <th>Updated</th>
                  <th aria-label="Actions" />
                </tr>
              </thead>
              <tbody>
                {projects.map((p) => (
                  <tr
                    key={p.slug}
                    onClick={() => nav(`/projects/${p.slug}`)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        nav(`/projects/${p.slug}`);
                      }
                    }}
                    tabIndex={0}
                    role="button"
                    aria-label={`Open project ${p.display_name}`}
                    style={{ cursor: "pointer" }}
                  >
                    <td className="name" style={{ fontFamily: "var(--font-sans)" }}>{p.display_name}</td>
                    {/* §Ship 37: `p.status` is "draft" for every project on
                        this install — see `useRunningSlugs`. */}
                    <td>
                      <Badge status={projectBadgeStatus(
                        p.status, runningSlugs.has(p.slug),
                      )} />
                    </td>
                    <td>{adapterShort(p.adapter_class)}</td>
                    <td style={{ fontFamily: "var(--font-sans)" }}>{humanizeSlug(p.library_slug) || p.env_id || "—"}</td>
                    <td style={{ color: p.primary_metric == null ? "var(--rs-muted)" : "var(--st-amber)" }}>
                      {p.primary_metric == null ? "—" : p.primary_metric.toFixed(1)}
                    </td>
                    <td>
                      <Sparkline
                        data={p.primary_metric_history ?? []}
                        w={64}
                        h={20}
                        color={p.status === "errored" ? "var(--st-rose)" : "var(--st-emerald)"}
                      />
                    </td>
                    <td style={{ color: "var(--rs-muted)" }}>{formatRelative(p.created_at)}</td>
                    <td style={{ textAlign: "right" }}>
                      <IconBtn
                        icon="trash"
                        label={`Delete ${p.display_name}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          setConfirm(p);
                        }}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          </div>
        )}
      </div>

      {confirm && (
        <Modal
          title="Delete project"
          icon="trash"
          onClose={() => { if (!del.isPending) setConfirm(null); }}
          footer={
            <>
              <Btn kind="quiet" onClick={() => setConfirm(null)} disabled={del.isPending}>Cancel</Btn>
              <Btn
                kind="danger"
                icon="trash"
                disabled={del.isPending}
                onClick={() => del.mutate(confirm.slug, { onSuccess: () => setConfirm(null) })}
              >
                {del.isPending ? "Deleting…" : "Delete project"}
              </Btn>
            </>
          }
        >
          <p className="rs-sub" style={{ margin: 0 }}>
            Permanently delete <b style={{ color: "var(--ink)" }}>{confirm.display_name}</b> and all its runs,
            rewards, and knowledge graph? This cannot be undone.
          </p>
          {del.error && (
            <div className="rs-banner err">
              <Icon name="alert-triangle" size={17} />
              <span className="rs-grow">{(del.error as Error).message}</span>
            </div>
          )}
        </Modal>
      )}
    </div>
  );
}
