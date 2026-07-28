/**
 * The pipeline as one ordered list, on one screen.
 *
 * The showcase workflow — author a world, compose a novel motion out of
 * solved clips, split it into OGMP modes, author a reward per mode, train —
 * was spread across six tabs and eight modals, three levels deep, with no
 * state carried between them. Nothing said what order to do things in, what
 * was already done, or which artifact the next step would read. The composed
 * clip id in particular existed only inside whichever dialog had just made
 * it, so it had to be re-found by hand in a ~6015-clip library twice.
 *
 * This card is the spine. Every row states what it needs, shows what is
 * actually on disk for it, and opens the exact place that does it. It reads
 * authoritative artifacts for "done" — the world selection file, the reward
 * chain, the mode-reward files, the iteration count — and the project's
 * behavior draft only for "what are we building".
 */
import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

import { Icon } from "@/components/rs/icon";
import { Btn } from "@/components/rs/primitives";
import { useBehaviorDraft } from "@/hooks/useBehaviorDraft";
import { usePolicies } from "@/hooks/usePolicies";
import { useRewards } from "@/hooks/useRewards";
import { useWorldSelection } from "@/hooks/useWorlds";
import { listModeRewards } from "@/lib/api";
import type { ProjectDetail } from "@/lib/types";

export type FlowTab =
  | "overview" | "world" | "rewards" | "physics"
  | "knowledge" | "training" | "results";

type Step = {
  key: string;
  label: string;
  /** What this step is for, in one line. */
  hint: string;
  done: boolean;
  /** What is actually on disk for this step, when there is something. */
  evidence?: string;
  tab: FlowTab;
  action: string;
  /** True when this step is genuinely optional for a working run. */
  optional?: boolean;
};

export function BehaviorFlow({
  slug, project, robotConfigured, onGoTo,
}: {
  slug: string;
  project: ProjectDetail;
  robotConfigured: boolean;
  onGoTo: (tab: FlowTab) => void;
}) {
  const world = useWorldSelection(slug);
  const rewards = useRewards(slug);
  const policies = usePolicies(slug);
  const draft = useBehaviorDraft(slug);
  const modeRewards = useQuery({
    queryKey: ["modeRewards", slug],
    queryFn: () => listModeRewards(slug),
    retry: false,
    staleTime: 10_000,
  });

  const steps: Step[] = useMemo(() => {
    const hasIters = project.n_iterations_completed > 0;
    const clipId = draft.data?.reference_clip_id ?? "";
    const modeFile = modeRewards.data?.mode_rewards.find(
      (f) => !clipId || f.clip_id === clipId,
    );
    const authoredCount =
      modeFile?.modes.filter((m) => m.authored).length ?? 0;
    const modeCount = modeFile?.modes.length ?? 0;
    // Only the version chain proves a reward is what a run will train. A
    // mode_reward_v*.py is NOT a version — that is the whole reason `promote`
    // exists — so "all modes authored" is not the same as "in use".
    const versionCount = rewards.data?.length ?? 0;
    const rewardShaped = versionCount > 1 || hasIters;
    // When the newest version IS a per-mode module, say so in its own terms.
    // "2 versions" is true and useless; what the user needs to know is whether
    // the four windows they authored are the ones a run will pay.
    const promoted = modeRewards.data?.promoted ?? null;
    const rewardEvidence = promoted
      ? `v${promoted.version}.py · ${promoted.modes.length} modes`
        + (promoted.unauthored.length
            ? ` · ${promoted.unauthored.length} still a stub`
            : " · all authored")
      : versionCount
        ? `${versionCount} version${versionCount === 1 ? "" : "s"}`
        : undefined;

    return [
      {
        key: "robot",
        label: "Configure the robot",
        hint: "Pick a library robot or upload a URDF/MJCF.",
        done: robotConfigured,
        evidence: project.library_slug || undefined,
        tab: "overview",
        action: "Configure",
      },
      {
        key: "world",
        label: "Author the world",
        hint: "Describe the terrain and task in prose; the author compiles a scene, "
            + "a goal predicate, and train-only variations.",
        done: !!world.data?.selection,
        evidence: world.data?.selection
          ? `selection v${world.data.selection.selection_version}`
          : undefined,
        tab: "world",
        action: world.data?.selection ? "Open world" : "Author world",
      },
      {
        key: "motion",
        label: "Choose or compose a reference motion",
        hint: "A motion no single clip contains gets composed out of spans of "
            + "several solved ones. It becomes the immutable tracking base.",
        done: !!clipId,
        evidence: clipId || undefined,
        tab: "training",
        action: clipId ? "Change motion" : "Find motion",
        optional: true,
      },
      {
        key: "modes",
        label: "Author a reward per mode",
        hint: "A composite's phases ARE its OGMP modes. Each gets its own reward "
            + "terms, paid only inside its own window.",
        done: modeCount > 0 && authoredCount === modeCount,
        evidence: modeCount
          ? `${authoredCount}/${modeCount} authored · ${modeFile?.filename}`
            // The gap this closes: authoring writes mode_reward_v<n>.py, which
            // is not a version. Fully authored and never promoted looked
            // identical to fully authored and training.
            + (promoted && promoted.clip_id === modeFile?.clip_id
                ? ` · promoted as v${promoted.version}.py`
                : " · not promoted yet")
          : undefined,
        tab: "rewards",
        action: modeCount ? "Continue authoring" : "Scaffold modes",
        optional: true,
      },
      {
        key: "reward",
        label: "Put a reward in the chain",
        hint: "Only v<n>.py counts. Promote the per-mode reward, or let the "
            + "sculptor iterate from the grounded starting reward.",
        done: rewardShaped || !!promoted,
        evidence: rewardEvidence,
        tab: "rewards",
        action: "Open rewards",
      },
      {
        key: "train",
        label: "Train",
        hint: "Launch a run, or decompose the goal into a staged mission.",
        done: hasIters,
        evidence: hasIters
          ? `${project.n_iterations_completed} iteration${project.n_iterations_completed === 1 ? "" : "s"}`
          : undefined,
        tab: "training",
        action: "New run",
      },
      {
        key: "export",
        label: "Take the results",
        hint: "Build the evidence report, or export a deployment bundle.",
        done: (policies.data?.length ?? 0) > 0,
        evidence: policies.data?.length
          ? `${policies.data.length} checkpoint${policies.data.length === 1 ? "" : "s"}`
          : undefined,
        tab: "results",
        action: "Open results",
      },
    ];
  }, [project, robotConfigured, world.data, rewards.data, policies.data,
      draft.data, modeRewards.data]);

  // Don't flash a wrong list while the queries settle.
  if (world.isLoading || rewards.isLoading || policies.isLoading) return null;

  // The next thing to do is the first incomplete step that isn't optional,
  // OR an optional step the user has already started. An untouched optional
  // step must never be "next" — that would tell someone who just wants to
  // train a flat-ground gait that they have to go compose a motion first.
  const next = steps.find((s) => !s.done && (!s.optional || !!s.evidence))
    ?? steps.find((s) => !s.done && !s.optional);
  const goal = draft.data?.behavior_goal || project.description || "";

  return (
    <div className="rs-card">
      <div className="rs-card-head">
        <div className="rs-card-title">
          <Icon name="list" size={16} />
          Build a behavior
        </div>
        <span className="rs-grow" />
        <span className="rs-sub" style={{ fontSize: 11 }}>
          {steps.filter((s) => s.done).length}/{steps.length} done
        </span>
      </div>

      {goal && (
        <div
          className="rs-card-pad"
          style={{ paddingBottom: 0, paddingTop: 10 }}
        >
          <div className="rs-sub" style={{ fontSize: 10.5 }}>Goal</div>
          <div style={{ fontSize: 12.5, lineHeight: 1.5, marginTop: 2 }}>
            {goal}
          </div>
        </div>
      )}

      <div className="rs-card-pad rs-vgap-8">
        {steps.map((s, i) => {
          const isNext = next === s;
          return (
            <div
              key={s.key}
              className="rs-flex rs-gap-10"
              style={{
                alignItems: "flex-start",
                padding: "7px 8px",
                margin: "0 -8px",
                borderRadius: "var(--radius-sm)",
                background: isNext
                  ? "color-mix(in srgb, var(--rs-primary) 6%, transparent)"
                  : undefined,
              }}
            >
              <span
                className="rs-sub mono"
                style={{ fontSize: 10.5, width: 14, flex: "0 0 14px",
                         paddingTop: 2, textAlign: "right" }}
              >
                {i + 1}
              </span>
              <Icon
                name={s.done ? "check-circle" : "circle"}
                size={15}
                color={s.done
                  ? "var(--st-emerald)"
                  : isNext ? "var(--rs-primary)" : "var(--rs-muted)"}
              />
              <div style={{ minWidth: 0, flex: 1 }}>
                <div style={{ fontSize: 13, fontWeight: isNext ? 650 : 450 }}>
                  {s.label}
                  {s.optional && !s.done && (
                    <span className="rs-sub" style={{ fontSize: 10.5, fontWeight: 400 }}>
                      {" "}· optional
                    </span>
                  )}
                </div>
                <div
                  className="rs-sub"
                  style={{ fontSize: 10.8, lineHeight: 1.45, marginTop: 2 }}
                >
                  {s.hint}
                </div>
                {s.evidence && (
                  <div
                    className="mono"
                    style={{
                      fontSize: 10.5, marginTop: 3,
                      color: s.done ? "var(--st-emerald)" : "var(--rs-muted)",
                      overflow: "hidden", textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                    title={s.evidence}
                  >
                    {s.evidence}
                  </div>
                )}
              </div>
              <Btn
                kind={isNext ? "primary" : "quiet"}
                size="xs"
                onClick={() => onGoTo(s.tab)}
              >
                {s.action}
              </Btn>
            </div>
          );
        })}
      </div>
    </div>
  );
}
