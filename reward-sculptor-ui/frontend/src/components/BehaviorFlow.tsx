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
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { ReferencePickerDialog } from "@/components/ReferencePickerDialog";
import { Icon } from "@/components/rs/icon";
import { Btn } from "@/components/rs/primitives";
import { useBehaviorDraft, useSaveBehaviorDraft } from "@/hooks/useBehaviorDraft";
import { usePolicies } from "@/hooks/usePolicies";
import { useRewards } from "@/hooks/useRewards";
import { useWorldSelection } from "@/hooks/useWorlds";
import { listModeRewards } from "@/lib/api";
import { referenceRobotForProject } from "@/lib/referenceRobot";
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
  /** Do the step HERE instead of navigating to the tab that hosts it.
   *  Without this the card is a table of contents: its only interaction was
   *  a tab switch, so the answer to "you have to go to five different
   *  places" was a list of the five places. */
  run?: () => void;
  /** Why the button is unavailable right now; also disables it. */
  blocked?: string;
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
  const saveDraft = useSaveBehaviorDraft(slug);
  const [pickingMotion, setPickingMotion] = useState(false);
  const referenceRobot = referenceRobotForProject(project);
  const modeRewards = useQuery({
    queryKey: ["modeRewards", slug],
    queryFn: () => listModeRewards(slug),
    retry: false,
    staleTime: 10_000,
  });

  const steps: Step[] = useMemo(() => {
    const hasIters = project.n_iterations_completed > 0;
    const training = project.status === "running";
    const clipId = draft.data?.reference_clip_id ?? "";
    // Match the chosen clip, or — with no clip chosen — whatever is actually
    // promoted. The old `!clipId || ...` fell through to the FIRST file in an
    // mtime-sorted list, so a project with no motion chosen could show
    // "Choose a reference motion" unchecked and "4/4 authored" checked
    // directly beneath it, for a clip the user had never selected.
    const promotedClip = modeRewards.data?.promoted?.clip_id ?? "";
    const matchClip = clipId || promotedClip;
    const modeFile = matchClip
      ? modeRewards.data?.mode_rewards.find((f) => f.clip_id === matchClip)
      : undefined;
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
    // A fully authored per-mode reward that was never promoted is inert —
    // the chain still trains something else. Counting the reward step done
    // then puts a green check immediately above a row reading "not promoted
    // yet", which is the exact confusion this card exists to prevent.
    const modesReadyUnpromoted =
      modeCount > 0 && authoredCount === modeCount
      && !(promoted && promoted.clip_id === modeFile?.clip_id);
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
        action: robotConfigured ? "Show robot" : "Configure",
        // The card only ever renders on the Overview tab, so `onGoTo(
        // "overview")` was a button that provably did nothing. The robot
        // card is in the other column of this same screen.
        run: () => document
          .getElementById("robot-config")
          ?.scrollIntoView({ behavior: "smooth", block: "start" }),
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
        // Opened here. The button used to navigate to the Training tab,
        // which contains no motion control at all — the picker is mounted
        // only inside the run dialog, four unsignposted clicks and three
        // modal levels down. That was the "how to reference the novel task"
        // complaint, verbatim.
        run: () => setPickingMotion(true),
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
        done: !modesReadyUnpromoted && (rewardShaped || !!promoted),
        evidence: rewardEvidence,
        tab: "rewards",
        action: modesReadyUnpromoted ? "Promote it" : "Open rewards",
      },
      {
        key: "train",
        label: "Train",
        hint: "Launch a run, or decompose the goal into a staged mission.",
        done: hasIters,
        // A run can go hours before its first sculpt iteration lands, and
        // `n_iterations_completed` is the only training signal here — so
        // this row used to sit unchecked, highlighted, offering "New run"
        // to someone who had just started one.
        evidence: training
          ? (hasIters
              ? `running · ${project.n_iterations_completed} done`
              : "running")
          : hasIters
            ? `${project.n_iterations_completed} iteration${project.n_iterations_completed === 1 ? "" : "s"}`
            : undefined,
        tab: "training",
        action: training ? "Watch run" : "New run",
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
                disabled={!!s.blocked}
                title={s.blocked}
                onClick={s.run ?? (() => onGoTo(s.tab))}
              >
                {s.action}
              </Btn>
            </div>
          );
        })}
      </div>

      {/* Standalone mode: the pick comes back here rather than mutating a
          mission stage, and is written to the draft immediately — the run
          dialog and the mode-reward panel both read it from there, so the
          clip survives closing this and never has to be re-found in a
          6000-clip library. Composing is reachable from inside the picker. */}
      {pickingMotion && (
        <ReferencePickerDialog
          slug={slug}
          robot={referenceRobot}
          currentClipId={draft.data?.reference_clip_id ?? null}
          initialQuery={goal}
          onPick={({ clipId, robot }) => {
            saveDraft.mutate({
              reference_clip_id: clipId,
              reference_robot: robot,
            });
          }}
          onClose={() => setPickingMotion(false)}
        />
      )}
    </div>
  );
}
