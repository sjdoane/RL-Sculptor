/**
 * The pipeline as one ordered list, on one screen.
 *
 * The showcase workflow — author a world, compose a novel motion out of
 * solved clips, split it into reward phases, author a reward per phase, train —
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
import { useHasActiveRun } from "@/hooks/useRuns";
import { useWorldSelection } from "@/hooks/useWorlds";
import { getReference, listModeRewards } from "@/lib/api";
import { deriveModeRewardReadiness } from "@/lib/behaviorFlow";
import { hasExactTierDReceipt } from "@/lib/referenceAdmission";
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
  /** A researcher command that must be run outside this UI. */
  externalCommand?: string;
  /** Why an external command cannot be formed yet. */
  externalCommandIssue?: string;
  /** Full immutable identities, kept out of the plain-language primary line. */
  technicalReceipt?: string;
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

const CERTIFICATION_TERM_LABELS: Record<string, string> = {
  root_xy_tracking: "root-XY tracking",
  contact_safety: "contact safety",
  collision_avoidance: "collision avoidance",
  general_dynamics_feasibility: "general dynamics feasibility",
};

function certificationTermLabel(term: string): string {
  return CERTIFICATION_TERM_LABELS[term] ?? term.replaceAll("_", " ");
}

function shellQuote(value: string): string {
  return `'${value.replaceAll("'", `'"'"'`)}'`;
}

export function tierDCertificationCommands({
  clipId,
  robot,
  donorProject,
  targetProject,
}: {
  clipId: string;
  robot: string;
  donorProject: string;
  targetProject: string;
}): { commands: string[]; error: string | null } {
  const donor = donorProject.trim();
  if (!donor) {
    return {
      commands: [],
      error: "Choose an existing trusted donor project before generating certification commands.",
    };
  }
  if (!donor.startsWith("/") || /[\r\n\0]/.test(donor)) {
    return {
      commands: [],
      error: "The trusted donor must be one absolute local WSL path on a single line.",
    };
  }
  const normalized = (value: string) => value.trim().replace(/[\\/]+$/, "");
  if (normalized(donor) === normalized(targetProject)) {
    return {
      commands: [],
      error: "The new target project is not a donor. Choose an existing trusted project with the compatible robot/task interface.",
    };
  }
  const donorArg = shellQuote(donor);
  const base = `sculpt refs track --clip-id ${shellQuote(clipId)} --robot ${shellQuote(robot)} --donor-project ${donorArg}`;
  return {
    commands: [
      `sculpt refs export-tierd-interface --donor-project ${donorArg}`,
      `${base} --dry-run`,
      base,
    ],
    error: null,
  };
}

/** Scroll to an element that does not exist yet.
 *
 *  Switching tabs mounts a lazily-loaded route, so the target is absent for
 *  a frame or several. Without the retry the flow's "author a reward per
 *  mode" button lands you at the top of the Rewards tab, with the panel it
 *  means still below a 420px editor and two other cards. Gives up quietly. */
function scrollToWhenReady(id: string, tries = 40) {
  const el = document.getElementById(id);
  if (el) {
    el.scrollIntoView({ behavior: "smooth", block: "start" });
    return;
  }
  if (tries > 0) requestAnimationFrame(() => scrollToWhenReady(id, tries - 1));
}

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
  const [trustedDonorProject, setTrustedDonorProject] = useState("");
  // §Ship 37: the live-run signal. `project.status` is never "running" —
  // see `hasActiveRun` — so the Train row below used to offer "New run" to
  // someone whose run was already hours in.
  const training = useHasActiveRun(slug);
  const referenceRobot = referenceRobotForProject(project);
  const selectedClipId = draft.data?.reference_clip_id?.trim() ?? "";
  const selectedClipRobot =
    draft.data?.reference_robot?.trim() || referenceRobot;
  const modeRewards = useQuery({
    queryKey: ["modeRewards", slug],
    queryFn: () => listModeRewards(slug),
    retry: false,
    staleTime: 10_000,
  });
  const referenceDetail = useQuery({
    queryKey: ["reference-detail", selectedClipRobot, selectedClipId],
    queryFn: () => getReference(selectedClipRobot, selectedClipId),
    enabled: selectedClipId.length > 0,
    retry: false,
    staleTime: 10_000,
  });

  const steps: Step[] = useMemo(() => {
    const hasIters = project.n_iterations_completed > 0;
    const clipId = selectedClipId;
    const clipRobot = selectedClipRobot;
    const dynamicsAdmission = referenceDetail.data?.dynamics_admission ?? null;
    const artifactIdentity = referenceDetail.data?.artifact_identity ?? null;
    const certificationScope = dynamicsAdmission?.certification_scope ?? null;
    const certificationExclusions = certificationScope?.not_certified
      .map(certificationTermLabel)
      .join(", ") || "the exclusions recorded in its exact receipt";
    const hasExactDynamicsReceipt = hasExactTierDReceipt(referenceDetail.data);
    const certificationDone =
      referenceDetail.isSuccess
      && dynamicsAdmission?.admitted === true
      && hasExactDynamicsReceipt;
    const certificationCommands = tierDCertificationCommands({
      clipId,
      robot: clipRobot,
      donorProject: trustedDonorProject,
      targetProject: project.project_dir,
    });
    const certificationEvidence = certificationDone
      ? `Tier-D exact-schedule tracking evidence verified`
        + (dynamicsAdmission.certificate_digest
            ? ` · certificate ${dynamicsAdmission.certificate_digest.slice(0, 12)}`
            : "")
        + (dynamicsAdmission.rollout_sha256
            ? ` · rollout ${dynamicsAdmission.rollout_sha256.slice(0, 12)}`
            : "")
      : referenceDetail.isLoading
        ? `Checking ${clipRobot}/${clipId}…`
        : referenceDetail.isError
          ? `Could not verify ${clipRobot}/${clipId}`
          : dynamicsAdmission?.admitted
            ? "Tier D evidence is incomplete · live training blocked"
            : `Tier ${dynamicsAdmission?.tier || "K"} candidate · live training blocked`
              + (dynamicsAdmission?.reason
                  ? ` · ${dynamicsAdmission.reason}`
                  : "");
    const certificationStep: Step[] = clipId
      ? [{
          key: "motion-certification",
          label: "Certify motion",
          hint: certificationDone
            ? "Tier-D evidence was earned by an external exact-schedule tracking job. It certifies "
              + "exact-schedule joint-position and root-height tracking only; "
              + `it does not certify ${certificationExclusions}. `
              + "This UI re-verifies the exact "
              + "clip and tracked-rollout bytes; it does not issue certificates."
            : "Live training requires Tier-D exact-schedule tracking evidence "
              + "from an external job. Select an existing trusted donor below, export its data-only "
              + "interface receipt, run CPU preflight, then run certification. This UI verifies "
              + "evidence but does not create certificates.",
          done: certificationDone,
          evidence: certificationEvidence,
          externalCommand: certificationDone
            ? undefined
            : certificationCommands.commands.join("\n") || undefined,
          externalCommandIssue: certificationDone
            ? undefined
            : certificationCommands.error ?? undefined,
          technicalReceipt: certificationDone
            ? [
                `robot=${clipRobot}`,
                `clip_id=${clipId}`,
                `artifact_clip_sha256=${dynamicsAdmission.clip_sha256}`,
                `raw_source_sha256=${artifactIdentity?.source_content_sha256 || "unavailable"}`,
                `certificate_digest=${dynamicsAdmission.certificate_digest}`,
                `rollout_sha256=${dynamicsAdmission.rollout_sha256}`,
                `certification_claim=${certificationScope?.claim || "unavailable"}`,
                `not_certified=${certificationScope?.not_certified.join(",") || "unavailable"}`,
              ].join("\n")
            : undefined,
          tab: "training",
          action: referenceDetail.isLoading
            ? "Checking…"
            : certificationDone ? "Re-check evidence" : "Refresh status",
          run: () => { void referenceDetail.refetch(); },
          blocked: referenceDetail.isFetching
            ? "Checking the current Tier-D exact-schedule tracking evidence…"
            : undefined,
        }]
      : [];
    // Match the chosen clip, or — with no clip chosen — whatever is actually
    // promoted. The old `!clipId || ...` fell through to the FIRST file in an
    // mtime-sorted list, so a project with no motion chosen could show
    // "Choose a reference motion" unchecked and "4/4 authored" checked
    // directly beneath it, for a clip the user had never selected.
    const promotedClip = modeRewards.data?.promoted?.clip_id ?? "";
    const promotedRobot = modeRewards.data?.promoted?.reference_robot ?? "";
    const matchClip = clipId || promotedClip;
    const matchRobot = clipId ? clipRobot : promotedRobot;
    const modeReadiness = deriveModeRewardReadiness({
      files: modeRewards.data?.mode_rewards ?? [],
      promoted: modeRewards.data?.promoted ?? null,
      clipId: matchClip,
      robot: matchRobot,
    });
    const modeFile = modeReadiness.modeFile ?? undefined;
    const authoredCount = modeReadiness.authoredCount;
    const modeCount = modeReadiness.modeCount;
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
      modeReadiness.authoredCurrent && !modeReadiness.promotedExact;
    const rewardEvidence = promoted
      ? `v${promoted.version}.py · ${promoted.modes.length} modes`
        + (promoted.unauthored.length
            ? ` · ${promoted.unauthored.length} still a stub`
            : " · all authored")
        + (modeReadiness.promotionBlocker
            ? ` · blocked: ${modeReadiness.promotionBlocker}`
            : " · exact current selection")
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
            + "several solved ones. It becomes an immutable tracking candidate; "
            + "the next step checks Tier-D exact-schedule tracking evidence.",
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
      ...certificationStep,
      {
        key: "modes",
        label: "Author phase-specific rewards",
        hint: "Each fixed clip-time window gets its own reward terms. Runtime "
            + "dispatch follows immutable episode-time windows; transition guards "
            + "are inspectable metadata only.",
        done: modeReadiness.authoredCurrent,
        evidence: modeCount
          ? `${authoredCount}/${modeCount} authored · ${modeFile?.filename}`
            // The gap this closes: authoring writes mode_reward_v<n>.py, which
            // is not a version. Fully authored and never promoted looked
            // identical to fully authored and training.
            + (modeReadiness.promotedExact
                ? ` · promoted as v${promoted?.version}.py`
                : " · not promoted yet")
            + (modeReadiness.modeBlocker
                ? ` · blocked: ${modeReadiness.modeBlocker}`
                : " · current context")
          : undefined,
        tab: "rewards",
        action: modeReadiness.modeBlocker
          ? "Refresh modes"
          : modeCount ? "Continue authoring" : "Scaffold modes",
        // Both this step and the next one target the Rewards tab, and the
        // panel each of them means is off-screen on arrival. Land on the
        // control, not the tab.
        run: () => { onGoTo("rewards"); scrollToWhenReady("mode-reward-panel"); },
        optional: true,
      },
      {
        key: "reward",
        label: "Put a reward in the chain",
        hint: "Only v<n>.py counts. Promote the per-mode reward, or let the "
            + "sculptor iterate from the grounded starting reward.",
        done: modeCount > 0
          ? modeReadiness.promotedExact
          : rewardShaped,
        evidence: rewardEvidence,
        tab: "rewards",
        action: modesReadyUnpromoted ? "Promote it" : "Open rewards",
        // "Promote it" means the button at the bottom of the mode panel, so
        // go there; plain "Open rewards" means the version chain at the top.
        run: modesReadyUnpromoted
          ? () => { onGoTo("rewards"); scrollToWhenReady("mode-reward-panel"); }
          : undefined,
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
      selectedClipId, selectedClipRobot, modeRewards.data, onGoTo, training,
      trustedDonorProject,
      referenceDetail.data, referenceDetail.isError, referenceDetail.isFetching,
      referenceDetail.isLoading, referenceDetail.isSuccess,
      referenceDetail.refetch]);

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
                    role={s.key === "motion-certification" ? "status" : undefined}
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
                {s.key === "motion-certification" && !s.done && (
                  <label
                    style={{ display: "grid", gap: 3, marginTop: 6, maxWidth: 620 }}
                  >
                    <span className="rs-sub" style={{ fontSize: 10.5 }}>
                      Trusted donor project · command input only, not yet verified
                    </span>
                    <input
                      className="rs-input mono"
                      aria-label="Trusted Tier-D donor project"
                      placeholder="/absolute/path/to/existing-compatible-project"
                      value={trustedDonorProject}
                      onChange={(event) => setTrustedDonorProject(event.target.value)}
                      spellCheck={false}
                    />
                  </label>
                )}
                {s.externalCommandIssue && (
                  <div
                    role="status"
                    style={{ color: "var(--st-amber)", fontSize: 10.5, lineHeight: 1.45, marginTop: 4 }}
                  >
                    {s.externalCommandIssue}
                  </div>
                )}
                {s.externalCommand && (
                  <div
                    aria-label="External certification command"
                    style={{ fontSize: 10.5, lineHeight: 1.45, marginTop: 5 }}
                  >
                    <div className="rs-sub">Run externally in order:</div>
                    <pre
                      className="mono"
                      style={{ margin: "3px 0 0", overflowWrap: "anywhere", whiteSpace: "pre-wrap" }}
                    >
                      {s.externalCommand}
                    </pre>
                  </div>
                )}
                {s.technicalReceipt && (
                  <details style={{ fontSize: 10.5, marginTop: 5 }}>
                    <summary style={{ cursor: "pointer" }}>
                      Exact Tier-D tracking receipt
                    </summary>
                    <pre
                      className="mono"
                      style={{
                        margin: "5px 0 0", overflowWrap: "anywhere",
                        whiteSpace: "pre-wrap",
                      }}
                    >
                      {s.technicalReceipt}
                    </pre>
                  </details>
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
