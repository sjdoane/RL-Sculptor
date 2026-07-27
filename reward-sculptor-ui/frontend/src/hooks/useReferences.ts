import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  attachStageReference,
  composeReference,
  detachStageReference,
  listReferences,
  searchReferences,
} from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import type { MissionDetail, RefIndexRow, RefMatch } from "@/lib/types";

/** GET /references?q=... — deterministic (llm=0) search, the picker's
 *  as-you-type path. Disabled for an empty/whitespace query — the
 *  caller falls back to useReferenceIndex for a "browse everything"
 *  empty state instead of hitting the search endpoint with "". */
export function useReferenceSearch(
  query: string,
  opts?: { robot?: string; enabled?: boolean },
) {
  const robot = opts?.robot ?? "g1";
  const trimmed = query.trim();
  const enabled = trimmed.length > 0 && (opts?.enabled ?? true);
  return useQuery<RefMatch[]>({
    queryKey: qk.referenceSearch(trimmed, robot),
    queryFn: () => searchReferences(trimmed, { robot, useLlm: false }),
    enabled,
  });
}

/** GET /references (no q) — slim index listing, used when the search
 *  box is empty so the picker still shows something browsable. */
export function useReferenceIndex(
  opts?: { robot?: string; enabled?: boolean },
) {
  const robot = opts?.robot ?? "g1";
  return useQuery<RefIndexRow[]>({
    queryKey: qk.referenceIndex(robot),
    queryFn: () => listReferences({ robot }),
    enabled: opts?.enabled ?? true,
  });
}

export interface AttachStageReferenceVariables {
  missionSlug: string;
  stageName: string;
  clipId: string;
}

/** POST .../stages/{stage}/reference. Invalidates the mission query on
 *  success so StageCard picks up the new reference_clip_id/tier
 *  immediately. 409 (mission jobs live) surfaces via ApiError — the
 *  caller (ReferencePickerDialog) toasts it. */
export function useAttachStageReference(slug: string) {
  const qc = useQueryClient();
  return useMutation<MissionDetail, Error, AttachStageReferenceVariables>({
    mutationFn: ({ missionSlug, stageName, clipId }) =>
      attachStageReference(slug, missionSlug, stageName, clipId),
    onSuccess: (_mission, { missionSlug }) => {
      qc.invalidateQueries({ queryKey: qk.mission(slug, missionSlug) });
    },
  });
}

export interface DetachStageReferenceVariables {
  missionSlug: string;
  stageName: string;
}

export function useDetachStageReference(slug: string) {
  const qc = useQueryClient();
  return useMutation<MissionDetail, Error, DetachStageReferenceVariables>({
    mutationFn: ({ missionSlug, stageName }) =>
      detachStageReference(slug, missionSlug, stageName),
    onSuccess: (_mission, { missionSlug }) => {
      qc.invalidateQueries({ queryKey: qk.mission(slug, missionSlug) });
    },
  });
}

/** POST /references/compose — build a novel clip from spans of solved ones.
 *
 *  Invalidates every reference query on success: the composite is a real
 *  library clip the moment it registers, so the picker's browse listing and
 *  any open search must show it immediately (it is usually the clip the user
 *  is about to attach). */
export function useComposeReference() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: composeReference,
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["references"] });
    },
  });
}
