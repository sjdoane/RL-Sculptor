/** Environment-authoring hooks (env-authoring item 5).
 *
 *  The selection endpoint 404s until the first authoring — `retry: false`
 *  so an unauthored project doesn't hammer the backend, and consumers
 *  branch on `error` to show the empty state. Mutations invalidate both
 *  world keys plus the project (authoring can advance project status).
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  applyWorldAuthor,
  authorWorld,
  editWorldVariations,
  getWorldCurriculum,
  getWorldLineage,
  getWorldScene,
  getWorldSelection,
  getWorldValidate,
  previewWorldDraft,
} from "@/lib/api";
import type { WorldCurriculum } from "@/lib/types";
import { qk } from "@/lib/queryKeys";
import type {
  WorldApplyRequest,
  WorldApplyResponse,
  WorldAuthorRequest,
  WorldAuthorResponse,
  WorldDraftPreview,
  WorldLineageEntry,
  WorldScene,
  WorldSelection,
  WorldValidateResult,
} from "@/lib/types";

export function useWorldSelection(slug: string | undefined) {
  return useQuery<WorldSelection>({
    queryKey: slug ? qk.worldSelection(slug) : ["worldSelection", "_none"],
    queryFn: () => getWorldSelection(slug!),
    enabled: !!slug,
    staleTime: 30_000,
    retry: false,
  });
}

/** Independent integrity status for the authoritative tuple.  Consumers use
 *  this as a launch readiness signal; launch itself revalidates server-side. */
export function useWorldValidation(
  slug: string | undefined,
  enabled = true,
) {
  return useQuery<WorldValidateResult>({
    queryKey: slug ? qk.worldValidation(slug) : ["worldValidation", "_none"],
    queryFn: () => getWorldValidate(slug!),
    enabled: !!slug && enabled,
    staleTime: 15_000,
    retry: false,
  });
}

export function useWorldLineage(slug: string | undefined) {
  return useQuery<WorldLineageEntry[]>({
    queryKey: slug ? qk.worldLineage(slug) : ["worldLineage", "_none"],
    queryFn: () => getWorldLineage(slug!),
    enabled: !!slug,
    staleTime: 30_000,
    retry: false,
  });
}

export function useWorldCurriculum(slug: string | undefined) {
  return useQuery<WorldCurriculum>({
    queryKey: slug ? qk.worldCurriculum(slug) : ["worldCurriculum", "_none"],
    queryFn: () => getWorldCurriculum(slug!),
    enabled: !!slug,
    staleTime: 30_000,
    retry: false,
  });
}

export function useWorldScene(slug: string | undefined, enabled = true) {
  return useQuery<WorldScene>({
    queryKey: slug ? qk.worldScene(slug) : ["worldScene", "_none"],
    queryFn: () => getWorldScene(slug!),
    enabled: !!slug && enabled,
    // The scene is immutable per selection version; invalidation on
    // promote (below) is the only refresh that matters.
    staleTime: Infinity,
    retry: false,
  });
}

/** Gated dry-run for the build loop — returns the admission report +
 *  compiled draft scene without promoting anything. */
export function usePreviewWorldDraft(slug: string) {
  return useMutation<WorldDraftPreview, Error, WorldApplyRequest>({
    mutationFn: (body) => previewWorldDraft(slug, body),
  });
}

/** Train-variation edits on the promoted world (tweak-a-dimension loop).
 *  Promotes a new selection version under the same evaluation lineage,
 *  so every world query refreshes. */
export function useEditWorldVariations(slug: string) {
  const qc = useQueryClient();
  return useMutation<
    import("@/lib/types").WorldVariationEditResult,
    Error,
    Parameters<typeof editWorldVariations>[1]
  >({
    mutationFn: (body) => editWorldVariations(slug, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.worldSelection(slug) });
      qc.invalidateQueries({ queryKey: qk.worldValidation(slug) });
      qc.invalidateQueries({ queryKey: qk.worldLineage(slug) });
      qc.invalidateQueries({ queryKey: qk.worldScene(slug) });
    },
  });
}

export function useAuthorWorld(slug: string) {
  return useMutation<WorldAuthorResponse, Error, WorldAuthorRequest>({
    mutationFn: (body) => authorWorld(slug, body),
  });
}

export function useApplyWorldAuthor(slug: string) {
  const qc = useQueryClient();
  return useMutation<WorldApplyResponse, Error, WorldApplyRequest>({
    mutationFn: (body) => applyWorldAuthor(slug, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.worldSelection(slug) });
      qc.invalidateQueries({ queryKey: qk.worldValidation(slug) });
      qc.invalidateQueries({ queryKey: qk.worldLineage(slug) });
      qc.invalidateQueries({ queryKey: qk.worldScene(slug) });
      qc.invalidateQueries({ queryKey: qk.project(slug) });
    },
  });
}
