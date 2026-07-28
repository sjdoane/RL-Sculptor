import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { deleteSaved, getSaved, listSaved, saveMission } from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import type { JobDetail, SavedEntryDetail, SavedEntrySummary } from "@/lib/types";

/** GET /saved — durable mission archive, rebuilt from disk on load so a
 *  backend restart never hides it. */
export function useSaved() {
  return useQuery<SavedEntrySummary[]>({
    queryKey: qk.saved(),
    queryFn: listSaved,
  });
}

export function useSavedEntry(
  entryId: string | undefined,
  opts?: { enabled?: boolean },
) {
  const enabled = !!entryId && (opts?.enabled ?? true);
  return useQuery<SavedEntryDetail>({
    queryKey: entryId ? qk.savedEntry(entryId) : ["saved", "_none"],
    queryFn: () => getSaved(entryId!),
    enabled,
  });
}

export interface SaveMissionVariables {
  missionSlug: string;
  /** Stage name → iter indices to keep no matter what. Without pins the
   *  archive keeps only each stage's final and best checkpoints. */
  pinned?: Record<string, number[]>;
}

/** POST /projects/{slug}/missions/{ms}/save — archive a mission on demand.
 *
 *  The archive also auto-populates as missions run (sculptor's
 *  `_archive_mission_snapshot`), which is why this being uncallable went
 *  unnoticed: the page was never empty. But SavedMissionsPage's own empty
 *  state says "Save a mission from its detail view", and until now there
 *  was no such control anywhere in the UI — nor any way to pin a specific
 *  iteration's checkpoint, which only this request body can express.
 *
 *  Returns a 202 JobDetail: archiving copies checkpoints, so it runs as a
 *  background job rather than blocking the dialog. */
export function useSaveMission(slug: string) {
  const qc = useQueryClient();
  return useMutation<JobDetail, Error, SaveMissionVariables>({
    mutationFn: ({ missionSlug, pinned }) =>
      saveMission(slug, missionSlug, pinned ? { pinned } : undefined),
    onSuccess: () => {
      // The job writes the entry asynchronously; invalidate so the archive
      // list refetches when the user next opens it.
      qc.invalidateQueries({ queryKey: qk.saved() });
    },
  });
}

/** DELETE /saved/{id} — moves the entry to Trash (recoverable). */
export function useDeleteSaved() {
  const qc = useQueryClient();
  return useMutation<void, Error, string>({
    mutationFn: (entryId) => deleteSaved(entryId),
    onSuccess: (_r, entryId) => {
      qc.invalidateQueries({ queryKey: qk.saved() });
      qc.removeQueries({ queryKey: qk.savedEntry(entryId) });
      // It now lives in Trash — refresh that surface too.
      qc.invalidateQueries({ queryKey: qk.trash() });
    },
  });
}
