import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { deleteSaved, getSaved, listSaved } from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import type { SavedEntryDetail, SavedEntrySummary } from "@/lib/types";

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
