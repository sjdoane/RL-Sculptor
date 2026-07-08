import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { listTrash, purgeTrash, restoreTrash } from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import type { TrashEntry } from "@/lib/types";

/** GET /trash — soft-deleted entries recoverable from Settings → Trash.
 *  A restore or purge changes the list AND may change projects/saved, so
 *  the mutations invalidate broadly. */
export function useTrash() {
  return useQuery<TrashEntry[]>({
    queryKey: qk.trash(),
    queryFn: listTrash,
  });
}

export function useRestoreTrash() {
  const qc = useQueryClient();
  return useMutation<void, Error, string>({
    mutationFn: (entryId) => restoreTrash(entryId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.trash() });
      // A restored entry reappears wherever it lives — refresh the
      // surfaces that list projects / saved missions.
      qc.invalidateQueries({ queryKey: qk.projects() });
      qc.invalidateQueries({ queryKey: ["saved"] });
    },
  });
}

export function usePurgeTrash() {
  const qc = useQueryClient();
  return useMutation<void, Error, string>({
    mutationFn: (entryId) => purgeTrash(entryId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.trash() });
    },
  });
}
