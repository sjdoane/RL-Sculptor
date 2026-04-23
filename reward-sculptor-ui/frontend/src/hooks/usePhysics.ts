import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { getPhysics, promptPhysicsEdit, rematerializePhysics } from "@/lib/api";
import type { JobSummary, PhysicsLoadResponse } from "@/lib/types";

export function usePhysics(slug: string | undefined) {
  return useQuery<PhysicsLoadResponse>({
    queryKey: slug ? ["physics", slug] : ["physics", "_none"],
    queryFn: () => getPhysics(slug!),
    enabled: !!slug,
    staleTime: 60_000,
    retry: false,
  });
}

/** POST /projects/{slug}/physics/prompt → fires a background job,
 * returns its handle. Caller should watch the job via useJob and
 * invalidate the physics query on completion. */
export function usePhysicsPromptEdit(slug: string) {
  const qc = useQueryClient();
  return useMutation<JobSummary, Error, { prompt: string }>({
    mutationFn: ({ prompt }) => promptPhysicsEdit(slug, prompt),
    onSettled: () => {
      qc.invalidateQueries({ queryKey: ["physics", slug] });
    },
  });
}

/** POST /projects/{slug}/physics/rematerialize → wipes uploads/robot/
 * and re-copies the library MJCF + assets. Heals broken states where
 * the original materialize dropped mesh assets (H1 bug). Primes the
 * ["physics", slug] cache with the fresh response so the tab's panels
 * update without an extra fetch. */
export function usePhysicsRematerialize(slug: string) {
  const qc = useQueryClient();
  return useMutation<PhysicsLoadResponse, Error, void>({
    mutationFn: () => rematerializePhysics(slug),
    onSuccess: (data) => {
      qc.setQueryData(["physics", slug], data);
    },
  });
}
