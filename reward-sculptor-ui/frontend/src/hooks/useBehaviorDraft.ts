/** The behavior being built, carried across the steps that build it.
 *
 *  Composing a motion, authoring per-mode rewards for it, and launching a run
 *  that trains it were three disconnected screens. The composed clip id lived
 *  only in whichever dialog had just produced it, so the same id had to be
 *  re-found by hand in a ~6000-clip library twice, and the run dialog forgot
 *  both the goal and the motion between launches.
 *
 *  This is intent, not configuration. Every step still reads the
 *  authoritative artifact — the selection file, the reward chain, the run's
 *  own params — for truth; the draft only says what the user is working on.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  getBehaviorDraft,
  patchBehaviorDraft,
  type BehaviorDraft,
  type BehaviorDraftPatch,
} from "@/lib/api";

export function behaviorDraftKey(slug: string) {
  return ["behaviorDraft", slug] as const;
}

export function useBehaviorDraft(slug: string | undefined) {
  return useQuery<BehaviorDraft>({
    queryKey: behaviorDraftKey(slug ?? "_none"),
    queryFn: () => getBehaviorDraft(slug!),
    enabled: !!slug,
    // A fresh project has no draft; that is the normal state, not an error
    // worth retrying against.
    retry: false,
    staleTime: 10_000,
  });
}

export function useSaveBehaviorDraft(slug: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (patch: BehaviorDraftPatch) => patchBehaviorDraft(slug, patch),
    onSuccess: (draft) => {
      qc.setQueryData(behaviorDraftKey(slug), draft);
    },
  });
}
