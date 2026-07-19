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
  getWorldLineage,
  getWorldSelection,
} from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import type {
  WorldApplyRequest,
  WorldApplyResponse,
  WorldAuthorRequest,
  WorldAuthorResponse,
  WorldLineageEntry,
  WorldSelection,
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

export function useWorldLineage(slug: string | undefined) {
  return useQuery<WorldLineageEntry[]>({
    queryKey: slug ? qk.worldLineage(slug) : ["worldLineage", "_none"],
    queryFn: () => getWorldLineage(slug!),
    enabled: !!slug,
    staleTime: 30_000,
    retry: false,
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
      qc.invalidateQueries({ queryKey: qk.worldLineage(slug) });
      qc.invalidateQueries({ queryKey: qk.project(slug) });
    },
  });
}
