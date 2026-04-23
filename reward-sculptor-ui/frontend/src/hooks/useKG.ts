import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  addSeeds,
  getPaper,
  getPendingSeeds,
  listPapers,
  listTechniques,
  researchKgTopic,
} from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import type {
  AddSeedsPayload,
  JobSummary,
  PaperDetail,
  PaperSummary,
  PendingSeedsResponse,
  TechniqueSummary,
} from "@/lib/types";

export interface PapersFilter {
  extracted?: boolean;
  search?: string;
}

export function usePapers(slug: string | undefined, filter: PapersFilter = {}) {
  return useQuery<PaperSummary[]>({
    queryKey: slug ? qk.kgPapers(slug, filter) : ["kg", "papers", "_none"],
    queryFn: () => listPapers(slug!, filter),
    enabled: !!slug,
    staleTime: 10_000,
  });
}

export function usePaper(slug: string | undefined, arxivId: string | undefined) {
  return useQuery<PaperDetail>({
    queryKey:
      slug && arxivId
        ? qk.kgPaper(slug, arxivId)
        : ["kg", "paper", "_none"],
    queryFn: () => getPaper(slug!, arxivId!),
    enabled: !!slug && !!arxivId,
  });
}

export function useTechniques(slug: string | undefined) {
  return useQuery<TechniqueSummary[]>({
    queryKey: slug ? qk.kgTechniques(slug) : ["kg", "techniques", "_none"],
    queryFn: () => listTechniques(slug!),
    enabled: !!slug,
  });
}

export function usePendingSeeds(slug: string | undefined) {
  return useQuery<PendingSeedsResponse>({
    queryKey: slug ? qk.kgPendingSeeds(slug) : ["kg", "pendingSeeds", "_none"],
    queryFn: () => getPendingSeeds(slug!),
    enabled: !!slug,
  });
}

/** Kicks off an ingest (+ optional extract) job. Returns the JobSummary
 *  with `job_id`; caller passes it to `useJob` for polling. */
export function useAddSeeds(slug: string) {
  const qc = useQueryClient();
  return useMutation<JobSummary, Error, AddSeedsPayload>({
    mutationFn: (payload) => addSeeds(slug, payload),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.kgPendingSeeds(slug) });
    },
  });
}

export interface ResearchTopicPayload {
  topic: string;
  max_papers?: number;
  auto_extract?: boolean;
}

/** Fires POST /projects/{slug}/kg/research. Claude picks arxiv IDs
 * relevant to the topic, dedupes against the shared KG, then ingests
 * + extracts them. Returns a JobSummary for progress polling. */
export function useResearchTopic(slug: string) {
  const qc = useQueryClient();
  return useMutation<JobSummary, Error, ResearchTopicPayload>({
    mutationFn: (payload) => researchKgTopic(slug, payload),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.kgPapers(slug) });
    },
  });
}
