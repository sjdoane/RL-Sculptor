import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";

import {
  createProject,
  deleteProject,
  getProject,
  listProjects,
} from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import type {
  CreateProjectPayload,
  ProjectDetail,
  ProjectSummary,
} from "@/lib/types";

export function useProjects() {
  return useQuery<ProjectSummary[]>({
    queryKey: qk.projects(),
    queryFn: listProjects,
  });
}

export function useProject(slug: string | undefined) {
  return useQuery<ProjectDetail>({
    queryKey: slug ? qk.project(slug) : ["project", "_none"],
    queryFn: () => getProject(slug!),
    enabled: !!slug,
  });
}

export function useCreateProject() {
  const qc = useQueryClient();
  return useMutation<ProjectDetail, Error, CreateProjectPayload>({
    mutationFn: createProject,
    onSuccess: (project) => {
      qc.invalidateQueries({ queryKey: qk.projects() });
      qc.setQueryData(qk.project(project.slug), project);
    },
  });
}

export function useDeleteProject() {
  const qc = useQueryClient();
  return useMutation<void, Error, string>({
    mutationFn: deleteProject,
    onSuccess: (_void, slug) => {
      qc.invalidateQueries({ queryKey: qk.projects() });
      qc.removeQueries({ queryKey: qk.project(slug) });
      qc.removeQueries({ queryKey: qk.robot(slug) });
      qc.removeQueries({ queryKey: qk.preview(slug) });
    },
  });
}
