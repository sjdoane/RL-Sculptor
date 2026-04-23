import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  getRobot,
  pickLibraryRobot,
  uploadRobotModel,
} from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import type { LibraryRobotName, RobotStateResponse } from "@/lib/types";

export function useRobot(slug: string | undefined) {
  return useQuery<RobotStateResponse>({
    queryKey: slug ? qk.robot(slug) : ["robot", "_none"],
    queryFn: () => getRobot(slug!),
    enabled: !!slug,
  });
}

export function usePickLibraryRobot(slug: string) {
  const qc = useQueryClient();
  return useMutation<RobotStateResponse, Error, LibraryRobotName>({
    mutationFn: (robotName) => pickLibraryRobot(slug, robotName),
    onSuccess: (robot) => {
      qc.setQueryData(qk.robot(slug), robot);
      qc.invalidateQueries({ queryKey: qk.project(slug) });
      qc.invalidateQueries({ queryKey: qk.projects() });
      qc.invalidateQueries({ queryKey: ["preview", slug] });
    },
  });
}

export function useUploadRobotModel(slug: string) {
  const qc = useQueryClient();
  return useMutation<
    RobotStateResponse,
    Error,
    { modelFile: File; meshesZip?: File }
  >({
    mutationFn: ({ modelFile, meshesZip }) =>
      uploadRobotModel(slug, modelFile, meshesZip),
    onSuccess: (robot) => {
      qc.setQueryData(qk.robot(slug), robot);
      qc.invalidateQueries({ queryKey: qk.project(slug) });
      qc.invalidateQueries({ queryKey: qk.projects() });
      qc.invalidateQueries({ queryKey: ["preview", slug] });
    },
  });
}
