import { useCallback, useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { ApiError, fetchPreviewBlob } from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import type { CameraAngle } from "@/lib/types";

interface UseProjectPreviewOptions {
  /** Camera angle preset. Defaults to "iso". Part of the cache key,
   *  so switching angles is surgical — iso results aren't clobbered
   *  when the user flips to top-down. */
  angle?: CameraAngle;
  /** Force the server to re-render (bypasses its cached PNG). Also
   *  part of the cache key. */
  regenerate?: boolean;
  /** Pause fetching (e.g., while slug is undefined during route
   *  transitions). */
  enabled?: boolean;
}

/** Canonical preview fetcher. Every consumer — ProjectCard thumbnail,
 *  ProjectDetail preview panel, library picker cards — uses this hook.
 *  Benefits:
 *    - one place to tweak staleTime / retry policy;
 *    - cache key carries `angle` + `regenerate` so invalidation is
 *      surgical across the three surfaces;
 *    - we revoke the object URL on unmount / key change so we don't
 *      leak blob memory.
 *
 *  Returned `data` is an object URL pointing at the PNG. Bind it to
 *  `<img src={data} />` in the caller.
 *
 *  Returns an extra `invalidate()` helper that clears every cached
 *  variant for this slug — use it from the "Re-render" button.
 */
export function useProjectPreview(
  slug: string | undefined,
  options: UseProjectPreviewOptions = {},
) {
  const { angle = "iso", regenerate = false, enabled = true } = options;
  const qc = useQueryClient();

  const query = useQuery<string, ApiError>({
    queryKey: slug
      ? qk.preview(slug, { angle, regenerate })
      : ["preview", "_none"],
    queryFn: () => fetchPreviewBlob(slug!, { angle, regenerate }),
    enabled: !!slug && enabled,
    staleTime: 60_000,
    gcTime: 5 * 60_000,
    retry: false,
  });

  // Revoke the object URL when the component unmounts or the URL
  // changes so we don't leak blobs across angle-switch or project-
  // switch navigations.
  useEffect(() => {
    const url = query.data;
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [query.data]);

  const invalidate = useCallback(() => {
    if (!slug) return;
    // Match every ["preview", slug, ...] entry regardless of angle /
    // regenerate shape. React Query's partial key matcher handles this
    // out of the box.
    qc.invalidateQueries({ queryKey: ["preview", slug] });
  }, [qc, slug]);

  return { ...query, invalidate };
}
