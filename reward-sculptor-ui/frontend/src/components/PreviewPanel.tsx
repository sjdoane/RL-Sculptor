import { useState } from "react";
import { ImageOff, Loader2, RefreshCcw } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Select } from "@/components/ui/select";
import { useProjectPreview } from "@/hooks/useProjectPreview";
import { ApiError, previewUrl } from "@/lib/api";
import type { CameraAngle } from "@/lib/types";
import { CAMERA_ANGLES } from "@/lib/types";

const ANGLE_LABEL: Record<CameraAngle, string> = {
  iso: "Iso",
  front: "Front",
  side: "Side",
  top: "Top",
};

export function PreviewPanel({ slug }: { slug: string }) {
  const [angle, setAngle] = useState<CameraAngle>("iso");
  const [reRendering, setReRendering] = useState(false);
  const { data: url, isLoading, error, invalidate } = useProjectPreview(
    slug,
    { angle },
  );

  const onReRender = async () => {
    setReRendering(true);
    try {
      // Hit the endpoint once with regenerate=true for this angle, then
      // invalidate every cached variant in React Query so the <img>
      // flips to the new blob when it lands.
      const res = await fetch(previewUrl(slug, { angle, regenerate: true }));
      if (!res.ok) {
        let msg = `HTTP ${res.status}`;
        try {
          const body = await res.json();
          msg = body.detail ?? body.title ?? msg;
        } catch {
          // ignore
        }
        throw new Error(msg);
      }
      invalidate();
      toast.success(`Re-rendered ${ANGLE_LABEL[angle]} view`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      toast.error("Re-render failed", { description: msg });
    } finally {
      setReRendering(false);
    }
  };

  return (
    <Card className="overflow-hidden">
      <CardHeader className="flex-row items-center justify-between gap-2 space-y-0 border-b py-2">
        <CardTitle className="text-sm">Preview</CardTitle>
        <div className="flex items-center gap-2">
          <Select
            value={angle}
            onChange={(e) => setAngle(e.target.value as CameraAngle)}
            className="h-7 w-24 text-xs"
            aria-label="Camera angle"
          >
            {CAMERA_ANGLES.map((a) => (
              <option key={a} value={a}>
                {ANGLE_LABEL[a]}
              </option>
            ))}
          </Select>
          <Button
            variant="ghost"
            size="sm"
            onClick={onReRender}
            disabled={reRendering || isLoading}
            aria-label="Re-render preview"
          >
            <RefreshCcw className={reRendering ? "animate-spin" : ""} />
            Re-render
          </Button>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="relative flex aspect-[4/3] w-full items-center justify-center bg-muted/30">
          {(isLoading || reRendering) && (
            <div className="absolute inset-0 z-10 flex items-center justify-center bg-background/60 backdrop-blur-sm">
              <Loader2 className="h-5 w-5 animate-spin text-foreground" />
            </div>
          )}
          {url ? (
            <img
              src={url}
              alt={`${ANGLE_LABEL[angle]} view of the configured robot`}
              className="h-full w-full object-contain"
            />
          ) : error ? (
            <ErrorBlock error={error} />
          ) : null}
        </div>
      </CardContent>
    </Card>
  );
}

function ErrorBlock({ error }: { error: ApiError | Error }) {
  const isApi = error instanceof ApiError;
  const title = isApi ? error.problem.title : "Preview unavailable";
  const detail = isApi ? error.problem.detail : error.message;
  return (
    <div className="flex max-w-md flex-col items-center gap-2 p-6 text-center">
      <ImageOff className="h-8 w-8 text-muted-foreground" />
      <div className="text-sm font-medium">{title}</div>
      {detail && (
        <p className="font-mono text-xs text-muted-foreground">{detail}</p>
      )}
    </div>
  );
}
