import { useState } from "react";
import { Loader2, Sparkles } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useCreateMission } from "@/hooks/useMissions";
import { ApiError } from "@/lib/api";

const SLUG_PATTERN = /^[a-z][a-z0-9_-]{0,63}$/;
const GOAL_MIN = 8;
const GOAL_MAX = 2000;

export function NewMissionDialog({ slug }: { slug: string }) {
  const [open, setOpen] = useState(false);
  const [goal, setGoal] = useState("");
  const [missionSlug, setMissionSlug] = useState("");
  const [noKg, setNoKg] = useState(false);

  const create = useCreateMission(slug);

  const reset = () => {
    setGoal("");
    setMissionSlug("");
    setNoKg(false);
  };

  const submit = () => {
    const trimmedGoal = goal.trim();
    if (trimmedGoal.length < GOAL_MIN) {
      toast.error("Goal too short", {
        description: `At least ${GOAL_MIN} characters.`,
      });
      return;
    }
    if (trimmedGoal.length > GOAL_MAX) {
      toast.error("Goal too long", {
        description: `At most ${GOAL_MAX} characters.`,
      });
      return;
    }
    const trimmedSlug = missionSlug.trim();
    if (trimmedSlug && !SLUG_PATTERN.test(trimmedSlug)) {
      toast.error("Invalid mission slug", {
        description:
          "Lowercase letter first, then letters/digits/underscores/hyphens; ≤64 chars.",
      });
      return;
    }
    create.mutate(
      {
        goal: trimmedGoal,
        mission_slug: trimmedSlug || undefined,
        no_kg: noKg || undefined,
      },
      {
        onSuccess: (job) => {
          setOpen(false);
          reset();
          toast.success("Decompose job queued", {
            description: `job_id: ${job.job_id}`,
          });
        },
        onError: (err) => {
          const detail =
            err instanceof ApiError
              ? err.problem.detail ?? err.problem.title
              : err.message;
          toast.error("Could not create mission", { description: detail });
        },
      },
    );
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        setOpen(o);
        if (!o) reset();
      }}
    >
      <DialogTrigger asChild>
        <Button size="sm">
          <Sparkles />
          New mission
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>New mission</DialogTitle>
          <DialogDescription>
            Claude decomposes the goal into a curriculum of stages. The
            decompose job runs immediately; once it finishes, the
            mission becomes <code>ready</code> and can be run.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3">
          <div className="grid gap-1.5">
            <Label htmlFor="mission-goal">Goal</Label>
            <Textarea
              id="mission-goal"
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              placeholder="Stand on one leg and kick a ball with the other."
              rows={4}
              maxLength={GOAL_MAX}
              disabled={create.isPending}
            />
            <p className="text-[10.5px] text-muted-foreground">
              {goal.trim().length} / {GOAL_MAX} chars · min {GOAL_MIN}
            </p>
          </div>

          <div className="grid gap-1.5">
            <Label htmlFor="mission-slug">
              Mission slug{" "}
              <span className="text-[10.5px] font-normal text-muted-foreground">
                (optional override)
              </span>
            </Label>
            <Input
              id="mission-slug"
              value={missionSlug}
              onChange={(e) => setMissionSlug(e.target.value)}
              placeholder="auto — derived from goal"
              maxLength={64}
              disabled={create.isPending}
              spellCheck={false}
              autoCapitalize="off"
              autoCorrect="off"
            />
            <p className="text-[10.5px] text-muted-foreground">
              Lowercase letter first, then [a-z0-9_-]; ≤64 chars. Leave
              blank to auto-derive.
            </p>
          </div>

          <label className="flex cursor-pointer items-center gap-2 text-xs">
            <input
              type="checkbox"
              checked={noKg}
              onChange={(e) => setNoKg(e.target.checked)}
              disabled={create.isPending}
            />
            <span>
              <span className="font-medium">--no-kg</span>
              <span className="ml-1 text-muted-foreground">
                ablation: decomposer ignores the literature graph
              </span>
            </span>
          </label>
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => setOpen(false)}
            disabled={create.isPending}
          >
            Cancel
          </Button>
          <Button onClick={submit} disabled={create.isPending}>
            {create.isPending ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Queueing…
              </>
            ) : (
              <>
                <Sparkles />
                Decompose
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
