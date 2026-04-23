import * as React from "react";

import { cn } from "@/lib/utils";

type BadgeVariant = "default" | "outline" | "secondary" | "destructive";

const VARIANT: Record<BadgeVariant, string> = {
  default:
    "border-transparent bg-foreground text-background",
  outline:
    "border border-border text-foreground",
  secondary:
    "border-transparent bg-secondary text-secondary-foreground",
  destructive:
    "border-transparent bg-destructive text-destructive-foreground",
};

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
}

export const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className, variant = "default", ...props }, ref) => {
    return (
      <span
        ref={ref}
        className={cn(
          "inline-flex items-center rounded-md border px-1.5 py-0.5 text-[10px] font-semibold transition-colors",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
          VARIANT[variant],
          className,
        )}
        {...props}
      />
    );
  },
);
Badge.displayName = "Badge";
