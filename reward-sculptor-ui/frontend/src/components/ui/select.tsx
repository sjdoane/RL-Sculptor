/** Native <select> with shadcn-style chrome. Radix Select is nicer for
 *  keyboard nav but has a lot of boilerplate; for v1's small
 *  enum-picker needs, native is enough. */
import * as React from "react";
import { cn } from "@/lib/utils";

export const Select = React.forwardRef<
  HTMLSelectElement,
  React.SelectHTMLAttributes<HTMLSelectElement>
>(({ className, children, ...props }, ref) => (
  <select
    ref={ref}
    className={cn(
      "flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm " +
        "transition-colors focus-visible:outline-none focus-visible:ring-2 " +
        "focus-visible:ring-ring focus-visible:ring-offset-1 disabled:cursor-not-allowed disabled:opacity-50",
      className,
    )}
    {...props}
  >
    {children}
  </select>
));
Select.displayName = "Select";
