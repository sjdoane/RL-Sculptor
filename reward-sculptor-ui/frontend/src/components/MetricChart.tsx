import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export function MetricChart({
  history,
}: {
  history: Array<number | null>;
}) {
  const data = useMemo(
    () =>
      history.map((v, i) => ({
        iter: i,
        metric: typeof v === "number" ? v : null,
      })),
    [history],
  );
  const hasData = data.some((d) => typeof d.metric === "number");
  return (
    <div className="rounded-md border bg-background p-2">
      <div className="mb-1 flex items-center justify-between">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          primary metric
        </span>
        <span className="text-[10px] text-muted-foreground">
          {data.length} iter{data.length === 1 ? "" : "s"}
        </span>
      </div>
      <div className="h-40 w-full">
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
              <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="3 3" />
              <XAxis
                dataKey="iter"
                tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
              />
              <YAxis
                width={38}
                tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                allowDecimals={false}
              />
              <Tooltip
                contentStyle={{
                  fontSize: 11,
                  padding: "4px 8px",
                  border: "1px solid hsl(var(--border))",
                  backgroundColor: "hsl(var(--background))",
                }}
                labelFormatter={(l) => `iter ${l}`}
              />
              <Line
                type="monotone"
                dataKey="metric"
                stroke="hsl(var(--primary))"
                strokeWidth={1.75}
                dot={{ r: 2.5 }}
                activeDot={{ r: 4 }}
                isAnimationActive={false}
                connectNulls
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
            No metric data yet.
          </div>
        )}
      </div>
    </div>
  );
}
