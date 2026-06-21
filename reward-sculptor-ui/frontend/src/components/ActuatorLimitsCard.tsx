import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Icon } from "@/components/rs/icon";
import { qk } from "@/lib/queryKeys";

// §reports: per-motor SPEED-vs-no-load-speed and TORQUE-vs-effort-limit utilization,
// so the actuator-limit enforcement can be visually confirmed (every bar <= 100% =
// within the real motor envelope).

interface Motor {
  name: string;
  speed_p99: number;
  velocity_limit: number;
  velocity_limit_known: boolean;
  speed_util_p99: number;
  torque_p99?: number;
  effort_limit?: number | null;
  torque_util_p99?: number | null;
}

interface ActuatorLimits {
  schema: number;
  ok: boolean;
  has_torque: boolean;
  reason?: string;
  available_iters: number[];
  iter: number | null;
  motors: Motor[];
}

async function fetchActuatorLimits(
  slug: string,
  iter: number | null,
): Promise<ActuatorLimits> {
  const q = iter != null ? `?iter=${iter}` : "";
  const r = await fetch(`/api/projects/${slug}/reports/actuator-limits${q}`);
  if (!r.ok)
    return { schema: 1, ok: false, has_torque: false, available_iters: [], iter: null, motors: [] };
  return (await r.json()) as ActuatorLimits;
}

function utilColor(u: number): string {
  if (u > 1.0) return "var(--st-rose)";
  if (u > 0.9) return "var(--st-amber)";
  return "var(--st-emerald)";
}

interface Row {
  name: string;
  util: number;
  raw: number;
  limit: number;
}

function UtilChart({ title, unit, data }: { title: string; unit: string; data: Row[] }) {
  const sorted = [...data].sort((a, b) => b.util - a.util);
  const maxUtil = Math.max(1.1, ...sorted.map((d) => d.util));
  const worst = sorted[0]?.util ?? 0;
  return (
    <div>
      <div className="rs-flex-between" style={{ marginBottom: 6 }}>
        <span style={{ fontSize: 12.5, fontWeight: 500 }}>{title}</span>
        <span
          className="rs-num"
          style={{ fontSize: 11, color: utilColor(worst), fontWeight: 500 }}
        >
          worst {Math.round(worst * 100)}%
        </span>
      </div>
      <div style={{ height: Math.max(160, sorted.length * 15) }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={sorted}
            layout="vertical"
            margin={{ top: 4, right: 44, bottom: 4, left: 4 }}
          >
            <XAxis
              type="number"
              domain={[0, maxUtil]}
              tickFormatter={(v) => `${Math.round(v * 100)}%`}
              tick={{ fontSize: 10, fill: "var(--body)" }}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={148}
              tick={{ fontSize: 9.5, fill: "var(--body)" }}
              interval={0}
            />
            <Tooltip
              cursor={{ fill: "var(--canvas-soft)" }}
              contentStyle={{
                fontSize: 11,
                padding: "4px 8px",
                border: "1px solid var(--hairline)",
                background: "var(--canvas)",
              }}
              formatter={(v: number, _n, p: { payload?: Row }) => [
                `${Math.round(v * 100)}%  (${p.payload?.raw} / ${p.payload?.limit} ${unit})`,
                "p99 utilization",
              ]}
            />
            <ReferenceLine
              x={1.0}
              stroke="var(--st-rose)"
              strokeDasharray="3 3"
              label={{ value: "limit", fontSize: 10, fill: "var(--st-rose)", position: "right" }}
            />
            <Bar dataKey="util" isAnimationActive={false} radius={[0, 2, 2, 0]}>
              {sorted.map((d, i) => (
                <Cell key={i} fill={utilColor(d.util)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export function ActuatorLimitsCard({ slug }: { slug: string }) {
  const [iter, setIter] = useState<number | null>(null);
  const q = useQuery<ActuatorLimits>({
    queryKey: [...qk.project(slug), "report", "actuator-limits", iter],
    queryFn: () => fetchActuatorLimits(slug, iter),
    staleTime: 10_000,
    // Keep the prior iteration's chart (and the <select>) mounted while a newly
    // selected iter loads — without this the iter-key change drops data to
    // undefined and the whole card unmounts mid-interaction (visible flicker).
    placeholderData: (prev) => prev,
  });
  const data = q.data;
  // Hide the card entirely until a rollout with trajectory data exists.
  if (!data || data.available_iters.length === 0) return null;
  if (!data.ok || data.motors.length === 0) return null;

  const strip = (n: string) => n.replace(/_joint$/, "");
  const speedRows: Row[] = data.motors.map((m) => ({
    name: strip(m.name),
    util: m.speed_util_p99,
    raw: m.speed_p99,
    limit: m.velocity_limit,
  }));
  const torqueRows: Row[] = data.has_torque
    ? data.motors
        .filter((m) => m.effort_limit && m.torque_util_p99 != null)
        .map((m) => ({
          name: strip(m.name),
          util: m.torque_util_p99 as number,
          raw: m.torque_p99 as number,
          limit: m.effort_limit as number,
        }))
    : [];

  return (
    <div className="rs-card" style={{ marginBottom: 22 }}>
      <div className="rs-card-head">
        <div className="rs-card-title">
          <Icon name="trending-up" size={16} />
          Actuator limits
        </div>
        <div className="rs-flex rs-gap-8" style={{ alignItems: "center" }}>
          <span className="rs-sub" style={{ fontSize: 12 }}>
            per-motor p99 vs real limit · iter
          </span>
          <select
            value={data.iter ?? ""}
            onChange={(e) => setIter(Number(e.target.value))}
            style={{ fontSize: 12 }}
          >
            {data.available_iters.map((it) => (
              <option key={it} value={it}>
                {it}
              </option>
            ))}
          </select>
        </div>
      </div>
      <div className="rs-card-pad rs-vgap-8">
        <UtilChart title="Joint speed vs no-load speed" unit="rad/s" data={speedRows} />
        {data.has_torque && torqueRows.length > 0 ? (
          <UtilChart title="Motor torque vs effort limit" unit="N·m" data={torqueRows} />
        ) : (
          <div className="rs-sub" style={{ fontSize: 11.5 }}>
            Torque-vs-limit shows on runs recorded after the joint-torque capture
            (this run has speed only). Every speed bar ≤ 100% confirms the policy
            stays within each motor's real no-load speed.
          </div>
        )}
      </div>
    </div>
  );
}

export default ActuatorLimitsCard;
