import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import type {
  ButtonHTMLAttributes, CSSProperties, ReactNode,
} from "react";
import { Icon } from "./icon";

// ── Status metadata ─────────────────────────────────────────────────
// One superset map keyed by string, covering ProjectStatus + JobStatus +
// StageStatus + MissionLifecycle. Unknown statuses fall back to "draft".
export interface StatusMeta {
  label: string;
  cls: "slate" | "blue" | "amber" | "emerald" | "rose";
  icon: string;
  spin?: boolean;
}

export const STATUS_META: Record<string, StatusMeta> = {
  draft:      { label: "Draft",      cls: "slate",   icon: "pencil" },
  configured: { label: "Configured", cls: "blue",    icon: "sliders" },
  ready:      { label: "Ready",      cls: "blue",    icon: "check-circle" },
  queued:     { label: "Queued",     cls: "slate",   icon: "clock" },
  running:    { label: "Running",    cls: "amber",   icon: "loader", spin: true },
  training:   { label: "Training",   cls: "amber",   icon: "loader", spin: true },
  completed:  { label: "Completed",  cls: "emerald", icon: "check" },
  "project-history": { label: "Run history", cls: "slate", icon: "history" },
  succeeded:  { label: "Succeeded",  cls: "emerald", icon: "check" },
  errored:    { label: "Errored",    cls: "rose",    icon: "alert-triangle" },
  failed:     { label: "Failed",     cls: "rose",    icon: "alert-triangle" },
  stopped:    { label: "Stopped",    cls: "slate",   icon: "square" },
  halted:     { label: "Halted",     cls: "slate",   icon: "square" },
  pending:    { label: "Pending",    cls: "slate",   icon: "circle" },
  skipped:    { label: "Skipped",    cls: "slate",   icon: "minus" },
  held:       { label: "Held",       cls: "slate",   icon: "minus" },
  // §mission-persistence: replaced by redecomposition children; terminal,
  // non-alarming (artifacts retained) — distinct from "failed".
  superseded: { label: "Superseded", cls: "slate",   icon: "git-branch" },
  // §env-authoring: a promoted world selection — terminal + healthy. NOT
  // "running": lineage entries are immutable history, and a spinner there
  // reads as a stuck load.
  promoted:   { label: "Promoted",   cls: "emerald", icon: "check-circle" },
};

export function Badge({
  status, big, label,
}: { status: string; big?: boolean; label?: string }) {
  const m = STATUS_META[status] ?? STATUS_META.draft;
  return (
    <span className={"rs-badge " + m.cls + (big ? " big" : "")}>
      <Icon name={m.icon} size={12} className={m.spin ? "rs-spin" : undefined} />
      {label || m.label}
    </span>
  );
}

export function AuthorBadge({ author }: { author: string }) {
  const human = author === "human";
  return (
    <span
      className={"rs-abadge " + (human ? "human" : "sculptor")}
      title={human
        ? "Hand-written by a human (manual edit or saved draft)"
        : "Written by the sculptor's diagnose-and-edit loop"}
    >
      <Icon name={human ? "user" : "sparkles"} size={11} />
      {human ? "Human" : "Sculptor"}
    </span>
  );
}

export function Delta({ value, suffix, title }: { value: number | null | undefined; suffix?: string; title?: string }) {
  if (value == null) return <span className="rs-delta flat" title={title}>—</span>;
  const cls = value > 0 ? "up" : value < 0 ? "down" : "flat";
  const sign = value > 0 ? "+" : "";
  return <span className={"rs-delta " + cls} title={title}>{sign}{value.toFixed(1)}{suffix ?? ""}</span>;
}

export function FactChip({
  k, children, icon,
}: { k: string; children: ReactNode; icon?: string }) {
  return (
    <span className="rs-chip">
      {icon && <Icon name={icon} size={13} color="var(--rs-muted)" />}
      <span className="k">{k}</span>
      <span className="v">{children}</span>
    </span>
  );
}

// ── Buttons ─────────────────────────────────────────────────────────
type BtnKind = "ghost" | "primary" | "quiet" | "danger";
interface BtnProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  kind?: BtnKind;
  size?: "sm" | "xs";
  icon?: string;
  iconRight?: string;
}

export function Btn({
  kind = "ghost", size, icon, iconRight, children, className, ...rest
}: BtnProps) {
  const cls = [
    "rs-btn",
    "rs-btn-" + kind,
    size === "sm" ? "rs-btn-sm" : size === "xs" ? "rs-btn-xs" : "",
    className ?? "",
  ].join(" ").trim();
  const isz = size === "xs" ? 13 : size === "sm" ? 14 : 15;
  return (
    <button className={cls} {...rest}>
      {icon && <Icon name={icon} size={isz} />}
      {children}
      {iconRight && <Icon name={iconRight} size={isz} />}
    </button>
  );
}

interface IconBtnProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  icon: string;
  label: string;
  on?: boolean;
  size?: number;
}

export function IconBtn({ icon, label, on, size = 16, className, ...rest }: IconBtnProps) {
  return (
    <button
      className={"rs-iconbtn" + (on ? " on" : "") + (className ? " " + className : "")}
      aria-label={label}
      title={label}
      {...rest}
    >
      <Icon name={icon} size={size} />
    </button>
  );
}

// ── Sparkline (null-safe for real metric histories) ─────────────────
export function Sparkline({
  data, w = 80, h = 24, color = "var(--st-emerald)", strokeWidth = 1.5,
}: {
  data: Array<number | null> | null | undefined;
  w?: number; h?: number; color?: string; strokeWidth?: number;
}) {
  const pts0 = (data ?? []).filter((v): v is number => typeof v === "number");
  if (pts0.length < 2) {
    return (
      <svg width={w} height={h} aria-hidden>
        <line x1="0" y1={h - 3} x2={w} y2={h - 3} stroke="var(--hairline-strong)" strokeWidth="1.5" strokeDasharray="3 3" />
      </svg>
    );
  }
  const min = Math.min(...pts0), max = Math.max(...pts0), span = max - min || 1;
  const pts = pts0.map((v, i) => [(i / (pts0.length - 1)) * (w - 2) + 1, h - 3 - ((v - min) / span) * (h - 6)]);
  const d = pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
  return (
    <svg width={w} height={h} aria-hidden>
      <path d={d} fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={pts[pts.length - 1][0]} cy={pts[pts.length - 1][1]} r="2.2" fill={color} />
    </svg>
  );
}

// ── Metric line chart (SVG; replaces recharts) ──────────────────────
export function MetricChart({
  data, h = 150, live, label = "Mean reward per iteration", decimals = 1,
}: {
  data: Array<number | null> | null | undefined;
  h?: number; live?: boolean;
  // §Ship 35: label/decimals so the chart can render objective fitness
  // (0-1, 2 decimals) as the primary series, not just the reward metric.
  label?: string; decimals?: number;
}) {
  const vals = (data ?? []).filter((v): v is number => typeof v === "number");
  if (!vals.length) {
    return (
      <div className="rs-empty" style={{ padding: 28 }}>
        <span className="rs-sub">No metric data yet</span>
      </div>
    );
  }
  const w = 280, pad = 22;
  const min = Math.min(...vals) * 0.96, max = Math.max(...vals) * 1.02, span = max - min || 1;
  const X = (i: number) => pad + (i / Math.max(1, vals.length - 1)) * (w - pad - 8);
  const Y = (v: number) => h - pad - ((v - min) / span) * (h - pad - 12);
  const pts = vals.map((v, i) => [X(i), Y(v)]);
  const line = pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
  const area = line + ` L ${X(vals.length - 1)} ${h - pad} L ${pad} ${h - pad} Z`;
  return (
    <svg className="rs-chart" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label}>
      {[0, 0.5, 1].map((t) => (
        <line key={t} x1={pad} x2={w - 8} y1={pad + t * (h - pad - 12)} y2={pad + t * (h - pad - 12)} stroke="var(--hairline)" strokeWidth="1" />
      ))}
      <path d={area} fill="rgba(245,78,0,0.08)" />
      <path d={line} fill="none" stroke="var(--rs-primary)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      {pts.map((p, i) => (
        <circle key={i} cx={p[0]} cy={p[1]} r={i === pts.length - 1 && live ? 3.5 : 2.4} fill="var(--rs-primary)" />
      ))}
      <text x={pad} y={12} fontSize="9" fill="var(--rs-muted)" fontFamily="var(--font-mono)">{max.toFixed(decimals)}</text>
      <text x={pad} y={h - pad + 11} fontSize="9" fill="var(--rs-muted)" fontFamily="var(--font-mono)">iter 0</text>
      <text x={w - 8} y={h - pad + 11} fontSize="9" fill="var(--rs-muted)" fontFamily="var(--font-mono)" textAnchor="end">iter {vals.length - 1}</text>
    </svg>
  );
}

// ── Segmented control ───────────────────────────────────────────────
export interface SegOption { value: string; label: string; icon?: string }
export function Segmented({
  value, onChange, options,
}: { value: string; onChange: (v: string) => void; options: SegOption[] }) {
  return (
    <div className="rs-seg" role="tablist">
      {options.map((o) => (
        <button
          key={o.value}
          role="tab"
          aria-selected={value === o.value}
          className={value === o.value ? "on" : ""}
          onClick={() => onChange(o.value)}
        >
          {o.icon && <Icon name={o.icon} size={14} />}{o.label}
        </button>
      ))}
    </div>
  );
}

// ── Toggle switch ───────────────────────────────────────────────────
export function Toggle({
  on, onChange, label,
}: { on: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button
      className={"rs-switch" + (on ? " on" : "")}
      role="switch"
      aria-checked={on}
      aria-label={label}
      onClick={() => onChange(!on)}
    />
  );
}

// ── Form field ──────────────────────────────────────────────────────
export function Field({
  label, hint, children, htmlFor,
}: { label?: ReactNode; hint?: ReactNode; children: ReactNode; htmlFor?: string }) {
  return (
    <div className="rs-field">
      {label && (
        <label htmlFor={htmlFor}>
          {label}{hint && <span className="hint">{hint}</span>}
        </label>
      )}
      {children}
    </div>
  );
}

export function Check({
  on, onChange, title, desc,
}: { on: boolean; onChange: (v: boolean) => void; title: ReactNode; desc?: ReactNode }) {
  return (
    <button className={"rs-check" + (on ? " on" : "")} onClick={() => onChange(!on)} aria-pressed={on} type="button">
      <span className="rs-check-box">{on && <Icon name="check" size={12} />}</span>
      <span style={{ textAlign: "left" }}>
        <span style={{ fontSize: 13.5, fontWeight: 500, color: "var(--ink)", display: "block" }}>{title}</span>
        {desc && <span style={{ fontSize: 12, color: "var(--rs-muted)", display: "block", marginTop: 2 }}>{desc}</span>}
      </span>
    </button>
  );
}

// Toggle-row: a labelled switch (title + optional desc on the left, a
// Toggle on the right). Used across the launch/config dialogs for boolean
// flags. Unlike Check, the row body is not a click target — only the switch
// toggles, so adjacent help text is selectable.
export function ToggleRow({
  on, onChange, title, desc, label,
}: { on: boolean; onChange: (v: boolean) => void; title: ReactNode; desc?: ReactNode; label: string }) {
  return (
    <div className="rs-toggle-row">
      <span style={{ flex: 1, minWidth: 0 }}>
        <span className="ttl">{title}</span>
        {desc && <span className="desc">{desc}</span>}
      </span>
      <Toggle on={on} onChange={onChange} label={label} />
    </div>
  );
}

// ── Banner / empty / skeleton ───────────────────────────────────────
export function Banner({
  kind = "info", icon, children, action,
}: { kind?: "info" | "warn" | "err"; icon?: string; children: ReactNode; action?: ReactNode }) {
  return (
    <div className={"rs-banner " + kind}>
      <Icon name={icon || (kind === "err" ? "alert-triangle" : kind === "warn" ? "alert-circle" : "info")} size={17} />
      <span className="rs-grow">{children}</span>
      {action}
    </div>
  );
}

export function EmptyState({
  icon = "info", title, sub, action,
}: { icon?: string; title: ReactNode; sub?: ReactNode; action?: ReactNode }) {
  return (
    <div className="rs-empty">
      <span className="ic"><Icon name={icon} size={22} color="var(--rs-muted)" /></span>
      <div className="rs-h3" style={{ fontWeight: 500, color: "var(--ink)" }}>{title}</div>
      {sub && <div className="rs-sub" style={{ maxWidth: 360 }}>{sub}</div>}
      {action}
    </div>
  );
}

export function Skel({
  w, h = 14, r, style,
}: { w?: number | string; h?: number | string; r?: number; style?: CSSProperties }) {
  return <div className="rs-skel" style={{ width: w, height: h, borderRadius: r, ...style }} />;
}

// Stack of mounted Modals so only the TOPMOST handles Escape — required
// because Modals can nest (e.g. RunMission inside MissionDetail) and each
// registers a global keydown listener.
const MODAL_STACK: symbol[] = [];

// ── Modal (rs-scrim/rs-modal + a11y: Esc, initial focus, focus restore) ──
export function Modal({
  title, subtitle, icon, wide, full, flush, onClose, children, footer,
}: {
  title: string;
  subtitle?: ReactNode;
  icon?: string;
  wide?: boolean;
  /** Near-fullscreen (e.g. the KG graph): rs-modal flexes column-wise so a
   *  flex:1 body child fills the height. */
  full?: boolean;
  /** Remove body padding (for edge-to-edge content like an iframe). */
  flush?: boolean;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  // Scrim clicks only close when the pointer went DOWN on the scrim itself.
  // Dragging a text selection out of the dialog and releasing on the scrim
  // fires a click on the scrim (the common ancestor of down + up targets),
  // which used to close every dialog app-wide mid-selection.
  const scrimPointerDown = useRef(false);
  // Latest onClose via ref so the mount-once effect never re-binds (avoids
  // focus-thrash when the parent passes an inline arrow for onClose).
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  useEffect(() => {
    const id = Symbol("modal");
    MODAL_STACK.push(id);
    const prev = document.activeElement as HTMLElement | null;
    const FOCUSABLE =
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';
    const onKey = (e: KeyboardEvent) => {
      // Only the topmost modal reacts (modals can nest).
      if (MODAL_STACK[MODAL_STACK.length - 1] !== id) return;
      if (e.key === "Escape") {
        e.stopPropagation();
        onCloseRef.current();
        return;
      }
      // Focus trap — Radix gave us this for free; the bespoke Modal must
      // keep Tab inside the dialog (WCAG 2.1 dialog pattern).
      if (e.key === "Tab") {
        const root = ref.current;
        if (!root) return;
        const items = Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE))
          .filter((el) => el.offsetParent !== null || el === document.activeElement);
        if (items.length === 0) { e.preventDefault(); root.focus(); return; }
        const first = items[0];
        const last = items[items.length - 1];
        const active = document.activeElement as HTMLElement | null;
        if (e.shiftKey) {
          if (active === first || !root.contains(active)) { e.preventDefault(); last.focus(); }
        } else if (active === last || !root.contains(active)) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    window.addEventListener("keydown", onKey);
    const focusable = ref.current?.querySelector<HTMLElement>(FOCUSABLE);
    (focusable ?? ref.current)?.focus();
    return () => {
      window.removeEventListener("keydown", onKey);
      const i = MODAL_STACK.indexOf(id);
      if (i >= 0) MODAL_STACK.splice(i, 1);
      prev?.focus?.();
    };
  }, []);
  // Portal to <body>. `.rs-modal` animates `transform` with fill-mode `both`,
  // so it keeps a non-none transform forever — which makes it a containing
  // block for `position: fixed` descendants. A modal rendered INSIDE another
  // modal therefore had its `position: fixed` scrim resolve against the parent
  // dialog's box instead of the viewport, clipping the nested dialog's header
  // and footer (seen on the reference picker's compose dialog). The portal
  // takes the scrim out of that subtree so `fixed` means fixed again.
  // MODAL_STACK still orders Escape/focus-trap correctly across nesting.
  return createPortal(
    <div
      className="rs-scrim"
      onPointerDown={(e) => { scrimPointerDown.current = e.target === e.currentTarget; }}
      onClick={(e) => {
        const armed = scrimPointerDown.current;
        scrimPointerDown.current = false;
        if (armed && e.target === e.currentTarget) onClose();
      }}
    >
      <div
        ref={ref}
        className={"rs-modal" + (wide ? " wide" : "") + (full ? " full" : "")}
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
      >
        <div className="rs-modal-head">
          <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
            {icon && <span style={{ color: "var(--rs-primary)" }}><Icon name={icon} size={18} /></span>}
            <div>
              <div className="rs-h3" style={{ fontSize: 17 }}>{title}</div>
              {subtitle && <div className="rs-sub" style={{ fontSize: 13, marginTop: 2 }}>{subtitle}</div>}
            </div>
          </div>
          <button className="rs-modal-x" onClick={onClose} aria-label="Close"><Icon name="x" size={18} /></button>
        </div>
        <div className={"rs-modal-body" + (flush ? " flush" : "")}>{children}</div>
        {footer && <div className="rs-modal-foot">{footer}</div>}
      </div>
    </div>,
    document.body,
  );
}
