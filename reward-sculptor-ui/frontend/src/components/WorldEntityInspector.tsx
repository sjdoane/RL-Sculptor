/** Entity list + parameter inspector beside the 3D world viewer.
 *
 *  Lists every semantic entity of the compiled scene (course elements,
 *  objects, zones, terrain, robot); selecting one — here or by clicking
 *  the scene — shows its authored parameters (straight from the
 *  WorldSpec artifact) and any train variations targeting it.
 *
 *  When `onEditVariation` is provided (the promoted-world view, not
 *  draft previews), uniform train variations become editable in place —
 *  each save runs the full re-admission + eval-invariance chain
 *  server-side and promotes a new selection version.
 */
import { useState } from "react";

import { Btn } from "@/components/rs/primitives";
import type { WorldScene, WorldSceneEntity } from "@/lib/types";

export type EditVariationFn = (
  variationId: string, distribution: Record<string, unknown>,
) => void;

const KIND_ORDER = ["course_element", "object", "zone", "terrain", "robot"];
const KIND_LABEL: Record<string, string> = {
  course_element: "Course",
  object: "Objects",
  zone: "Zones",
  terrain: "Terrain",
  robot: "Robot",
};

function fmtValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return String(value);
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

export default function WorldEntityInspector({
  scene, selected, onSelect, onEditVariation, editBusy,
}: {
  scene: WorldScene;
  selected: string | null;
  onSelect: (id: string | null) => void;
  onEditVariation?: EditVariationFn;
  editBusy?: boolean;
}) {
  const groups = KIND_ORDER
    .map((kind) => ({
      kind,
      entities: scene.entities.filter((e) => e.kind === kind),
    }))
    .filter((g) => g.entities.length > 0);
  const active = scene.entities.find((e) => e.id === selected) ?? null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10,
                  minWidth: 0 }}>
      {groups.map((group) => (
        <div key={group.kind}>
          <div className="rs-eyebrow" style={{ marginBottom: 4 }}>
            {KIND_LABEL[group.kind] ?? group.kind}
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
            {group.entities.map((entity) => {
              const isActive = entity.id === selected;
              const hollow = entity.geoms.length === 0 && !entity.sites?.length;
              return (
                <button
                  key={entity.id}
                  onClick={() => onSelect(isActive ? null : entity.id)}
                  className="rs-tag"
                  title={hollow
                    ? `${entity.label} — no geometry (spacing/predicate)`
                    : entity.label}
                  style={{
                    cursor: "pointer",
                    font: "inherit",
                    fontSize: 12,
                    border: isActive
                      ? "1px solid var(--rs-primary)"
                      : "1px solid var(--hairline)",
                    background: isActive
                      ? "color-mix(in srgb, var(--rs-primary) 18%, transparent)"
                      : undefined,
                    opacity: hollow && !isActive ? 0.75 : 1,
                  }}
                >
                  {entity.label}{hollow ? " ◌" : ""}
                </button>
              );
            })}
          </div>
        </div>
      ))}

      {active ? (
        <ActiveEntityDetail entity={active}
                            onEditVariation={onEditVariation}
                            editBusy={editBusy} />
      ) : (
        <p className="rs-hintline" style={{ margin: 0 }}>
          Click an element in the scene (or a chip above) to inspect its
          authored parameters. ◌ = no geometry (gap spacing, predicate
          region).
        </p>
      )}
    </div>
  );
}

function ActiveEntityDetail({
  entity, onEditVariation, editBusy,
}: {
  entity: WorldSceneEntity;
  onEditVariation?: EditVariationFn;
  editBusy?: boolean;
}) {
  const params = Object.entries(entity.params ?? {});
  return (
    <div style={{ border: "1px solid var(--hairline)",
                  borderRadius: "var(--radius-md)", padding: 10 }}>
      <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 6 }}>
        {entity.label}
        <span className="rs-hintline" style={{ marginLeft: 8 }}>
          {entity.kind}{entity.element ? ` · ${entity.element}` : ""}
        </span>
      </div>
      {params.length > 0 ? (
        <div className="rs-kv" style={{ fontSize: 12 }}>
          {params.map(([key, value]) => (
            <div key={key} style={{ display: "contents" }}>
              <span className="mono">{key}</span>
              <span className="mono">{fmtValue(value)}</span>
            </div>
          ))}
        </div>
      ) : (
        <p className="rs-hintline" style={{ margin: 0 }}>No parameters.</p>
      )}
      {entity.variations.length > 0 && (
        <>
          <div className="rs-eyebrow" style={{ margin: "8px 0 4px" }}>
            Train variations
          </div>
          {entity.variations.map((variation) => (
            <VariationRow key={variation.id} variation={variation}
                          onEdit={onEditVariation} busy={editBusy} />
          ))}
        </>
      )}
      {entity.marker && (
        <p className="rs-hintline" style={{ margin: "6px 0 0" }}>
          Shown as a translucent spacing marker — this element has no
          solid geometry.
        </p>
      )}
    </div>
  );
}

/** One train variation: read-only line, or an in-place low/high editor
 *  for uniform distributions when editing is enabled. */
function VariationRow({
  variation, onEdit, busy,
}: {
  variation: WorldSceneEntity["variations"][number];
  onEdit?: EditVariationFn;
  busy?: boolean;
}) {
  const dist = variation.distribution as Record<string, unknown>;
  const isUniform = dist["kind"] === "uniform"
    && typeof dist["low"] === "number" && typeof dist["high"] === "number";
  const [low, setLow] = useState(String(dist["low"] ?? ""));
  const [high, setHigh] = useState(String(dist["high"] ?? ""));
  const dirty = isUniform
    && (Number(low) !== dist["low"] || Number(high) !== dist["high"]);
  const valid = Number.isFinite(Number(low)) && Number.isFinite(Number(high))
    && Number(low) <= Number(high);

  if (!onEdit || !isUniform) {
    return (
      <div className="rs-hintline mono" style={{ fontSize: 11.5 }}>
        {variation.id} ({variation.class}): {JSON.stringify(dist)}
      </div>
    );
  }
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6,
                  padding: "2px 0", flexWrap: "wrap" }}>
      <span className="mono" style={{ fontSize: 11.5 }}>
        {variation.id}
      </span>
      <input className="rs-input mono" type="number" step="any"
             value={low} onChange={(e) => setLow(e.target.value)}
             disabled={busy} aria-label={`${variation.id} low`}
             style={{ width: 76, fontSize: 11.5, padding: "2px 6px" }} />
      <span className="rs-hintline">to</span>
      <input className="rs-input mono" type="number" step="any"
             value={high} onChange={(e) => setHigh(e.target.value)}
             disabled={busy} aria-label={`${variation.id} high`}
             style={{ width: 76, fontSize: 11.5, padding: "2px 6px" }} />
      {dirty && (
        <Btn size="xs" kind="primary" disabled={busy || !valid}
             icon={busy ? "loader" : "check"}
             onClick={() => onEdit(variation.id, {
               ...dist, low: Number(low), high: Number(high),
             })}>
          {busy ? "Promoting…" : "Apply"}
        </Btn>
      )}
    </div>
  );
}
