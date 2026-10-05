import { useId, useState } from "react";

/* ---------------------------------------------------------------------------
   Number formatting. Every value shown comes from dashboard.json; these only
   decide how many digits of it to print.
--------------------------------------------------------------------------- */
export const fmt = {
  pct: (value, digits = 1) => `${(value * 100).toFixed(digits)}%`,
  pctOf100: (value, digits = 1) => `${Number(value).toFixed(digits)}%`,
  prob: (value, digits = 3) => Number(value).toFixed(digits),
  num: (value, digits = 2) => Number(value).toFixed(digits),
  int: (value) => Number(value).toLocaleString(),
};

/* Shared Recharts furniture - recessive grid and axes, behind the data. */
export const axisProps = {
  stroke: "var(--axis)",
  tick: { fill: "var(--text-secondary)", fontSize: 12 },
  tickLine: false,
};

export const gridProps = {
  stroke: "var(--grid)",
  strokeDasharray: "0",
  vertical: false,
};

/* ---------------------------------------------------------------------------
   Tooltip - one per chart type, all sharing this surface.
--------------------------------------------------------------------------- */
export function TooltipBox({ title, rows }) {
  return (
    <div
      className="rounded-lg px-3 py-2 text-sm shadow-lg"
      style={{
        background: "var(--surface-2)",
        border: "1px solid var(--border)",
        color: "var(--text-primary)",
      }}
    >
      {title ? (
        <div className="mb-1 font-medium" style={{ color: "var(--text-primary)" }}>
          {title}
        </div>
      ) : null}
      {rows.map((row) => (
        <div key={row.label} className="flex items-center gap-2 whitespace-nowrap">
          {row.color ? (
            <span
              aria-hidden="true"
              className="inline-block h-2.5 w-2.5 shrink-0 rounded-sm"
              style={{ background: row.color }}
            />
          ) : null}
          <span style={{ color: "var(--text-secondary)" }}>{row.label}</span>
          <span className="ml-auto tnum font-medium">{row.value}</span>
        </div>
      ))}
    </div>
  );
}

/* ---------------------------------------------------------------------------
   Legend - identity is a swatch; the text stays in text colours.
--------------------------------------------------------------------------- */
export function ChartLegend({ items }) {
  return (
    <ul className="mt-1 mb-2 flex flex-wrap items-center gap-x-4 gap-y-1.5 px-1">
      {items.map((item) => (
        <li
          key={item.label}
          className="flex items-center gap-1.5 text-xs"
          style={{ color: "var(--text-secondary)" }}
        >
          <span
            aria-hidden="true"
            className="inline-block shrink-0 rounded-sm"
            style={{
              width: item.dashed ? 14 : 10,
              height: item.dashed ? 0 : 10,
              background: item.dashed ? "transparent" : item.color,
              borderTop: item.dashed ? `2px dashed ${item.color}` : "none",
            }}
          />
          {item.label}
        </li>
      ))}
    </ul>
  );
}

/* ---------------------------------------------------------------------------
   Segmented control - used for the confusion-matrix and sweep-split toggles.
--------------------------------------------------------------------------- */
export function Segmented({ options, value, onChange, label }) {
  return (
    <div
      role="group"
      aria-label={label}
      className="inline-flex rounded-lg p-0.5"
      style={{ background: "var(--surface-page)", border: "1px solid var(--border)" }}
    >
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            aria-pressed={active}
            className="rounded-md px-2.5 py-1 text-xs font-medium transition-colors"
            style={{
              background: active ? "var(--surface-2)" : "transparent",
              color: active ? "var(--text-primary)" : "var(--text-secondary)",
              boxShadow: active ? "0 1px 2px rgb(0 0 0 / 0.08)" : "none",
            }}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

/* ---------------------------------------------------------------------------
   Stat tile - a number with no chart inside it.
--------------------------------------------------------------------------- */
export function StatTile({ value, label, context, accent }) {
  return (
    <div
      className="rounded-xl p-4"
      style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
    >
      <div
        className="tnum text-3xl leading-tight font-semibold"
        style={{ color: accent || "var(--text-primary)" }}
      >
        {value}
      </div>
      <div className="mt-1 text-sm font-medium" style={{ color: "var(--text-secondary)" }}>
        {label}
      </div>
      <div className="mt-1 text-xs" style={{ color: "var(--text-muted)" }}>
        {context}
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------------------
   Chart card - title, caption, the chart, and the required "Show data" table.
--------------------------------------------------------------------------- */
export function ChartCard({ title, subtitle, caption, controls, table, children, className = "" }) {
  const [open, setOpen] = useState(false);
  const tableId = useId();

  return (
    <section
      className={`rounded-xl p-4 sm:p-5 ${className}`}
      style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>
            {title}
          </h3>
          {subtitle ? (
            <p className="mt-0.5 text-sm tnum" style={{ color: "var(--text-secondary)" }}>
              {subtitle}
            </p>
          ) : null}
        </div>
        {controls ? <div className="shrink-0">{controls}</div> : null}
      </div>

      <div className="mt-3">{children}</div>

      {caption ? (
        <p className="mt-3 text-xs leading-relaxed" style={{ color: "var(--text-muted)" }}>
          {caption}
        </p>
      ) : null}

      {table ? (
        <>
          <button
            type="button"
            onClick={() => setOpen((value) => !value)}
            aria-expanded={open}
            aria-controls={tableId}
            className="mt-3 text-xs font-medium underline underline-offset-2"
            style={{ color: "var(--text-secondary)" }}
          >
            {open ? "Hide data" : "Show data"}
          </button>
          {open ? (
            <div id={tableId} className="mt-2 overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr style={{ borderBottom: "1px solid var(--border)" }}>
                    {table.columns.map((column) => (
                      <th
                        key={column.key}
                        scope="col"
                        className={`py-1.5 pr-3 font-medium ${
                          column.numeric ? "text-right" : "text-left"
                        }`}
                        style={{ color: "var(--text-secondary)" }}
                      >
                        {column.label}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {table.rows.map((row, index) => (
                    <tr key={index} style={{ borderBottom: "1px solid var(--border)" }}>
                      {table.columns.map((column) => (
                        <td
                          key={column.key}
                          className={`py-1.5 pr-3 ${column.numeric ? "tnum text-right" : ""}`}
                          style={{ color: "var(--text-primary)" }}
                        >
                          {row[column.key]}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </>
      ) : null}
    </section>
  );
}

export function SectionIntro({ title, children }) {
  return (
    <div className="mb-1">
      <h2 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
        {title}
      </h2>
      <p className="mt-1 max-w-3xl text-sm leading-relaxed" style={{ color: "var(--text-secondary)" }}>
        {children}
      </p>
    </div>
  );
}
