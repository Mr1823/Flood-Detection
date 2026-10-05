import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { axisProps, ChartCard, ChartLegend, fmt, gridProps, SectionIntro, TooltipBox } from "./ui.jsx";
import ThresholdExplorer from "./ThresholdExplorer.jsx";

const LINES = [
  { key: "precision", color: "var(--series-1)" },
  { key: "recall", color: "var(--series-2)" },
  { key: "f1", color: "var(--series-3)" },
];

function SweepChart({ split, rows, threshold, caption, title, subtitle }) {
  const data = useMemo(
    () =>
      rows
        .filter((row) => row.split === split)
        .sort((a, b) => a.threshold - b.threshold)
        .map((row) => ({
          threshold: row.threshold,
          precision: row.precision,
          recall: row.recall,
          f1: row.f1,
        })),
    [rows, split],
  );

  return (
    <ChartCard
      title={title}
      subtitle={subtitle}
      caption={caption}
      table={{
        columns: [
          { key: "threshold", label: "Threshold", numeric: true },
          { key: "precision", label: "Precision", numeric: true },
          { key: "recall", label: "Recall", numeric: true },
          { key: "f1", label: "F1", numeric: true },
        ],
        rows: data.map((row) => ({
          threshold: fmt.num(row.threshold, 2),
          precision: fmt.prob(row.precision, 4),
          recall: fmt.prob(row.recall, 4),
          f1: fmt.prob(row.f1, 4),
        })),
      }}
    >
      <ChartLegend
        items={[
          ...LINES.map((line) => ({ label: line.key, color: line.color })),
          { label: `chosen ${fmt.num(threshold, 2)}`, color: "var(--reference)", dashed: true },
        ]}
      />
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} margin={{ top: 14, right: 12, bottom: 16, left: -14 }}>
          <CartesianGrid {...gridProps} />
          <XAxis
            dataKey="threshold"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(value) => fmt.num(value, 2)}
            label={{
              value: "Decision threshold",
              position: "insideBottom",
              offset: -10,
              fill: "var(--text-secondary)",
              fontSize: 12,
            }}
            {...axisProps}
          />
          <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} {...axisProps} />
          <Tooltip
            cursor={{ stroke: "var(--reference)", strokeDasharray: "3 3" }}
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <TooltipBox
                  title={`Threshold ${fmt.num(label, 2)}`}
                  rows={payload.map((item) => ({
                    label: item.dataKey,
                    color: item.color,
                    value: fmt.prob(item.value, 4),
                  }))}
                />
              ) : null
            }
          />
          <ReferenceLine
            x={threshold}
            stroke="var(--reference)"
            strokeDasharray="5 5"
            strokeWidth={1.5}
            label={{
              value: `chosen ${fmt.num(threshold, 2)}`,
              position: "top",
              fill: "var(--text-primary)",
              fontSize: 11,
            }}
          />
          {LINES.map((line) => (
            <Line
              key={line.key}
              type="monotone"
              dataKey={line.key}
              stroke={line.color}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, strokeWidth: 2, stroke: "var(--surface-1)" }}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

export default function Threshold({ data }) {
  const [showTest, setShowTest] = useState(false);
  const threshold = data.meta.threshold;
  const chosen = data.threshold_sweep.find(
    (row) => row.split === "validation" && Math.abs(row.threshold - threshold) < 1e-9,
  );

  return (
    <div className="space-y-4">
      <SectionIntro title="Threshold">
        {data.meta.threshold_rationale}
      </SectionIntro>

      <ThresholdExplorer data={data} />

      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => setShowTest((value) => !value)}
          aria-pressed={showTest}
          className="rounded-lg px-3 py-1.5 text-xs font-medium"
          style={{
            background: showTest ? "var(--surface-2)" : "var(--surface-1)",
            border: "1px solid var(--border)",
            color: "var(--text-primary)",
          }}
        >
          {showTest ? "Hide the test sweep" : "Show the test sweep beside it"}
        </button>
        <span className="text-xs" style={{ color: "var(--text-muted)" }}>
          Test is shown for reporting only.
        </span>
      </div>

      <div className={`grid grid-cols-1 gap-4 ${showTest ? "lg:grid-cols-2" : ""}`}>
        <SweepChart
          split="validation"
          rows={data.threshold_sweep}
          threshold={threshold}
          title="Validation sweep — the threshold is chosen here"
          subtitle={
            chosen
              ? `At ${fmt.num(threshold, 2)}: precision ${fmt.prob(
                  chosen.precision,
                  3,
                )}, recall ${fmt.prob(chosen.recall, 3)}, F1 ${fmt.prob(chosen.f1, 3)}`
              : undefined
          }
          caption={`Precision, recall and F1 across every swept threshold on the validation split. The rule that picked ${fmt.num(
            threshold,
            2,
          )} is stated above; the vertical line is where it landed.`}
        />
        {showTest ? (
          <SweepChart
            split="test"
            rows={data.threshold_sweep}
            threshold={threshold}
            title="Test sweep — reported only, never used to choose"
            subtitle="Computed after the decision was already made on validation"
            caption="This sweep exists so the test behaviour is visible, not so a threshold can be picked from it. Choosing the threshold here and then reporting these same test numbers would be selection on the test set, and the result would be optimistic."
          />
        ) : null}
      </div>
    </div>
  );
}
