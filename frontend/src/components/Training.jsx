import { useMemo } from "react";
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

const SPLIT_COLOR = { train: "var(--series-1)", validation: "var(--series-2)" };

/* One metric, one chart. Never two metrics on one pair of axes. */
function MetricChart({ metric, title, rows, finetuneStart, zeroBased, unit }) {
  const data = useMemo(() => {
    const byEpoch = new Map();
    for (const row of rows) {
      if (row.metric !== metric) continue;
      if (!byEpoch.has(row.epoch)) byEpoch.set(row.epoch, { epoch: row.epoch, phase: row.phase });
      byEpoch.get(row.epoch)[row.split] = row.value;
    }
    return [...byEpoch.values()].sort((a, b) => a.epoch - b.epoch);
  }, [rows, metric]);

  /* A zoomed axis is easier to read here, so the range is stated in the caption
     rather than hidden. Loss starts at zero because its floor is meaningful. */
  const values = data.flatMap((row) => [row.train, row.validation]).filter((v) => v != null);
  const low = Math.floor(Math.min(...values) * 20) / 20;
  const high = Math.ceil(Math.max(...values) * 20) / 20;
  const domain = zeroBased ? [0, high] : [low, 1];
  const rule = finetuneStart - 0.5;

  return (
    <ChartCard
      title={title}
      subtitle={`Deployed run · y-axis ${fmt.num(domain[0], 2)}–${fmt.num(domain[1], 2)}${
        zeroBased ? "" : ", not zero-based"
      }`}
      caption={`Train and validation ${metric === "auc" ? "AUC" : metric} per epoch. The dashed rule marks where fine-tuning begins (epoch ${finetuneStart}). The y-axis runs ${fmt.num(
        domain[0],
        2,
      )}–${fmt.num(domain[1], 2)}${zeroBased ? "." : ", zoomed so the differences are readable — it does not start at zero."}`}
      table={{
        columns: [
          { key: "epoch", label: "Epoch", numeric: true },
          { key: "phase", label: "Phase" },
          { key: "train", label: "Train", numeric: true },
          { key: "validation", label: "Validation", numeric: true },
        ],
        rows: data.map((row) => ({
          epoch: row.epoch,
          phase: row.phase === "finetuned" ? "fine-tuned" : "frozen",
          train: fmt.prob(row.train, 4),
          validation: fmt.prob(row.validation, 4),
        })),
      }}
    >
      <ChartLegend
        items={[
          { label: "train", color: SPLIT_COLOR.train },
          { label: "validation", color: SPLIT_COLOR.validation },
          { label: "fine-tuning starts", color: "var(--reference)", dashed: true },
        ]}
      />
      <ResponsiveContainer width="100%" height={210}>
        <LineChart data={data} margin={{ top: 14, right: 12, bottom: 14, left: -14 }}>
          <CartesianGrid {...gridProps} />
          <XAxis
            dataKey="epoch"
            type="number"
            domain={["dataMin", "dataMax"]}
            label={{
              value: "Epoch",
              position: "insideBottom",
              offset: -10,
              fill: "var(--text-secondary)",
              fontSize: 12,
            }}
            {...axisProps}
          />
          <YAxis
            domain={domain}
            tickFormatter={(value) => fmt.num(value, 2)}
            {...axisProps}
          />
          <Tooltip
            cursor={{ stroke: "var(--reference)", strokeDasharray: "3 3" }}
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <TooltipBox
                  title={`Epoch ${label} · ${
                    payload[0].payload.phase === "finetuned" ? "fine-tuned" : "frozen"
                  }`}
                  rows={payload.map((item) => ({
                    label: item.dataKey,
                    color: item.color,
                    value: `${fmt.prob(item.value, 4)}${unit || ""}`,
                  }))}
                />
              ) : null
            }
          />
          <ReferenceLine
            x={rule}
            stroke="var(--reference)"
            strokeDasharray="5 5"
            strokeWidth={1.5}
            label={{
              value: "fine-tuning starts",
              position: "top",
              fill: "var(--text-primary)",
              fontSize: 10,
            }}
          />
          <Line
            type="monotone"
            dataKey="train"
            stroke={SPLIT_COLOR.train}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, strokeWidth: 2, stroke: "var(--surface-1)" }}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="validation"
            stroke={SPLIT_COLOR.validation}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, strokeWidth: 2, stroke: "var(--surface-1)" }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

export default function Training({ data }) {
  return (
    <div className="space-y-4">
      <SectionIntro title="Training">
        The deployed run (seed {data.meta.seed_deployed}): {data.finetune_start_epoch - 1} epochs with
        the MobileNetV2 base frozen, then fine-tuning from epoch {data.finetune_start_epoch}. Each
        metric gets its own chart — accuracy, loss and AUC are on different scales and never share an
        axis.
      </SectionIntro>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <MetricChart
          metric="accuracy"
          title="Accuracy"
          rows={data.history}
          finetuneStart={data.finetune_start_epoch}
        />
        <MetricChart
          metric="loss"
          title="Loss"
          rows={data.history}
          finetuneStart={data.finetune_start_epoch}
          zeroBased
        />
        <MetricChart
          metric="auc"
          title="AUC"
          rows={data.history}
          finetuneStart={data.finetune_start_epoch}
        />
      </div>
    </div>
  );
}
