import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { axisProps, ChartCard, ChartLegend, fmt, gridProps, SectionIntro, TooltipBox } from "./ui.jsx";

/* (j) Dataset validation - what a content-free model can do with each dataset */
function DatasetValidation({ audit }) {
  const data = [
    {
      name: "AIDER (balanced sample)",
      value: audit.aider.size_leak_balanced_accuracy,
      n: audit.aider.n_images,
      target: audit.aider.target,
      color: "var(--series-2)",
    },
    {
      name: "Rejected dataset",
      value: audit.rejected.size_leak_balanced_accuracy,
      n: audit.rejected.n_images,
      target: audit.rejected.target,
      color: "var(--series-4)",
    },
    {
      name: "The split trained on",
      value: audit.split.size_leak_balanced_accuracy,
      n: audit.split.n_images,
      target: audit.split.target,
      color: "var(--series-1)",
    },
  ];
  const chance = audit.split.chance_baseline;

  return (
    <ChartCard
      title="What image dimensions alone can predict"
      subtitle="Balanced accuracy of a classifier given only each image's width and height"
      caption={`${audit.note} A dataset where dimensions alone beat chance carries a shortcut: a model can learn "this size means flood" without ever looking at water. The split actually trained on is the lowest of the three, but none of them is at chance, which is why the dimension check is reported rather than waved away.`}
      table={{
        columns: [
          { key: "name", label: "Dataset" },
          { key: "target", label: "Path" },
          { key: "n", label: "Images", numeric: true },
          { key: "value", label: "Balanced accuracy", numeric: true },
          { key: "plain", label: "Plain accuracy", numeric: true },
          { key: "blocks", label: "In single-class size blocks", numeric: true },
        ],
        rows: [
          { key: "aider", label: "AIDER (balanced sample)", source: audit.aider },
          { key: "rejected", label: "Rejected dataset", source: audit.rejected },
          { key: "split", label: "The split trained on", source: audit.split },
        ].map(({ label, source }) => ({
          name: label,
          target: source.target,
          n: fmt.int(source.n_images),
          value: fmt.pctOf100(source.size_leak_balanced_accuracy, 2),
          plain: fmt.pctOf100(source.size_leak_plain_accuracy, 2),
          blocks: fmt.pctOf100(source.single_class_size_block_pct, 1),
        })),
      }}
    >
      <ChartLegend
        items={[{ label: `chance baseline (${fmt.pctOf100(chance, 0)})`, color: "var(--reference)", dashed: true }]}
      />
      <ResponsiveContainer width="100%" height={240}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 52, bottom: 16, left: 8 }}
          barCategoryGap={10}
        >
          <CartesianGrid {...gridProps} horizontal={false} vertical />
          <XAxis
            type="number"
            domain={[0, 100]}
            ticks={[0, 25, 50, 75, 100]}
            tickFormatter={(value) => `${value}%`}
            label={{
              value: "Balanced accuracy from dimensions alone",
              position: "insideBottom",
              offset: -10,
              fill: "var(--text-secondary)",
              fontSize: 12,
            }}
            {...axisProps}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={170}
            tick={{ fill: "var(--text-secondary)", fontSize: 12 }}
            tickLine={false}
            stroke="var(--axis)"
          />
          <Tooltip
            cursor={{ fill: "var(--grid)", fillOpacity: 0.4 }}
            content={({ active, payload }) =>
              active && payload?.length ? (
                <TooltipBox
                  title={payload[0].payload.name}
                  rows={[
                    {
                      label: "From dimensions alone",
                      color: payload[0].payload.color,
                      value: fmt.pctOf100(payload[0].value, 2),
                    },
                    { label: "Chance", value: fmt.pctOf100(chance, 0) },
                    { label: "Images audited", value: fmt.int(payload[0].payload.n) },
                  ]}
                />
              ) : null
            }
          />
          <ReferenceLine
            x={chance}
            stroke="var(--reference)"
            strokeDasharray="5 5"
            strokeWidth={1.5}
            label={{
              value: "chance",
              position: "top",
              fill: "var(--text-primary)",
              fontSize: 11,
            }}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={34}>
            {data.map((row) => (
              <Cell key={row.name} fill={row.color} />
            ))}
            <LabelList
              dataKey="value"
              position="right"
              offset={8}
              formatter={(value) => fmt.pctOf100(value, 1)}
              style={{ fill: "var(--text-primary)", fontSize: 12, fontVariantNumeric: "tabular-nums" }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

/* (k) Split composition - stacked bar of train/val/test by class */
function SplitComposition({ meta }) {
  const [negative, positive] = meta.class_names;
  const order = ["train", "val", "test"];
  const label = { train: "train", val: "validation", test: "test" };
  const data = order
    .filter((split) => meta.class_counts[split])
    .map((split) => ({
      split: label[split],
      [negative]: meta.class_counts[split][negative],
      [positive]: meta.class_counts[split][positive],
      total: meta.class_counts[split][negative] + meta.class_counts[split][positive],
    }));

  return (
    <ChartCard
      title="Split composition"
      subtitle={`${fmt.int(meta.n_train + meta.n_val + meta.n_test)} images in total`}
      caption={`Counts per split, stacked by class. Floods are the minority everywhere, which is the reason the threshold is tuned rather than left at 0.50 and the reason the precision–recall baseline sits at the flood share.`}
      table={{
        columns: [
          { key: "split", label: "Split" },
          { key: "negative", label: negative, numeric: true },
          { key: "positive", label: positive, numeric: true },
          { key: "total", label: "Total", numeric: true },
          { key: "share", label: `${positive} share`, numeric: true },
        ],
        rows: data.map((row) => ({
          split: row.split,
          negative: fmt.int(row[negative]),
          positive: fmt.int(row[positive]),
          total: fmt.int(row.total),
          share: fmt.pct(row[positive] / row.total),
        })),
      }}
    >
      <ChartLegend
        items={[
          { label: negative, color: "var(--series-1)" },
          { label: positive, color: "var(--series-2)" },
        ]}
      />
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -14 }}>
          <CartesianGrid {...gridProps} />
          <XAxis dataKey="split" {...axisProps} />
          <YAxis {...axisProps} />
          <Tooltip
            cursor={{ fill: "var(--grid)", fillOpacity: 0.4 }}
            content={({ active, payload, label: name }) =>
              active && payload?.length ? (
                <TooltipBox
                  title={name}
                  rows={[
                    ...payload.map((item) => ({
                      label: item.dataKey,
                      color: item.color,
                      value: `${fmt.int(item.value)} images`,
                    })),
                    { label: "Total", value: fmt.int(payload[0].payload.total) },
                  ]}
                />
              ) : null
            }
          />
          {/* 2px surface gap between stacked segments */}
          <Bar
            dataKey={negative}
            stackId="split"
            fill="var(--series-1)"
            maxBarSize={72}
            stroke="var(--surface-1)"
            strokeWidth={2}
          />
          <Bar
            dataKey={positive}
            stackId="split"
            fill="var(--series-2)"
            radius={[4, 4, 0, 0]}
            maxBarSize={72}
            stroke="var(--surface-1)"
            strokeWidth={2}
          />
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

export default function Dataset({ data }) {
  return (
    <div className="space-y-4">
      <SectionIntro title="Dataset">
        Before any result is believable, the data has to be checked for shortcuts a model could take
        instead of learning the task. Both checks here are reported, including where they fail.
      </SectionIntro>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <DatasetValidation audit={data.audit} />
        <SplitComposition meta={data.meta} />
      </div>
    </div>
  );
}
