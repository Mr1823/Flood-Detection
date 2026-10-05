import { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  axisProps,
  ChartCard,
  ChartLegend,
  fmt,
  gridProps,
  Segmented,
  SectionIntro,
  TooltipBox,
} from "./ui.jsx";

const VARIANT_LABEL = { frozen: "frozen", finetuned: "fine-tuned" };
const VARIANT_COLOR = { frozen: "var(--series-1)", finetuned: "var(--series-2)" };

/* --------------------------------------------------------------------------
   (a) Model comparison - grouped bar, error bars from the 3-seed std
-------------------------------------------------------------------------- */
function ModelComparison({ comparison, seeds, note }) {
  const data = useMemo(() => {
    const byMetric = new Map();
    for (const row of comparison) {
      if (!byMetric.has(row.metric)) byMetric.set(row.metric, { metric: row.metric });
      const entry = byMetric.get(row.metric);
      entry[row.variant] = row.mean;
      entry[`${row.variant}_std`] = row.std;
    }
    return [...byMetric.values()];
  }, [comparison]);

  return (
    <ChartCard
      title="Frozen vs fine-tuned"
      subtitle={`Test set, mean ± std over ${seeds.length} seeds (${seeds.join(", ")})`}
      caption={`Bars start at zero. Error bars are ±1 sample standard deviation across the ${seeds.length} seeds. ${note}`}
      table={{
        columns: [
          { key: "metric", label: "Metric" },
          { key: "frozen", label: "Frozen", numeric: true },
          { key: "finetuned", label: "Fine-tuned", numeric: true },
        ],
        rows: data.map((row) => ({
          metric: row.metric,
          frozen: `${fmt.prob(row.frozen, 4)} ± ${fmt.prob(row.frozen_std, 4)}`,
          finetuned: `${fmt.prob(row.finetuned, 4)} ± ${fmt.prob(row.finetuned_std, 4)}`,
        })),
      }}
    >
      <ChartLegend
        items={[
          { label: "frozen feature extractor", color: VARIANT_COLOR.frozen },
          { label: "fine-tuned", color: VARIANT_COLOR.finetuned },
        ]}
      />
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: -16 }} barGap={2}>
          <CartesianGrid {...gridProps} />
          <XAxis dataKey="metric" {...axisProps} />
          <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} {...axisProps} />
          <Tooltip
            cursor={{ fill: "var(--grid)", fillOpacity: 0.4 }}
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <TooltipBox
                  title={label}
                  rows={payload.map((item) => ({
                    label: VARIANT_LABEL[item.dataKey],
                    color: item.color,
                    value: `${fmt.prob(item.value, 4)} ± ${fmt.prob(
                      item.payload[`${item.dataKey}_std`],
                      4,
                    )}`,
                  }))}
                />
              ) : null
            }
          />
          <Bar dataKey="frozen" fill={VARIANT_COLOR.frozen} radius={[4, 4, 0, 0]} maxBarSize={46}>
            <ErrorBar dataKey="frozen_std" width={5} strokeWidth={1.5} stroke="var(--text-secondary)" />
          </Bar>
          <Bar
            dataKey="finetuned"
            fill={VARIANT_COLOR.finetuned}
            radius={[4, 4, 0, 0]}
            maxBarSize={46}
          >
            <ErrorBar
              dataKey="finetuned_std"
              width={5}
              strokeWidth={1.5}
              stroke="var(--text-secondary)"
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

/* --------------------------------------------------------------------------
   Operating point - one filled dot, direct-labelled, with its own tooltip
   carrying precision and recall (not just the threshold).
-------------------------------------------------------------------------- */
function operatingDot({ cx, cy }, title) {
  return (
    <g>
      {/* hit target larger than the mark */}
      <circle cx={cx} cy={cy} r={14} fill="transparent">
        <title>{title}</title>
      </circle>
      <circle
        cx={cx}
        cy={cy}
        r={5.5}
        fill="var(--text-primary)"
        stroke="var(--surface-1)"
        strokeWidth={2}
        pointerEvents="none"
      />
    </g>
  );
}

/* --------------------------------------------------------------------------
   (b) ROC curve
-------------------------------------------------------------------------- */
function RocCurve({ roc, rocAuc, operating }) {
  const dotTitle =
    `Tuned threshold ${fmt.num(operating.threshold, 2)} — ` +
    `recall ${fmt.pct(operating.recall)}, precision ${fmt.pct(operating.precision)}, ` +
    `false positive rate ${fmt.pct(operating.fpr)}`;

  return (
    <ChartCard
      title={`ROC curve — test set (AUC = ${fmt.prob(rocAuc, 4)})`}
      subtitle="True positive rate against false positive rate, across every threshold"
      caption={`The dot is the deployed operating point at threshold ${fmt.num(
        operating.threshold,
        2,
      )}; hover it for its precision and recall. The dashed diagonal is chance.`}
      table={{
        columns: [
          { key: "fpr", label: "False positive rate", numeric: true },
          { key: "tpr", label: "True positive rate", numeric: true },
          { key: "threshold", label: "Threshold", numeric: true },
        ],
        rows: roc.map((point) => ({
          fpr: fmt.prob(point.fpr, 4),
          tpr: fmt.prob(point.tpr, 4),
          threshold: point.threshold === null ? "—" : fmt.prob(point.threshold, 4),
        })),
      }}
    >
      <ChartLegend
        items={[
          { label: "MobileNetV2", color: "var(--series-1)" },
          { label: "chance (AUC = 0.5)", color: "var(--reference)", dashed: true },
          { label: "tuned threshold", color: "var(--text-primary)" },
        ]}
      />
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={roc} margin={{ top: 8, right: 16, bottom: 18, left: -12 }}>
          <CartesianGrid {...gridProps} vertical />
          <XAxis
            dataKey="fpr"
            type="number"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            label={{
              value: "False positive rate (1 − specificity)",
              position: "insideBottom",
              offset: -12,
              fill: "var(--text-secondary)",
              fontSize: 12,
            }}
            {...axisProps}
          />
          <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} {...axisProps} />
          <Tooltip
            cursor={{ stroke: "var(--reference)", strokeDasharray: "3 3" }}
            content={({ active, payload }) =>
              active && payload?.length ? (
                <TooltipBox
                  title={`False positive rate ${fmt.prob(payload[0].payload.fpr, 3)}`}
                  rows={[
                    {
                      label: "True positive rate",
                      color: "var(--series-1)",
                      value: fmt.prob(payload[0].payload.tpr, 3),
                    },
                    {
                      label: "Threshold",
                      value:
                        payload[0].payload.threshold === null
                          ? "—"
                          : fmt.prob(payload[0].payload.threshold, 3),
                    },
                  ]}
                />
              ) : null
            }
          />
          <ReferenceLine
            segment={[
              { x: 0, y: 0 },
              { x: 1, y: 1 },
            ]}
            stroke="var(--reference)"
            strokeDasharray="5 5"
            strokeWidth={1.5}
            ifOverflow="extendDomain"
          />
          <Line
            type="linear"
            dataKey="tpr"
            stroke="var(--series-1)"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, strokeWidth: 2, stroke: "var(--surface-1)" }}
            isAnimationActive={false}
          />
          <ReferenceDot
            x={operating.fpr}
            y={operating.recall}
            shape={(props) => operatingDot(props, dotTitle)}
            label={{
              value: `threshold ${fmt.num(operating.threshold, 2)}`,
              position: "right",
              offset: 12,
              fill: "var(--text-primary)",
              fontSize: 11,
            }}
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

/* --------------------------------------------------------------------------
   (c) Precision-recall curve - chance line at the positive-class share
-------------------------------------------------------------------------- */
function PrCurve({ pr, averagePrecision, positiveShare, operating }) {
  const sorted = useMemo(() => [...pr].sort((a, b) => a.recall - b.recall), [pr]);
  const dotTitle =
    `Tuned threshold ${fmt.num(operating.threshold, 2)} — ` +
    `recall ${fmt.pct(operating.recall)}, precision ${fmt.pct(operating.precision)}`;

  return (
    <ChartCard
      title={`Precision–recall curve — test set (AP = ${fmt.prob(averagePrecision, 4)})`}
      subtitle="Precision against recall, across every threshold"
      caption={`The chance line sits at the flood share of the test set (${fmt.prob(
        positiveShare,
        2,
      )}), not at 0.5 — that is the right baseline for an imbalanced set. The dot is the deployed operating point; hover it for its precision and recall.`}
      table={{
        columns: [
          { key: "recall", label: "Recall", numeric: true },
          { key: "precision", label: "Precision", numeric: true },
          { key: "threshold", label: "Threshold", numeric: true },
        ],
        rows: sorted.map((point) => ({
          recall: fmt.prob(point.recall, 4),
          precision: fmt.prob(point.precision, 4),
          threshold: point.threshold === null ? "—" : fmt.prob(point.threshold, 4),
        })),
      }}
    >
      <ChartLegend
        items={[
          { label: "MobileNetV2", color: "var(--series-1)" },
          {
            label: `chance (flood share ${fmt.prob(positiveShare, 2)})`,
            color: "var(--reference)",
            dashed: true,
          },
          { label: "tuned threshold", color: "var(--text-primary)" },
        ]}
      />
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={sorted} margin={{ top: 8, right: 16, bottom: 18, left: -12 }}>
          <CartesianGrid {...gridProps} vertical />
          <XAxis
            dataKey="recall"
            type="number"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            label={{
              value: "Recall",
              position: "insideBottom",
              offset: -12,
              fill: "var(--text-secondary)",
              fontSize: 12,
            }}
            {...axisProps}
          />
          <YAxis domain={[0, 1.02]} ticks={[0, 0.25, 0.5, 0.75, 1]} {...axisProps} />
          <Tooltip
            cursor={{ stroke: "var(--reference)", strokeDasharray: "3 3" }}
            content={({ active, payload }) =>
              active && payload?.length ? (
                <TooltipBox
                  title={`Recall ${fmt.prob(payload[0].payload.recall, 3)}`}
                  rows={[
                    {
                      label: "Precision",
                      color: "var(--series-1)",
                      value: fmt.prob(payload[0].payload.precision, 3),
                    },
                    {
                      label: "Threshold",
                      value:
                        payload[0].payload.threshold === null
                          ? "—"
                          : fmt.prob(payload[0].payload.threshold, 3),
                    },
                  ]}
                />
              ) : null
            }
          />
          <ReferenceLine
            y={positiveShare}
            stroke="var(--reference)"
            strokeDasharray="5 5"
            strokeWidth={1.5}
          />
          <Line
            type="linear"
            dataKey="precision"
            stroke="var(--series-1)"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, strokeWidth: 2, stroke: "var(--surface-1)" }}
            isAnimationActive={false}
          />
          <ReferenceDot
            x={operating.recall}
            y={operating.precision}
            shape={(props) => operatingDot(props, dotTitle)}
            label={{
              value: `threshold ${fmt.num(operating.threshold, 2)}`,
              position: "left",
              offset: 12,
              fill: "var(--text-primary)",
              fontSize: 11,
            }}
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

/* --------------------------------------------------------------------------
   (d) Confusion matrix - sequential single hue, toggle between thresholds
-------------------------------------------------------------------------- */
const SEQ_STEPS = ["var(--seq-1)", "var(--seq-2)", "var(--seq-3)", "var(--seq-4)", "var(--seq-5)"];

function ConfusionMatrix({ confusion, classNames }) {
  const [which, setWhich] = useState("tuned");
  const [hover, setHover] = useState(null);
  const matrix = confusion[which];
  const threshold = which === "tuned" ? confusion.tuned_threshold : confusion.default_threshold;
  const max = Math.max(...matrix.flat());
  const total = matrix.flat().reduce((sum, value) => sum + value, 0);

  const cellMeaning = [
    ["correctly cleared", "false alarm"],
    ["missed flood", "flood caught"],
  ];

  return (
    <ChartCard
      title="Confusion matrix — test set"
      subtitle={`At threshold ${fmt.num(threshold, 2)} · ${total} test images`}
      controls={
        <Segmented
          label="Decision threshold"
          value={which}
          onChange={setWhich}
          options={[
            { value: "default", label: `default ${fmt.num(confusion.default_threshold, 2)}` },
            { value: "tuned", label: `tuned ${fmt.num(confusion.tuned_threshold, 2)}` },
          ]}
        />
      }
      caption={`Rows are what the image actually is, columns are what the model predicted. ${confusion.note}. Colour is a single hue, light to dark, by count.`}
      table={{
        columns: [
          { key: "cell", label: "Cell" },
          { key: "count", label: "Count", numeric: true },
          { key: "meaning", label: "Meaning" },
        ],
        rows: matrix.flatMap((row, r) =>
          row.map((value, c) => ({
            cell: `actual ${classNames[r]} → pred. ${classNames[c]}`,
            count: value,
            meaning: cellMeaning[r][c],
          })),
        ),
      }}
    >
      <div className="relative">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[320px] border-separate" style={{ borderSpacing: 2 }}>
            <thead>
              <tr>
                <th />
                {classNames.map((name) => (
                  <th
                    key={name}
                    scope="col"
                    className="pb-1.5 text-xs font-medium"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    pred. {name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, r) => (
                <tr key={classNames[r]}>
                  <th
                    scope="row"
                    className="pr-2 text-right text-xs font-medium whitespace-nowrap"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    actual {classNames[r]}
                  </th>
                  {row.map((value, c) => {
                    const step = max === 0 ? 0 : Math.min(4, Math.floor((value / max) * 4.999));
                    return (
                      <td key={c} className="p-0">
                        <div
                          role="img"
                          aria-label={`actual ${classNames[r]}, predicted ${classNames[c]}: ${value} images, ${cellMeaning[r][c]}`}
                          onMouseEnter={() => setHover({ r, c, value })}
                          onMouseLeave={() => setHover(null)}
                          className="flex h-20 items-center justify-center rounded-md text-xl font-semibold transition-transform sm:h-24 sm:text-2xl"
                          style={{
                            background: SEQ_STEPS[step],
                            color: step >= 3 ? "var(--seq-ink-dark)" : "var(--seq-ink-light)",
                          }}
                        >
                          <span className="tnum">{value}</span>
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {hover ? (
          <div className="pointer-events-none absolute top-0 right-0 z-10">
            <TooltipBox
              title={`actual ${classNames[hover.r]} → pred. ${classNames[hover.c]}`}
              rows={[
                { label: "Images", value: hover.value },
                { label: "Share of test set", value: fmt.pct(hover.value / total) },
                { label: "Meaning", value: cellMeaning[hover.r][hover.c] },
              ]}
            />
          </div>
        ) : null}
      </div>
    </ChartCard>
  );
}

/* --------------------------------------------------------------------------
   (e) Probability distribution - why the threshold sits where it does
-------------------------------------------------------------------------- */
function ProbabilityHistogram({ probabilities, threshold, classNames }) {
  const BINS = 20;
  const data = useMemo(() => {
    const bins = Array.from({ length: BINS }, (_, i) => ({
      bin: i / BINS,
      binLabel: `${fmt.num(i / BINS, 2)}–${fmt.num((i + 1) / BINS, 2)}`,
      [classNames[0]]: 0,
      [classNames[1]]: 0,
    }));
    for (const row of probabilities) {
      const index = Math.min(BINS - 1, Math.floor(row.p_flood * BINS));
      bins[index][row.true_class] += 1;
    }
    return bins;
  }, [probabilities, classNames]);

  return (
    <ChartCard
      className="lg:col-span-2"
      title="Where the model puts its probabilities"
      subtitle={`P(flood) for all ${probabilities.length} test images, by what the image actually is`}
      caption={`Both classes pile up at the ends, which is why a threshold anywhere in the middle separates them. The rule sits at ${fmt.num(
        threshold,
        2,
      )} rather than 0.50 because lowering it only moves borderline images into "flood", trading a few false alarms for catching the floods near the boundary.`}
      table={{
        columns: [
          { key: "binLabel", label: "P(flood) bin" },
          { key: "no_flood", label: `actual ${classNames[0]}`, numeric: true },
          { key: "flood", label: `actual ${classNames[1]}`, numeric: true },
        ],
        rows: data.map((row) => ({
          binLabel: row.binLabel,
          no_flood: row[classNames[0]],
          flood: row[classNames[1]],
        })),
      }}
    >
      <ChartLegend
        items={[
          { label: `actual ${classNames[0]}`, color: "var(--series-1)" },
          { label: `actual ${classNames[1]}`, color: "var(--series-2)" },
          { label: `tuned threshold ${fmt.num(threshold, 2)}`, color: "var(--reference)", dashed: true },
        ]}
      />
      <ResponsiveContainer width="100%" height={280}>
        <AreaChart data={data} margin={{ top: 8, right: 12, bottom: 18, left: -12 }}>
          <CartesianGrid {...gridProps} />
          <XAxis
            dataKey="bin"
            type="number"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tickFormatter={(value) => fmt.num(value, 2)}
            label={{
              value: "P(flood)",
              position: "insideBottom",
              offset: -12,
              fill: "var(--text-secondary)",
              fontSize: 12,
            }}
            {...axisProps}
          />
          <YAxis
            {...axisProps}
            label={{
              value: "images",
              angle: -90,
              position: "insideLeft",
              offset: 20,
              fill: "var(--text-secondary)",
              fontSize: 12,
            }}
          />
          <Tooltip
            cursor={{ stroke: "var(--reference)", strokeDasharray: "3 3" }}
            content={({ active, payload }) =>
              active && payload?.length ? (
                <TooltipBox
                  title={`P(flood) ${payload[0].payload.binLabel}`}
                  rows={payload.map((item) => ({
                    label: `actual ${item.dataKey}`,
                    color: item.stroke,
                    value: `${item.value} images`,
                  }))}
                />
              ) : null
            }
          />
          <Area
            type="stepAfter"
            dataKey={classNames[0]}
            stroke="var(--series-1)"
            strokeWidth={2}
            fill="var(--series-1)"
            fillOpacity={0.3}
            isAnimationActive={false}
          />
          <Area
            type="stepAfter"
            dataKey={classNames[1]}
            stroke="var(--series-2)"
            strokeWidth={2}
            fill="var(--series-2)"
            fillOpacity={0.3}
            isAnimationActive={false}
          />
          <ReferenceLine
            x={threshold}
            stroke="var(--reference)"
            strokeDasharray="5 5"
            strokeWidth={1.5}
            label={{
              value: `threshold ${fmt.num(threshold, 2)}`,
              position: "top",
              fill: "var(--text-primary)",
              fontSize: 11,
            }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

export default function Performance({ data }) {
  const classNames = data.meta.class_names;   // [negative, positive] - matches the confusion rows
  return (
    <div className="space-y-4">
      <SectionIntro title="Performance">
        How well the deployed model separates floods from everything else on the held-out test set
        of {data.meta.n_test} images, and what the chosen threshold does to that.
      </SectionIntro>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ModelComparison comparison={data.comparison} seeds={data.seeds} note={data.seed_note} />
        <ConfusionMatrix confusion={data.confusion} classNames={classNames} />
        <RocCurve roc={data.roc} rocAuc={data.roc_auc} operating={data.operating_point} />
        <PrCurve
          pr={data.pr}
          averagePrecision={data.average_precision}
          positiveShare={data.positive_share}
          operating={data.operating_point}
        />
        <ProbabilityHistogram
          probabilities={data.probabilities}
          threshold={data.meta.threshold}
          classNames={classNames}
        />
      </div>
    </div>
  );
}
