import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { axisProps, ChartCard, fmt, gridProps, SectionIntro, StatTile, TooltipBox } from "./ui.jsx";

/* (h) The deletion test - the headline evidence that attention is load-bearing */
function DeletionTest({ gradcam }) {
  const { deletion } = gradcam;
  const share = fmt.pctOf100(deletion.deleted_share * 100, 0);
  const data = [
    {
      label: "Untouched image",
      value: deletion.baseline,
      detail: "the model's own confidence before anything is removed",
      color: "var(--series-1)",
    },
    {
      label: `Most-attended ${share} removed`,
      value: deletion.most_attended_removed,
      detail: "the region Grad-CAM says it is looking at, replaced with grey",
      color: "var(--series-2)",
    },
    {
      label: `Least-attended ${share} removed`,
      value: deletion.least_attended_removed,
      detail: "an equal area it says it is not looking at, replaced with grey",
      color: "var(--series-3)",
    },
  ];

  const drop = deletion.baseline - deletion.most_attended_removed;
  const control = deletion.baseline - deletion.least_attended_removed;

  return (
    <ChartCard
      className="lg:col-span-2"
      title="Grad-CAM deletion test"
      subtitle={`Mean P(flood) over the ${deletion.images} correctly detected test floods · ${gradcam.layer}`}
      caption={`Removing the quarter of the image Grad-CAM attends to costs ${fmt.prob(
        drop,
        3,
      )} of mean confidence, while removing an equal area it ignores costs only ${fmt.prob(
        control,
        3,
      )}. The gap between those two bars is the evidence: if the heat map pointed somewhere irrelevant, blanking it would not matter more than blanking anywhere else. ${fmt.pctOf100(
        deletion.still_flood_most_removed_pct,
        1,
      )} of these floods are still called floods after the attended region is removed, against ${fmt.pctOf100(
        deletion.still_flood_least_removed_pct,
        1,
      )} for the control.`}
      table={{
        columns: [
          { key: "label", label: "Condition" },
          { key: "value", label: "Mean P(flood)", numeric: true },
          { key: "detail", label: "What it is" },
        ],
        rows: data.map((row) => ({
          label: row.label,
          value: fmt.prob(row.value, 4),
          detail: row.detail,
        })),
      }}
    >
      <ResponsiveContainer width="100%" height={210}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 56, bottom: 4, left: 8 }}
          barCategoryGap={10}
        >
          <CartesianGrid {...gridProps} horizontal={false} vertical />
          <XAxis type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} {...axisProps} />
          <YAxis
            type="category"
            dataKey="label"
            width={190}
            tick={{ fill: "var(--text-secondary)", fontSize: 12 }}
            tickLine={false}
            stroke="var(--axis)"
          />
          <Tooltip
            cursor={{ fill: "var(--grid)", fillOpacity: 0.4 }}
            content={({ active, payload }) =>
              active && payload?.length ? (
                <TooltipBox
                  title={payload[0].payload.label}
                  rows={[
                    {
                      label: "Mean P(flood)",
                      color: payload[0].payload.color,
                      value: fmt.prob(payload[0].value, 4),
                    },
                    { label: "", value: payload[0].payload.detail },
                  ]}
                />
              ) : null
            }
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={34}
            isAnimationActive={false}
          >
            {data.map((row) => (
              <Cell key={row.label} fill={row.color} />
            ))}
            <LabelList
              dataKey="value"
              position="right"
              offset={8}
              formatter={(value) => fmt.prob(value, 3)}
              style={{ fill: "var(--text-primary)", fontSize: 12, fontVariantNumeric: "tabular-nums" }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

/* (i) Attention area - a stat pair, not a chart */
function AttentionArea({ gradcam, classNames }) {
  return (
    <ChartCard
      className="lg:col-span-2"
      title="Attention area"
      subtitle={`How much of the image the heat map lights up, on true ${classNames[1]} images`}
      caption={`Area alone is inconclusive for aerial imagery: a flood often covers most of the frame, so a large attended area can mean the model found the water or simply that it spread its attention over everything. That is why the deletion test above, which removes the attended region and measures what happens, is the evidence — not the size of the blob. For contrast, true ${classNames[0]} images average ${fmt.pctOf100(
        gradcam.mean_attention_no_flood,
        1,
      )}.`}
      table={{
        columns: [
          { key: "stat", label: "Statistic" },
          { key: "value", label: "Value", numeric: true },
        ],
        rows: [
          {
            stat: `Mean attention area, true ${classNames[1]}`,
            value: fmt.pctOf100(gradcam.mean_attention_flood, 2),
          },
          {
            stat: `Share above the ${fmt.pctOf100(gradcam.warn_above_pct, 0)} warning line`,
            value: fmt.pctOf100(gradcam.pct_above_50, 2),
          },
          {
            stat: `Mean attention area, true ${classNames[0]}`,
            value: fmt.pctOf100(gradcam.mean_attention_no_flood, 2),
          },
        ],
      }}
    >
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <StatTile
          value={fmt.pctOf100(gradcam.mean_attention_flood, 1)}
          label={`Mean attention area on floods`}
          context={`Against ${fmt.pctOf100(gradcam.mean_attention_no_flood, 1)} on true ${
            classNames[0]
          } images`}
        />
        <StatTile
          value={fmt.pctOf100(gradcam.pct_above_50, 1)}
          label={`Above the ${fmt.pctOf100(gradcam.warn_above_pct, 0)} warning line`}
          context="Flood images whose heat map covers more than half the frame"
        />
      </div>
    </ChartCard>
  );
}

export default function Explainability({ data }) {
  const classNames = data.meta.class_names;
  return (
    <div className="space-y-4">
      <SectionIntro title="Explainability">
        Grad-CAM says where the model is looking. On its own that is a picture, not evidence — so the
        attended region is removed and the model is asked again.
      </SectionIntro>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <DeletionTest gradcam={data.gradcam} />
        <AttentionArea gradcam={data.gradcam} classNames={classNames} />
      </div>
    </div>
  );
}
