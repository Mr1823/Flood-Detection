import { useMemo, useState } from "react";
import { ChartCard, fmt, StatTile } from "./ui.jsx";

/* ---------------------------------------------------------------------------
   "What if the threshold were somewhere else?"

   Every number here is recomputed in the browser from the saved per-image test
   probabilities, so moving the slider is exact, not interpolated - the model is
   not re-run and nothing is estimated. The deployed threshold is still the one
   in dashboard.json; this panel only shows what a different choice would cost.
--------------------------------------------------------------------------- */
function scoresAt(probabilities, threshold, positive) {
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;
  for (const row of probabilities) {
    const predictedFlood = row.p_flood >= threshold;
    const actuallyFlood = row.true_class === positive;
    if (actuallyFlood && predictedFlood) tp += 1;
    else if (actuallyFlood) fn += 1;
    else if (predictedFlood) fp += 1;
    else tn += 1;
  }
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  return {
    tp,
    fp,
    tn,
    fn,
    precision,
    recall,
    f1: precision + recall ? (2 * precision * recall) / (precision + recall) : 0,
    accuracy: (tp + tn) / probabilities.length,
  };
}

export default function ThresholdExplorer({ data }) {
  const deployed = data.meta.threshold;
  const [negative, positive] = data.meta.class_names;
  const [value, setValue] = useState(deployed);

  const live = useMemo(
    () => scoresAt(data.probabilities, value, positive),
    [data.probabilities, value, positive],
  );
  const atDeployed = useMemo(
    () => scoresAt(data.probabilities, deployed, positive),
    [data.probabilities, deployed, positive],
  );

  /* The validation sweep is what the threshold was actually chosen from, so the
     nearest swept row is shown beside the live test numbers - labelled as the
     nearest, because the sweep has a 0.05 step and this slider does not. */
  const nearestValidation = useMemo(() => {
    const rows = data.threshold_sweep.filter((row) => row.split === "validation");
    return rows.reduce((best, row) =>
      Math.abs(row.threshold - value) < Math.abs(best.threshold - value) ? row : best,
    );
  }, [data.threshold_sweep, value]);

  const moved = Math.abs(value - deployed) > 1e-9;

  return (
    <ChartCard
      title="Try another threshold"
      subtitle={`Recomputed live from all ${data.probabilities.length} saved test probabilities`}
      caption={`Moving the slider does not re-run the model: it re-counts the saved per-image probabilities, so every number is exact. The deployed rule is still ${fmt.num(
        deployed,
        2,
      )}, chosen on validation — this panel only shows what a different choice would cost on the test set.`}
      table={{
        columns: [
          { key: "field", label: "At this threshold" },
          { key: "value", label: "Value", numeric: true },
          { key: "deployed", label: `At deployed ${fmt.num(deployed, 2)}`, numeric: true },
        ],
        rows: [
          { field: "Precision", value: fmt.prob(live.precision, 4), deployed: fmt.prob(atDeployed.precision, 4) },
          { field: "Recall", value: fmt.prob(live.recall, 4), deployed: fmt.prob(atDeployed.recall, 4) },
          { field: "F1", value: fmt.prob(live.f1, 4), deployed: fmt.prob(atDeployed.f1, 4) },
          { field: "Accuracy", value: fmt.prob(live.accuracy, 4), deployed: fmt.prob(atDeployed.accuracy, 4) },
          { field: `Floods missed (of ${live.tp + live.fn})`, value: live.fn, deployed: atDeployed.fn },
          { field: `False alarms (of ${live.tn + live.fp})`, value: live.fp, deployed: atDeployed.fp },
        ],
      }}
    >
      <label
        htmlFor="threshold-slider"
        className="block text-sm font-medium"
        style={{ color: "var(--text-secondary)" }}
      >
        Flood if probability ≥{" "}
        <span className="tnum text-base font-semibold" style={{ color: "var(--series-1)" }}>
          {fmt.num(value, 2)}
        </span>
        {moved ? (
          <button
            type="button"
            onClick={() => setValue(deployed)}
            className="ml-3 text-xs font-medium underline underline-offset-2"
            style={{ color: "var(--text-secondary)" }}
          >
            reset to deployed {fmt.num(deployed, 2)}
          </button>
        ) : (
          <span className="ml-3 text-xs font-normal" style={{ color: "var(--text-muted)" }}>
            the deployed value
          </span>
        )}
      </label>

      <input
        id="threshold-slider"
        type="range"
        min={0.01}
        max={0.99}
        step={0.01}
        value={value}
        onChange={(event) => setValue(Number(event.target.value))}
        className="mt-2 w-full"
        style={{ accentColor: "var(--series-1)" }}
        aria-valuetext={`threshold ${fmt.num(value, 2)}, recall ${fmt.pct(live.recall)}, precision ${fmt.pct(live.precision)}`}
      />
      <div className="flex justify-between text-xs tnum" style={{ color: "var(--text-muted)" }}>
        <span>0.01</span>
        <span>deployed {fmt.num(deployed, 2)}</span>
        <span>0.99</span>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile
          value={fmt.pct(live.recall)}
          label="Recall on floods"
          context={`${live.fn} of ${live.tp + live.fn} floods missed`}
          accent={live.fn > atDeployed.fn ? "var(--status-critical)" : undefined}
        />
        <StatTile
          value={fmt.pct(live.precision)}
          label="Precision"
          context={`${live.fp} false alarms among ${live.tn + live.fp} non-floods`}
        />
        <StatTile value={fmt.prob(live.f1, 3)} label="F1" context="Harmonic mean of the two" />
        <StatTile
          value={fmt.prob(nearestValidation.recall, 3)}
          label="Validation recall"
          context={`At the nearest swept threshold, ${fmt.num(nearestValidation.threshold, 2)} — the split the choice was made on`}
        />
      </div>

      {/* confusion matrix at the live threshold */}
      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[320px] border-separate" style={{ borderSpacing: 2 }}>
          <thead>
            <tr>
              <th />
              <th className="pb-1.5 text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
                pred. {negative}
              </th>
              <th className="pb-1.5 text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
                pred. {positive}
              </th>
            </tr>
          </thead>
          <tbody>
            {[
              [`actual ${negative}`, live.tn, live.fp],
              [`actual ${positive}`, live.fn, live.tp],
            ].map(([label, left, right]) => (
              <tr key={label}>
                <th
                  scope="row"
                  className="pr-2 text-right text-xs font-medium whitespace-nowrap"
                  style={{ color: "var(--text-secondary)" }}
                >
                  {label}
                </th>
                {[left, right].map((count, index) => (
                  <td key={index} className="p-0">
                    <div
                      className="flex h-14 items-center justify-center rounded-md text-lg font-semibold tnum"
                      style={{
                        background: "var(--surface-page)",
                        border: "1px solid var(--border)",
                        color: "var(--text-primary)",
                      }}
                    >
                      {count}
                    </div>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </ChartCard>
  );
}
