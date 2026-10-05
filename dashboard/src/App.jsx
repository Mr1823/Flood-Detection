import { useState } from "react";
import { ErrorCard, Skeleton, useDashboard, useTheme } from "./data.jsx";
import Dataset from "./components/Dataset.jsx";
import Explainability from "./components/Explainability.jsx";
import Performance from "./components/Performance.jsx";
import Threshold from "./components/Threshold.jsx";
import Training from "./components/Training.jsx";
import { fmt, StatTile } from "./components/ui.jsx";

const TABS = [
  { id: "performance", label: "Performance", Component: Performance },
  { id: "training", label: "Training", Component: Training },
  { id: "threshold", label: "Threshold", Component: Threshold },
  { id: "explainability", label: "Explainability", Component: Explainability },
  { id: "dataset", label: "Dataset", Component: Dataset },
];

function ThemeToggle() {
  const [theme, toggle] = useTheme();
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
      className="rounded-lg px-3 py-1.5 text-xs font-medium"
      style={{
        background: "var(--surface-1)",
        border: "1px solid var(--border)",
        color: "var(--text-primary)",
      }}
    >
      {theme === "dark" ? "Light mode" : "Dark mode"}
    </button>
  );
}

function Header({ meta }) {
  const generated = new Date(meta.generated_at);
  return (
    <header className="flex flex-wrap items-start justify-between gap-4">
      <div className="min-w-0">
        <h1 className="text-2xl font-semibold" style={{ color: "var(--text-primary)" }}>
          Flood detection — results
        </h1>
        <p className="mt-1 text-sm" style={{ color: "var(--text-secondary)" }}>
          {meta.architecture}
        </p>
        <p className="mt-1 text-xs tnum" style={{ color: "var(--text-muted)" }}>
          {meta.model_name} · seed {meta.seed_deployed} ({meta.variant_deployed}) · generated{" "}
          {generated.toLocaleString()}
        </p>
      </div>
      <ThemeToggle />
    </header>
  );
}

function StatRow({ data }) {
  const tflite = data.export
    .filter((row) => row.format.toLowerCase().includes("tflite"))
    .sort((a, b) => a.size_mb - b.size_mb)[0];
  const keras = data.export.find((row) => row.format.toLowerCase().includes("keras"));

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <StatTile
        value={fmt.pct(data.headline.recall)}
        label="Recall on floods"
        context={`${data.headline.test_floods - data.headline.false_negatives} of ${
          data.headline.test_floods
        } test floods caught at threshold ${fmt.num(data.meta.threshold, 2)}`}
      />
      <StatTile
        value={fmt.prob(data.roc_auc, 4)}
        label="ROC AUC"
        context={`Average precision ${fmt.prob(data.average_precision, 4)} on ${data.meta.n_test} test images`}
      />
      <StatTile
        value={fmt.num(data.meta.threshold, 2)}
        label="Decision threshold"
        context={`Chosen on ${data.meta.threshold_selected_on}, not on test`}
      />
      <StatTile
        value={`${fmt.num(tflite.size_mb, 1)} MB`}
        label="TFLite size and latency"
        context={`${fmt.num(tflite.latency_ms, 1)} ms per image on ${tflite.device}, against ${fmt.num(
          keras.size_mb,
          1,
        )} MB / ${fmt.num(keras.latency_ms, 1)} ms for Keras`}
      />
    </div>
  );
}

export default function App() {
  const { status, data, error } = useDashboard();
  const [active, setActive] = useState("performance");

  if (status === "loading") return <Skeleton />;
  if (status === "error") return <ErrorCard error={error} />;

  const Active = TABS.find((tab) => tab.id === active).Component;

  return (
    <div className="mx-auto max-w-7xl px-4 py-8">
      <Header meta={data.meta} />

      <div className="mt-6">
        <StatRow data={data} />
      </div>

      <nav
        className="mt-8 flex gap-1 overflow-x-auto"
        style={{ borderBottom: "1px solid var(--border)" }}
        aria-label="Dashboard sections"
      >
        {TABS.map((tab) => {
          const isActive = tab.id === active;
          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActive(tab.id)}
              aria-current={isActive ? "page" : undefined}
              className="shrink-0 px-3 py-2 text-sm font-medium whitespace-nowrap"
              style={{
                color: isActive ? "var(--text-primary)" : "var(--text-secondary)",
                borderBottom: `2px solid ${isActive ? "var(--series-1)" : "transparent"}`,
                marginBottom: -1,
              }}
            >
              {tab.label}
            </button>
          );
        })}
      </nav>

      <main className="mt-6">
        <Active data={data} />
      </main>

      <footer className="mt-10 text-xs" style={{ color: "var(--text-muted)" }}>
        Every number on this page comes from <code>dashboard/public/data/dashboard.json</code>,
        generated from <code>reports/</code> by <code>src/export_dashboard_data.py</code>. The
        dashboard runs no model — live prediction stays in the Streamlit app.
      </footer>
    </div>
  );
}
