import { createContext, useContext, useEffect, useMemo, useState } from "react";

/* dashboard.json is produced by src/export_dashboard_data.py from reports/.
   It is fetched once, here, and every chart reads from this context. No chart
   fetches anything, and no number in the UI comes from anywhere else. */
const DataContext = createContext(null);

export function DataProvider({ children }) {
  const [state, setState] = useState({ status: "loading", data: null, error: null });

  useEffect(() => {
    let cancelled = false;
    fetch(`${import.meta.env.BASE_URL}data/dashboard.json`)
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status} ${response.statusText}`);
        return response.json();
      })
      .then((data) => !cancelled && setState({ status: "ready", data, error: null }))
      .catch((error) => !cancelled && setState({ status: "error", data: null, error }));
    return () => {
      cancelled = true;
    };
  }, []);

  return <DataContext.Provider value={state}>{children}</DataContext.Provider>;
}

export function useDashboard() {
  const context = useContext(DataContext);
  if (!context) throw new Error("useDashboard must be used inside <DataProvider>");
  return context;
}

/* The deployed threshold, read from the data - never written as a literal. */
export function useThreshold() {
  const { data } = useDashboard();
  return data?.meta?.threshold;
}

/* Theme: system preference first, then the reader's explicit choice. */
export function useTheme() {
  const [theme, setTheme] = useState(() => {
    try {
      const saved = localStorage.getItem("dashboard-theme");
      if (saved === "light" || saved === "dark") return saved;
    } catch {
      /* private window or blocked storage - fall through to the system choice */
    }
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("dashboard-theme", theme);
    } catch {
      /* not persisting the choice is harmless */
    }
  }, [theme]);

  return [theme, () => setTheme((current) => (current === "dark" ? "light" : "dark"))];
}

export function Skeleton() {
  const bars = [160, 220, 190];
  return (
    <div className="mx-auto max-w-7xl px-4 py-10" aria-busy="true" aria-live="polite">
      <div className="h-8 w-72 animate-pulse rounded" style={{ background: "var(--grid)" }} />
      <div className="mt-3 h-4 w-96 animate-pulse rounded" style={{ background: "var(--grid)" }} />
      <div className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[0, 1, 2, 3].map((i) => (
          <div
            key={i}
            className="h-28 animate-pulse rounded-xl"
            style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
          />
        ))}
      </div>
      <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
        {bars.map((height, i) => (
          <div
            key={i}
            className="animate-pulse rounded-xl"
            style={{
              height,
              background: "var(--surface-1)",
              border: "1px solid var(--border)",
            }}
          />
        ))}
      </div>
      <p className="mt-6 text-sm" style={{ color: "var(--text-secondary)" }}>
        Loading dashboard.json…
      </p>
    </div>
  );
}

export function ErrorCard({ error }) {
  return (
    <div className="mx-auto max-w-3xl px-4 py-16">
      <div
        className="rounded-xl p-6"
        style={{ background: "var(--surface-1)", border: "1px solid var(--status-critical)" }}
      >
        <h1 className="text-lg font-semibold" style={{ color: "var(--status-critical)" }}>
          Could not load dashboard.json
        </h1>
        <p className="mt-2 text-sm" style={{ color: "var(--text-secondary)" }}>
          The dashboard shows only numbers the pipeline has already produced, so it cannot start
          without its data file.
        </p>
        <p className="mt-4 text-sm" style={{ color: "var(--text-secondary)" }}>
          Expected at <code className="tnum">dashboard/public/data/dashboard.json</code>. Generate it
          with:
        </p>
        <pre
          className="mt-3 overflow-x-auto rounded-lg p-3 text-sm"
          style={{ background: "var(--surface-page)", color: "var(--text-primary)" }}
        >
          npm run data
        </pre>
        <p className="mt-4 text-xs tnum" style={{ color: "var(--text-muted)" }}>
          {String(error)}
        </p>
      </div>
    </div>
  );
}
