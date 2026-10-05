import { useCallback, useEffect, useRef, useState } from "react";
import { ChartCard, fmt, SectionIntro, StatTile } from "./ui.jsx";

/* ---------------------------------------------------------------------------
   Live prediction. This is the one part of the dashboard that is not static:
   the Keras model cannot run in a browser, so an upload is sent to the local
   API (python src/serve_api.py), which reuses the same functions the Streamlit
   app and the evaluation scripts use.
--------------------------------------------------------------------------- */

function useApiHealth() {
  const [health, setHealth] = useState({ status: "checking" });

  const check = useCallback(() => {
    setHealth({ status: "checking" });
    fetch("/api/health")
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      // the payload carries its own "status": "ok", so our flag is set AFTER the spread
      .then((data) => setHealth({ ...data, status: "up" }))
      .catch((error) => setHealth({ status: "down", error: String(error) }));
  }, []);

  useEffect(check, [check]);
  return [health, check];
}

function ApiDownCard({ onRetry, error }) {
  return (
    <div
      className="rounded-xl p-5"
      style={{ background: "var(--surface-1)", border: "1px solid var(--status-warning)" }}
    >
      <h3 className="text-base font-semibold" style={{ color: "var(--status-warning)" }}>
        The prediction service is not running
      </h3>
      <p className="mt-2 text-sm" style={{ color: "var(--text-secondary)" }}>
        Everything else on this dashboard is static and needs nothing running. Prediction is the
        exception: the model has to run somewhere, so start the local API in a second terminal.
      </p>
      <pre
        className="mt-3 overflow-x-auto rounded-lg p-3 text-sm"
        style={{ background: "var(--surface-page)", color: "var(--text-primary)" }}
      >
        {"cd dashboard\nnpm run api"}
      </pre>
      <p className="mt-2 text-xs" style={{ color: "var(--text-muted)" }}>
        It loads the model once, which takes a few seconds, then serves http://127.0.0.1:8000.
      </p>
      <button
        type="button"
        onClick={onRetry}
        className="mt-3 rounded-lg px-3 py-1.5 text-xs font-medium"
        style={{
          background: "var(--surface-2)",
          border: "1px solid var(--border)",
          color: "var(--text-primary)",
        }}
      >
        Check again
      </button>
      {error ? (
        <p className="mt-2 text-xs tnum" style={{ color: "var(--text-muted)" }}>
          {error}
        </p>
      ) : null}
    </div>
  );
}

function Verdict({ result }) {
  const accent = result.is_flood ? "var(--status-critical)" : "var(--status-good)";
  return (
    <div
      className="rounded-xl p-4"
      style={{ background: "var(--surface-2)", border: `1px solid ${accent}` }}
    >
      <div className="flex items-baseline gap-3">
        <span className="text-2xl font-semibold" style={{ color: accent }}>
          {result.is_flood ? "Flood" : "No flood"}
        </span>
        <span className="tnum text-sm" style={{ color: "var(--text-secondary)" }}>
          P(flood) = {fmt.prob(result.p_flood, 4)}
        </span>
      </div>
      <p className="mt-2 text-sm" style={{ color: "var(--text-secondary)" }}>
        {result.is_flood
          ? `At or above the decision threshold of ${fmt.num(result.threshold, 2)}.`
          : `Below the decision threshold of ${fmt.num(result.threshold, 2)}.`}
      </p>

      {/* probability bar, with the threshold marked on it */}
      <div className="relative mt-3 h-3 w-full rounded-full" style={{ background: "var(--grid)" }}>
        <div
          className="h-3 rounded-full"
          style={{ width: `${result.p_flood * 100}%`, background: accent }}
        />
        <div
          className="absolute top-[-4px] h-5 w-0.5"
          style={{ left: `${result.threshold * 100}%`, background: "var(--text-primary)" }}
          title={`threshold ${fmt.num(result.threshold, 2)}`}
        />
      </div>
      <div className="mt-1 flex justify-between text-xs tnum" style={{ color: "var(--text-muted)" }}>
        <span>0</span>
        <span>threshold {fmt.num(result.threshold, 2)}</span>
        <span>1</span>
      </div>
    </div>
  );
}

export default function Predict({ data }) {
  const [health, recheck] = useApiHealth();
  const [state, setState] = useState({ status: "idle" });
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const send = useCallback((file) => {
    if (!file) return;
    setState({ status: "working", name: file.name });
    const body = new FormData();
    body.append("file", file);
    fetch("/api/predict", { method: "POST", body })
      .then(async (response) => {
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
        return payload;
      })
      .then((result) => setState({ status: "done", result }))
      .catch((error) => setState({ status: "failed", error: String(error.message || error) }));
  }, []);

  if (health.status === "down") {
    return (
      <div className="space-y-4">
        <SectionIntro title="Predict">
          Upload an aerial image and the deployed model classifies it, with Grad-CAM showing which
          part of the image drove the decision.
        </SectionIntro>
        <ApiDownCard onRetry={recheck} error={health.error} />
      </div>
    );
  }

  const result = state.status === "done" ? state.result : null;

  return (
    <div className="space-y-4">
      <SectionIntro title="Predict">
        Upload an aerial image and the deployed model classifies it at the tuned threshold of{" "}
        {fmt.num(data.meta.threshold, 2)}, with Grad-CAM showing which part of the image drove the
        decision. This is the only part of the dashboard that runs the model.
      </SectionIntro>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ChartCard
          title="Upload an image"
          subtitle={
            health.status === "up"
              ? `${health.model} on ${health.device} · threshold ${fmt.num(health.threshold, 2)}`
              : "checking the prediction service…"
          }
          caption="Trained on AIDER aerial imagery. Ground-level and CCTV images are untested — see the README, 'Known limitations'. JPEG, PNG or WebP, up to 12 MB. The image is centre-cropped to 224×224, exactly as it is during evaluation, so the overlay lines up with what the model saw."
        >
          <div
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              send(event.dataTransfer.files?.[0]);
            }}
            className="flex flex-col items-center justify-center rounded-xl px-4 py-10 text-center"
            style={{
              border: `2px dashed ${dragging ? "var(--series-1)" : "var(--border)"}`,
              background: dragging ? "var(--surface-2)" : "transparent",
            }}
          >
            <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
              Drop an image here
            </p>
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              className="mt-3 rounded-lg px-4 py-2 text-sm font-medium"
              style={{ background: "var(--series-1)", color: "#ffffff" }}
            >
              Choose a file
            </button>
            <input
              ref={inputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              className="hidden"
              onChange={(event) => send(event.target.files?.[0])}
            />
            {state.status === "working" ? (
              <p className="mt-3 text-xs tnum" style={{ color: "var(--text-muted)" }}>
                Classifying {state.name}…
              </p>
            ) : null}
            {state.status === "failed" ? (
              <p className="mt-3 text-xs" style={{ color: "var(--status-critical)" }}>
                {state.error}
              </p>
            ) : null}
          </div>

          {result ? (
            <div className="mt-4">
              <Verdict result={result} />
            </div>
          ) : null}
        </ChartCard>

        <ChartCard
          title="Where the model looked"
          subtitle={result ? result.filename : "Grad-CAM overlay appears after a prediction"}
          caption={
            result
              ? `The heat map covers ${fmt.pctOf100(
                  result.attention_area_pct,
                  1,
                )} of the frame. Area alone is not evidence — see the deletion test under Explainability, which removes the attended region and measures what happens to the prediction.`
              : "The overlay is the Grad-CAM heat map blended over the 224×224 crop the model actually sees."
          }
          table={
            result
              ? {
                  columns: [
                    { key: "field", label: "Field" },
                    { key: "value", label: "Value", numeric: true },
                  ],
                  rows: [
                    { field: "P(flood)", value: fmt.prob(result.p_flood, 6) },
                    { field: "Decision threshold", value: fmt.num(result.threshold, 2) },
                    { field: "Label", value: result.label },
                    {
                      field: "Attention area",
                      value: fmt.pctOf100(result.attention_area_pct, 2),
                    },
                    {
                      field: "Warning line",
                      value: fmt.pctOf100(result.attention_warn_pct, 0),
                    },
                  ],
                }
              : null
          }
        >
          {result ? (
            <div className="grid grid-cols-2 gap-3">
              <figure>
                <img
                  src={result.image}
                  alt="The 224×224 centre crop the model received"
                  className="w-full rounded-lg"
                  style={{ border: "1px solid var(--border)" }}
                />
                <figcaption className="mt-1 text-xs" style={{ color: "var(--text-muted)" }}>
                  What the model saw
                </figcaption>
              </figure>
              <figure>
                <img
                  src={result.overlay}
                  alt={`Grad-CAM heat map covering ${fmt.pctOf100(result.attention_area_pct, 1)} of the frame`}
                  className="w-full rounded-lg"
                  style={{ border: "1px solid var(--border)" }}
                />
                <figcaption className="mt-1 text-xs" style={{ color: "var(--text-muted)" }}>
                  Grad-CAM overlay
                </figcaption>
              </figure>
            </div>
          ) : (
            <div
              className="flex h-48 items-center justify-center rounded-lg text-sm"
              style={{ border: "1px dashed var(--border)", color: "var(--text-muted)" }}
            >
              No prediction yet
            </div>
          )}
        </ChartCard>
      </div>
    </div>
  );
}
