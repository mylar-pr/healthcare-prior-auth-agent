import { useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const STATUS_LABEL = {
  met: "Met",
  not_met: "Not met",
  needs_review: "Needs review",
  not_applicable: "N/A",
};

export default function App() {
  const [planText, setPlanText] = useState("");
  const [planFile, setPlanFile] = useState(null);
  const [session, setSession] = useState(null); // {session_id, source_name, num_chunks}
  const [indexing, setIndexing] = useState(false);
  const [indexError, setIndexError] = useState(null);

  const [drugName, setDrugName] = useState("");
  const [dosage, setDosage] = useState("");
  const [diagnosisCode, setDiagnosisCode] = useState("");
  const [quantity, setQuantity] = useState("");
  const [prescriberName, setPrescriberName] = useState("");
  const [patientName, setPatientName] = useState("");
  const [stepTherapyDocumented, setStepTherapyDocumented] = useState(""); // "" | "yes" | "no"

  const [draft, setDraft] = useState(null); // {draft, checklist, citations, tool_trace}
  const [drafting, setDrafting] = useState(false);
  const [draftError, setDraftError] = useState(null);

  async function handleIndex(e) {
    e.preventDefault();
    if (!planText.trim() && !planFile) return;
    setIndexing(true);
    setIndexError(null);
    setSession(null);
    setDraft(null);
    try {
      const form = new FormData();
      if (planFile) {
        form.append("file", planFile);
      } else {
        form.append("plan_text", planText.trim());
      }
      const res = await fetch(`${API_BASE}/plan`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to index plan document");
      setSession(data);
    } catch (err) {
      setIndexError(err.message);
    } finally {
      setIndexing(false);
    }
  }

  async function handleDraft(e) {
    e.preventDefault();
    if (!session || !drugName.trim() || !dosage.trim() || !diagnosisCode.trim()) return;
    setDrafting(true);
    setDraftError(null);
    setDraft(null);
    try {
      const res = await fetch(`${API_BASE}/draft`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: session.session_id,
          drug_name: drugName.trim(),
          dosage: dosage.trim(),
          diagnosis_code: diagnosisCode.trim(),
          quantity: quantity.trim() || null,
          prescriber_name: prescriberName.trim() || null,
          patient_name: patientName.trim() || null,
          step_therapy_documented:
            stepTherapyDocumented === "" ? null : stepTherapyDocumented === "yes",
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to draft request");
      setDraft(data);
    } catch (err) {
      setDraftError(err.message);
    } finally {
      setDrafting(false);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Healthcare Prior Auth Agent</h1>
        <p className="subtitle">
          Paste or upload a plan document, enter a prescription, and get a drafted
          prior-authorization request citing the plan's own clauses.
        </p>
      </header>

      <section className="panel">
        <h2>1. Plan document</h2>
        <form className="plan-form" onSubmit={handleIndex}>
          <textarea
            placeholder="Paste plan document text here…"
            value={planText}
            onChange={(e) => {
              setPlanText(e.target.value);
              if (e.target.value) setPlanFile(null);
            }}
            disabled={indexing || !!planFile}
            rows={8}
          />
          <div className="file-row">
            <span>— or —</span>
            <input
              type="file"
              accept=".pdf,.txt"
              onChange={(e) => {
                setPlanFile(e.target.files[0] || null);
                if (e.target.files[0]) setPlanText("");
              }}
              disabled={indexing}
            />
            {planFile && <span className="filename">{planFile.name}</span>}
          </div>
          <button type="submit" disabled={indexing || (!planText.trim() && !planFile)}>
            {indexing ? "Indexing…" : "Index plan document"}
          </button>
        </form>
        {indexError && <div className="error-banner">{indexError}</div>}
        {session && (
          <div className="session-banner">
            Indexed <strong>{session.source_name}</strong> — {session.num_chunks} chunks.
          </div>
        )}
      </section>

      <section className="panel">
        <h2>2. Prescription</h2>
        <form className="rx-form" onSubmit={handleDraft}>
          <div className="rx-grid">
            <label>
              Drug name
              <input
                type="text"
                value={drugName}
                onChange={(e) => setDrugName(e.target.value)}
                placeholder="e.g. Humira (adalimumab)"
                disabled={!session}
              />
            </label>
            <label>
              Dosage / strength
              <input
                type="text"
                value={dosage}
                onChange={(e) => setDosage(e.target.value)}
                placeholder="e.g. 40mg/0.4mL subcutaneous injection"
                disabled={!session}
              />
            </label>
            <label>
              Diagnosis code (ICD-10)
              <input
                type="text"
                value={diagnosisCode}
                onChange={(e) => setDiagnosisCode(e.target.value)}
                placeholder="e.g. M06.9"
                disabled={!session}
              />
            </label>
            <label>
              Quantity / frequency
              <input
                type="text"
                value={quantity}
                onChange={(e) => setQuantity(e.target.value)}
                placeholder="e.g. 2 pens every 2 weeks"
                disabled={!session}
              />
            </label>
            <label>
              Prescriber name
              <input
                type="text"
                value={prescriberName}
                onChange={(e) => setPrescriberName(e.target.value)}
                placeholder="e.g. Dr. Jane Alvarez"
                disabled={!session}
              />
            </label>
            <label>
              Patient name (synthetic)
              <input
                type="text"
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
                placeholder="e.g. Test Patient"
                disabled={!session}
              />
            </label>
            <label>
              Step therapy documented?
              <select
                value={stepTherapyDocumented}
                onChange={(e) => setStepTherapyDocumented(e.target.value)}
                disabled={!session}
              >
                <option value="">Not indicated</option>
                <option value="yes">Yes — trial/contraindication on file</option>
                <option value="no">No — not documented</option>
              </select>
            </label>
          </div>
          <button
            type="submit"
            disabled={!session || drafting || !drugName.trim() || !dosage.trim() || !diagnosisCode.trim()}
          >
            {drafting ? "Drafting…" : "Draft prior-auth request"}
          </button>
        </form>
        {draftError && <div className="error-banner">{draftError}</div>}
      </section>

      {draft && draft.checklist && draft.checklist.length > 0 && (
        <section className="panel">
          <h2>3. Coverage checklist</h2>
          <p className="checklist-note">
            Evaluated deterministically in code against structured plan criteria — the
            drafting model narrates these results, it does not re-derive them.
          </p>
          <table className="checklist-table">
            <thead>
              <tr>
                <th>Criterion</th>
                <th>Status</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody>
              {draft.checklist.map((item, i) => (
                <tr key={i} className={`status-${item.status}`}>
                  <td>{item.criterion}</td>
                  <td>
                    <span className={`status-pill status-${item.status}`}>
                      {STATUS_LABEL[item.status] || item.status}
                    </span>
                  </td>
                  <td>
                    {item.detail}
                    {item.citation && <span className="cite"> ({item.citation})</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {draft && (
        <section className="panel">
          <h2>4. Drafted request</h2>
          <div className="draft-content">{draft.draft}</div>

          {draft.citations && draft.citations.length > 0 && (
            <details className="sources" open>
              <summary>Retrieved plan clauses ({draft.citations.length})</summary>
              <ul>
                {draft.citations.map((c, j) => (
                  <li key={j}>
                    {c.source}:{c.start_line}-{c.end_line}{" "}
                    <span className="score">score {c.score.toFixed(3)}</span>
                  </li>
                ))}
              </ul>
            </details>
          )}

          {draft.tool_trace && draft.tool_trace.length > 0 && (
            <details className="sources">
              <summary>Tool calls ({draft.tool_trace.length})</summary>
              <ul>
                {draft.tool_trace.map((t, j) => (
                  <li key={j}>
                    <code>{t.tool}({JSON.stringify(t.args)})</code>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </section>
      )}
    </div>
  );
}
