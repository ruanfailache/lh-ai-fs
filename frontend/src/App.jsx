import { useState } from 'react'
import { analyzeCase } from './api'
import JudicialMemoCard from './components/JudicialMemoCard'
import SummaryBar from './components/SummaryBar'
import CitationCard from './components/CitationCard'
import FactCard from './components/FactCard'
import './App.css'

function App() {
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const runAnalysis = async () => {
    setLoading(true)
    setError(null)
    setReport(null)

    try {
      const data = await analyzeCase()
      setReport(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const verdictByCitationId = Object.fromEntries(
    (report?.verdicts ?? []).map((v) => [v.citation_id, v])
  )
  const factCheckByFactId = Object.fromEntries(
    (report?.fact_checks ?? []).map((fc) => [fc.fact_id, fc])
  )

  return (
    <div className="page">
      <h1>BS Detector</h1>
      <p className="subtitle">Legal brief verification pipeline</p>

      <button onClick={runAnalysis} disabled={loading} className="run-button">
        {loading ? 'Analyzing...' : 'Run Analysis'}
      </button>

      {error && (
        <div className="error-banner">
          <strong>Error:</strong> {error}
        </div>
      )}

      {report && (
        <div className="report">
          <JudicialMemoCard memo={report.judicial_memo} />
          <SummaryBar report={report} />

          <section>
            <h2>Citations</h2>
            {report.citations.length === 0 && <p className="empty-state">No citations were found.</p>}
            {report.citations.map((citation) => (
              <CitationCard key={citation.id} citation={citation} verdict={verdictByCitationId[citation.id]} />
            ))}
          </section>

          <section>
            <h2>Facts</h2>
            {report.facts.length === 0 && <p className="empty-state">No facts were found.</p>}
            {report.facts.map((fact) => (
              <FactCard key={fact.id} fact={fact} factCheck={factCheckByFactId[fact.id]} />
            ))}
          </section>
        </div>
      )}

      {report === null && !loading && !error && (
        <p className="empty-state">Click "Run Analysis" to analyze the case documents.</p>
      )}
    </div>
  )
}

export default App
