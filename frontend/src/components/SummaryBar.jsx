export default function SummaryBar({ report }) {
  const { citations, facts, citation_flagged_count, fact_flagged_count, errors } = report

  return (
    <div className="summary-bar">
      <div className="summary-stat">
        <span className="summary-number">{citations.length}</span>
        <span className="summary-label">citations checked</span>
      </div>
      <div className="summary-stat">
        <span className="summary-number summary-number-flagged">{citation_flagged_count}</span>
        <span className="summary-label">citations flagged</span>
      </div>
      <div className="summary-stat">
        <span className="summary-number">{facts.length}</span>
        <span className="summary-label">facts checked</span>
      </div>
      <div className="summary-stat">
        <span className="summary-number summary-number-flagged">{fact_flagged_count}</span>
        <span className="summary-label">facts flagged</span>
      </div>

      {errors.length > 0 && (
        <div className="error-banner">
          <strong>{errors.length} pipeline issue{errors.length > 1 ? 's' : ''} occurred</strong> — the
          results above may be partial.
          <ul>
            {errors.map((err, i) => (
              <li key={i}>{err}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
