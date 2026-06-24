import Badge from './Badge'

export default function CitationCard({ citation, verdict }) {
  return (
    <div className={`finding-card ${verdict?.flagged ? 'finding-card-flagged' : ''}`}>
      <div className="finding-header">
        <h3>{citation.case_name}</h3>
        {verdict?.flagged && <Badge value="flagged" tone="bad" />}
      </div>
      <p className="finding-citation-string">{citation.citation_string}</p>
      <p className="finding-proposition">"Cited for: {citation.proposition}"</p>
      {citation.quoted_text && <blockquote>{citation.quoted_text}</blockquote>}

      {verdict && (
        <div className="finding-verdict">
          <div className="finding-badges">
            <Badge value={verdict.support_status} />
            <Badge value={verdict.quote_accuracy} />
          </div>
          <p>{verdict.reasoning}</p>
          {verdict.confidence !== null && (
            <p className="finding-confidence">
              Confidence: <strong>{Math.round(verdict.confidence * 100)}%</strong> — {verdict.confidence_reasoning}
            </p>
          )}
        </div>
      )}
    </div>
  )
}
