import Badge from './Badge'

export default function FactCard({ fact, factCheck }) {
  return (
    <div className={`finding-card ${factCheck?.flagged ? 'finding-card-flagged' : ''}`}>
      <div className="finding-header">
        <h3>{fact.claim}</h3>
        {factCheck?.flagged && <Badge value="flagged" tone="bad" />}
      </div>

      {factCheck && (
        <div className="finding-verdict">
          <div className="finding-badges">
            <Badge value={factCheck.consistency_status} />
          </div>
          <p>{factCheck.reasoning}</p>
          {factCheck.confidence !== null && (
            <p className="finding-confidence">
              Confidence: <strong>{Math.round(factCheck.confidence * 100)}%</strong> — {factCheck.confidence_reasoning}
            </p>
          )}
        </div>
      )}
    </div>
  )
}
