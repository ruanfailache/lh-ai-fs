const TONE_BY_VALUE = {
  // citation support_status
  supports: 'ok',
  contradicts: 'bad',
  unsupported: 'bad',
  // citation quote_accuracy
  accurate: 'ok',
  altered: 'bad',
  fabricated: 'bad',
  no_quote: 'neutral',
  // fact consistency_status
  consistent: 'ok',
  contradicted: 'bad',
  unverifiable: 'neutral',
  // shared
  uncertain: 'neutral',
}

function labelFor(value) {
  return value.replaceAll('_', ' ')
}

export default function Badge({ value, tone }) {
  const resolvedTone = tone ?? TONE_BY_VALUE[value] ?? 'neutral'
  return <span className={`badge badge-${resolvedTone}`}>{labelFor(value)}</span>
}
