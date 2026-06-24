export default function JudicialMemoCard({ memo }) {
  if (!memo) return null

  return (
    <div className="memo-card">
      <h2>Memo to the Court</h2>
      <p>{memo}</p>
    </div>
  )
}
