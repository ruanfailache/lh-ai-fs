const API_BASE_URL = 'http://localhost:8002'

export async function analyzeCase() {
  const response = await fetch(`${API_BASE_URL}/analyze`, { method: 'POST' })

  if (!response.ok) {
    throw new Error(`Server responded with ${response.status}`)
  }

  return response.json()
}
