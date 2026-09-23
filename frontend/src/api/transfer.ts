import axios from 'axios'

const API_BASE = '/api'

export interface Player {
  player_id: number
  name: string
  position: string
  current_club_name: string
  market_value_in_eur: number
  total_goals: number
  total_assists: number
  total_minutes: number
  total_appearances: number
  age: number
  country_of_citizenship: string
}

export interface SimilarPlayer {
  name: string
  current_club_name: string
  market_value_in_eur: number
  country_of_citizenship: string
}

export interface PredictionResponse {
  player: {
    name: string
    position: string
    club: string
    age: number
    nationality: string
    stats: {
      goals: number
      assists: number
      minutes: number
      appearances: number
    }
  }
  prediction: {
    predicted_value: number
    actual_value: number
    difference: number
    verdict: string
  }
  similar_players: SimilarPlayer[]
}

export async function searchPlayers(name: string): Promise<Player[]> {
  const response = await axios.get(`${API_BASE}/players/search`, {
    params: { name }
  })
  return response.data.players
}

export async function predictValue(playerId: number): Promise<PredictionResponse> {
  const response = await axios.get(`${API_BASE}/predict/${playerId}`)
  return response.data
}

// get AI scout report for a player
export async function getScoutReport(playerId: number): Promise<string> {
  const response = await axios.get(`${API_BASE}/scout-report/${playerId}`)
  return response.data.report
}