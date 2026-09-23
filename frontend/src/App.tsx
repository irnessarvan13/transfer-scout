/*
App.tsx — Transfer Scout main component
1. Search box with debounced API calls
2. Dropdown showing matching players
3. Prediction results when player is selected
*/

import { useState, useEffect, useRef } from 'react'
import { searchPlayers, predictValue, getScoutReport } from './api/transfer'
import type { Player, PredictionResponse } from './api/transfer'
import './App.css'

// converts nationality to flag emoji — same as PitchIQ
function getFlag(nationality: string): string {
  const flags: Record<string, string> = {
    'France': '🇫🇷', 'England': '🏴󠁧󠁢󠁥󠁮󠁧󠁿', 'Portugal': '🇵🇹',
    'Germany': '🇩🇪', 'Spain': '🇪🇸', 'Brazil': '🇧🇷',
    'Argentina': '🇦🇷', 'Netherlands': '🇳🇱', 'Belgium': '🇧🇪',
    'Italy': '🇮🇹', 'Denmark': '🇩🇰', 'Norway': '🇳🇴',
    'Sweden': '🇸🇪', 'Croatia': '🇭🇷', 'Poland': '🇵🇱',
    'Senegal': '🇸🇳', 'Morocco': '🇲🇦', 'Colombia': '🇨🇴',
    'Uruguay': '🇺🇾', 'Austria': '🇦🇹', 'Switzerland': '🇨🇭',
    'Scotland': '🏴󠁧󠁢󠁳󠁣󠁴󠁿', 'Wales': '🏴󠁧󠁢󠁷󠁬󠁳󠁿', 'Ireland': '🇮🇪',
    'Turkey': '🇹🇷', 'Ukraine': '🇺🇦', 'Japan': '🇯🇵',
    'South Korea': '🇰🇷', 'Ghana': '🇬🇭', 'Nigeria': '🇳🇬',
    'Ivory Coast': '🇨🇮', 'Algeria': '🇩🇿', 'Egypt': '🇪🇬',
    'Bosnia-Herzegovina': '🇧🇦', 'Serbia': '🇷🇸', 'Slovenia': '🇸🇮',
    'Slovakia': '🇸🇰', 'Czech Republic': '🇨🇿', 'Romania': '🇷🇴',
    'Greece': '🇬🇷', 'Mexico': '🇲🇽', 'United States': '🇺🇸',
    'Canada': '🇨🇦', 'Australia': '🇦🇺', 'China': '🇨🇳',
  }
  return flags[nationality] || '🌍'
}

// format value in millions — €180,000,000 → €180M
function formatValue(value: number): string {
  if (value >= 1_000_000) return `€${(value / 1_000_000).toFixed(1)}M`
  if (value >= 1_000) return `€${(value / 1_000).toFixed(0)}K`
  return `€${value.toFixed(0)}`
}

function App() {
  const [query, setQuery] = useState('')                          // search input value
  const [results, setResults] = useState<Player[]>([])           // search dropdown results
  const [loading, setLoading] = useState(false)                  // search loading state
  const [predicting, setPredicting] = useState(false)            // prediction loading state
  const [scoutReport, setScoutReport] = useState<string | null>(null)  // Claude AI scout report
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)  // prediction result
  const [selectedPlayer, setSelectedPlayer] = useState<Player | null>(null)      // selected player
  const [showDropdown, setShowDropdown] = useState(false)        // show/hide dropdown
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)         // debounce timer

  // debounced search — waits 500ms after typing before calling API
  useEffect(() => {
    if (query.length < 2) {
      setResults([])
      setShowDropdown(false)
      return
    }

    if (debounceRef.current) clearTimeout(debounceRef.current)   // clear previous timer

    debounceRef.current = setTimeout(async () => {
      setLoading(true)
      try {
        const players = await searchPlayers(query)
        setResults(players)
        setShowDropdown(true)
      } catch {
        setResults([])
      } finally {
        setLoading(false)
      }
    }, 500)                                                       // 500ms debounce
  }, [query])

  // called when user clicks a player from dropdown
async function handleSelectPlayer(player: Player) {
    setSelectedPlayer(player)
    setShowDropdown(false)
    setQuery(player.name)
    setPredicting(true)
    setPrediction(null)
    setScoutReport(null)                                   // reset previous report

    try {
      const [result, report] = await Promise.all([
        predictValue(player.player_id),                    // fetch prediction
        getScoutReport(player.player_id)                   // fetch scout report simultaneously
      ])
      setPrediction(result)
      setScoutReport(report)
    } catch {
      console.error('Prediction failed')
    } finally {
      setPredicting(false)
    }
  }

  // reset — go back to search
  function handleReset() {
    setQuery('')
    setResults([])
    setPrediction(null)
    setSelectedPlayer(null)
    setShowDropdown(false)
    setScoutReport(null)                               // reset scout report
  }

  return (
    <div className="app">

      {/* Header */}
      <div className="header">
        <div className="header-title">
          <span className="header-icon">⚽</span>
          <h1>Transfer Scout</h1>
          <span className="header-badge">ML Powered</span>
        </div>
        <p className="header-subtitle">
          Find out what any player is really worth — powered by machine learning
          trained on 50,000+ Transfermarkt valuations.
        </p>
      </div>

      {/* Search */}
      <div className="search-container">
        <div className="search-box">
          <span className="search-icon">🔍</span>
          <input
            type="text"
            placeholder="Search any player — Mbappe, Haaland, Saka..."
            value={query}
            onChange={e => setQuery(e.target.value)}
            className="search-input"
          />
          {query && (
            <button className="search-clear" onClick={handleReset}>✕</button>
          )}
        </div>

        {/* Dropdown */}
        {showDropdown && results.length > 0 && (
          <div className="dropdown">
            {results.map(player => (
              <div
                key={player.player_id}
                className="dropdown-item"
                onClick={() => handleSelectPlayer(player)}
              >
                <div className="dropdown-flag">{getFlag(player.country_of_citizenship)}</div>
                <div className="dropdown-info">
                  <div className="dropdown-name">{player.name}</div>
                  <div className="dropdown-meta">{player.current_club_name} · {player.position}</div>
                </div>
                <div className="dropdown-value">{formatValue(player.market_value_in_eur)}</div>
              </div>
            ))}
          </div>
        )}

        {loading && <div className="search-loading">Searching...</div>}
      </div>

      {/* Predicting loading state */}
      {predicting && (
        <div className="predicting">
          <div className="predicting-text">Analyzing {selectedPlayer?.name}...</div>
        </div>
      )}

      {/* Prediction Results */}
      {prediction && !predicting && (
        <div className="results">

          {/* Player Header */}
          <div className="player-header card">
            <div className="player-avatar">
              {prediction.player.name.split(' ').map(n => n[0]).join('').slice(0, 2)}
            </div>
            <div className="player-info">
              <h2>{prediction.player.name} {getFlag(prediction.player.nationality)}</h2>
              <p>{prediction.player.club} · {prediction.player.position} · Age {prediction.player.age}</p>
            </div>
            <button className="reset-btn" onClick={handleReset}>← New search</button>
          </div>

          {/* Value Comparison */}
          <div className="value-grid">
            <div className="value-card card">
              <div className="value-label">Model prediction</div>
              <div className="value-amount predicted">{formatValue(prediction.prediction.predicted_value)}</div>
            </div>
            <div className="value-card card">
              <div className="value-label">Transfermarkt value</div>
              <div className="value-amount actual">{formatValue(prediction.prediction.actual_value)}</div>
            </div>
          </div>

          {/* Verdict */}
          <div className={`verdict card ${prediction.prediction.difference > 0 ? 'undervalued' : 'overvalued'}`}>
            <span className="verdict-icon">{prediction.prediction.difference > 0 ? '📈' : '📉'}</span>
            <span className="verdict-text">{prediction.prediction.verdict}</span>
          </div>

          {/* Stats */}
          <div className="stats-card card">
            <h3>Season stats (last 3 seasons)</h3>
            <div className="stats-grid">
              <div className="stat">
                <div className="stat-value">{prediction.player.stats.goals}</div>
                <div className="stat-label">Goals</div>
              </div>
              <div className="stat">
                <div className="stat-value">{prediction.player.stats.assists}</div>
                <div className="stat-label">Assists</div>
              </div>
              <div className="stat">
                <div className="stat-value">{prediction.player.stats.minutes.toLocaleString()}</div>
                <div className="stat-label">Minutes</div>
              </div>
              <div className="stat">
                <div className="stat-value">{prediction.player.stats.appearances}</div>
                <div className="stat-label">Apps</div>
              </div>
            </div>
          </div>

          {/* AI Scout Report */}
          {scoutReport && (
            <div className="scout-card card">
              <div className="scout-header">
                <span className="scout-icon">🤖</span>
                <h3>AI Scout Report</h3>
              </div>
              <p className="scout-text">{scoutReport}</p>
            </div>
          )}

          {/* Similar Players */}
          <div className="similar-card card">
            <h3>Similar players at this value</h3>
            {prediction.similar_players.map((player, i) => (
              <div key={i} className="similar-player">
                <div className="similar-flag">{getFlag(player.country_of_citizenship)}</div>
                <div className="similar-info">
                  <div className="similar-name">{player.name}</div>
                  <div className="similar-club">{player.current_club_name}</div>
                </div>
                <div className="similar-value">{formatValue(player.market_value_in_eur)}</div>
              </div>
            ))}
          </div>

        </div>
      )}
    </div>
  )
}

export default App