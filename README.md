# Transfer Scout ⚽

An ML-powered football player transfer value predictor. Search any player, get an instant market valuation powered by a Gradient Boosting model trained on 50,000+ Transfermarkt valuations, plus an AI-generated scout report from Claude.

🔗 **Live Demo:** http://13.52.240.231:3000

---

## Features

- **Player Search** — search any player by name with live dropdown results and accent handling
- **ML Valuation** — Gradient Boosting model predicts transfer value based on goals, assists, minutes, appearances, age, and position
- **Undervalued / Overvalued indicator** — compares model prediction vs actual Transfermarkt value
- **AI Scout Report** — Claude API generates a professional scouting paragraph analyzing the player's stats and value
- **Similar Players** — shows 3 players with similar predicted values at the same position
- **Season Stats** — displays last 3 seasons of aggregated statistics

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + TypeScript + Vite |
| Backend | Python + FastAPI |
| ML Model | scikit-learn Gradient Boosting Regressor |
| Data Processing | pandas |
| AI | Claude API (claude-sonnet-4-6) |
| Data Source | Transfermarkt dataset (Kaggle) |
| Deployment | AWS EC2 |
| CI/CD | GitHub Actions |
| Containerization | Docker + docker-compose |

---

## How It Works

### Data Pipeline
1. Load 3 CSV files from Transfermarkt — players, appearances, valuations
2. Aggregate 1.9M appearance records into per-player stats for last 3 seasons
3. Merge with player data to get name, position, age, market value
4. Clean — remove nulls, calculate age, encode positions

### ML Model
- **Algorithm:** Gradient Boosting Regressor (300 estimators, learning rate 0.05)
- **Features:** total goals, assists, minutes, appearances, age, position
- **Target:** log-transformed market value (log transform improves R² significantly)
- **R² Score:** 0.533
- **Improvement path:** started with Linear Regression (R² 0.147 on career data) → switched to 3-season window (R² 0.415) → log transform (R² 0.468) → Gradient Boosting (R² 0.533)

### Feature Importance
| Feature | Importance |
|---|---|
| Appearances | 44% |
| Age | 29% |
| Minutes | 18% |
| Goals | 5% |
| Assists | 2% |
| Position | 1% |

### API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| GET | `/players/search?name=Mbappe` | Search players by name |
| GET | `/predict/{player_id}` | Get ML prediction for a player |
| GET | `/scout-report/{player_id}` | Get Claude AI scout report |

---

## Why the Model Undervalues Elite Players

The model consistently undervalues superstars like Haaland and Mbappe. This is intentional and insightful — their actual market values include brand equity, commercial value, and global reputation that no statistical model can capture. The gap between predicted and actual value IS the insight.

---

## Running Locally

```bash
git clone https://github.com/irnessarvan13/transfer-scout.git
cd transfer-scout

# create .env
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# add data files (not in repo — too large)
# place model.pkl, scaler.pkl in backend/
# place clean_players.csv in backend/data/

# run with Docker
docker-compose up --build
```

Visit `http://localhost:3000`

---

## Author

Irnes Sarvan — CS Graduate, CSUEB  
[GitHub](https://github.com/irnessarvan13) | [LinkedIn](https://www.linkedin.com/in/irnes-sarvan-b85a52191/)
