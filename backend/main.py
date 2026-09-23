# main.py — FastAPI backend for Transfer Scout
# loads the trained ML model on startup
# exposes two endpoints: player search and value prediction

from fastapi import FastAPI, HTTPException    # FastAPI framework
from fastapi.middleware.cors import CORSMiddleware  # allow React to call this API
import pandas as pd                          # data manipulation
import joblib                                # load saved model
import os                                    # file paths
import numpy as np                           # numerical operations
import anthropic                          # Claude API
import os                                 # environment variables
from dotenv import load_dotenv            # reads .env file


load_dotenv()       # load .env file
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

app = FastAPI()

# allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# LOAD MODEL ON STARTUP
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # backend/ directory

print("Loading model...")
model = joblib.load(os.path.join(BASE_DIR, 'model.pkl'))    # trained linear regression
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))  # feature scaler

print("Loading player data...")
df = pd.read_csv(os.path.join(BASE_DIR, 'data/clean_players.csv'))  # clean player dataset
print(f"Loaded {len(df)} players")

# features must match exactly what we trained on
FEATURES = ['total_goals', 'total_assists', 'total_minutes', 'total_appearances', 'age', 'position_num']

# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"status": "Transfer Scout API is running", "players": len(df)}

# ─────────────────────────────────────────────
# PLAYER SEARCH
# ─────────────────────────────────────────────

@app.get("/players/search")
def search_players(name: str):
    # filter players by name — case insensitive partial match
    results = df[df['name'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('ascii').str.contains(
    name.encode('ascii', errors='ignore').decode('ascii'), case=False, na=False
)]

    if results.empty:
        raise HTTPException(status_code=404, detail="No players found")

    # return top 5 matches
    matches = results.head(5)[['player_id', 'name', 'position', 'current_club_name',
                                'market_value_in_eur', 'total_goals', 'total_assists',
                                'total_minutes', 'total_appearances', 'age',
                                'country_of_citizenship']].to_dict(orient='records')
    return {"players": matches}

# ─────────────────────────────────────────────
# PREDICT TRANSFER VALUE
# ─────────────────────────────────────────────

@app.get("/predict/{player_id}")
def predict_value(player_id: int):
    # find the player in our dataset
    player = df[df['player_id'] == player_id]

    if player.empty:
        raise HTTPException(status_code=404, detail="Player not found")

    player = player.iloc[0]  # get the first row as a series

    # extract features in the same order as training
    features = [[
        player['total_goals'],
        player['total_assists'],
        player['total_minutes'],
        player['total_appearances'],
        player['age'],
        player['position_num']
    ]]

    # scale features using the same scaler from training
    features_scaled = scaler.transform(features)

    # predict
    predicted_log = model.predict(features_scaled)[0]
    predicted_log = min(predicted_log, 25)             # cap to prevent overflow
    predicted_value = np.expm1(predicted_log)          # reverse log transform
    predicted_value = max(0, predicted_value)          # no negative values
    predicted_value = min(predicted_value, 500_000_000)  # cap at €500M

    actual_value = player['market_value_in_eur']

    # difference between predicted and actual
    difference = predicted_value - actual_value
    if difference > 0:
        verdict = f"Undervalued by €{abs(difference):,.0f}"
    elif difference < 0:
        verdict = f"Overvalued by €{abs(difference):,.0f}"
    else:
        verdict = "Fairly valued"

    # find similar players — same position, similar predicted value
    similar = df[
        (df['position'] == player['position']) &
        (df['player_id'] != player_id) &
        (df['market_value_in_eur'].notna())
    ].copy()

    similar['value_diff'] = abs(similar['market_value_in_eur'] - predicted_value)
    similar = similar.nsmallest(3, 'value_diff')[['name', 'current_club_name',
                                                   'market_value_in_eur',
                                                   'country_of_citizenship']].to_dict(orient='records')

    return {
        "player": {
            "name": player['name'],
            "position": player['position'],
            "club": player['current_club_name'],
            "age": round(player['age'], 1),
            "nationality": player['country_of_citizenship'],
            "stats": {
                "goals": int(player['total_goals']),
                "assists": int(player['total_assists']),
                "minutes": int(player['total_minutes']),
                "appearances": int(player['total_appearances'])
            }
        },
        "prediction": {
            "predicted_value": round(predicted_value),
            "actual_value": round(actual_value),
            "difference": round(difference),
            "verdict": verdict
        },
        "similar_players": similar
    }


# ─────────────────────────────────────────────
# CLAUDE AI SCOUT REPORT
# ─────────────────────────────────────────────

@app.get("/scout-report/{player_id}")
def scout_report(player_id: int):
    # find the player
    player = df[df['player_id'] == player_id]

    if player.empty:
        raise HTTPException(status_code=404, detail="Player not found")

    player = player.iloc[0]

    # get predicted value
    features = [[
        player['total_goals'],
        player['total_assists'],
        player['total_minutes'],
        player['total_appearances'],
        player['age'],
        player['position_num']
    ]]
    features_scaled = scaler.transform(features)
    predicted_log = model.predict(features_scaled)[0]
    predicted_log = min(predicted_log, 25)
    predicted_value = np.expm1(predicted_log)
    predicted_value = max(0, min(predicted_value, 500_000_000))

    actual_value = player['market_value_in_eur']
    difference = predicted_value - actual_value

    if difference > 0:
        verdict = f"undervalued by €{abs(difference):,.0f}"
    else:
        verdict = f"overvalued by €{abs(difference):,.0f}"

    # build prompt with real player data
    prompt = f"""You are an elite football scout writing a professional scouting report.

Player: {player['name']}
Club: {player['current_club_name']}
Position: {player['position']}
Age: {round(player['age'], 1)}
Nationality: {player['country_of_citizenship']}

Last 3 seasons statistics:
- Goals: {int(player['total_goals'])}
- Assists: {int(player['total_assists'])}
- Minutes played: {int(player['total_minutes'])}
- Appearances: {int(player['total_appearances'])}

ML Model assessment:
- Predicted market value: €{predicted_value:,.0f}
- Actual Transfermarkt value: €{actual_value:,.0f}
- Verdict: {verdict}

Write a 3-4 sentence professional scout report analyzing this player. 
mention their statistical output, what it means for their value, and one key insight.
Be specific, use the numbers, sound like a real football scout.
Do not use bullet points. Write in flowing prose. Keep it under 80 words."""

    # call Claude API
    message = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )

    return {"report": message.content[0].text}