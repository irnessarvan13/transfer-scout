# model.py — trains the transfer value prediction model
# run this file once to generate model.pkl
# FastAPI loads model.pkl on startup — no retraining on every request

import pandas as pd                          # data manipulation
import numpy as np                           # numerical operations
from sklearn.linear_model import LinearRegression  # the ML model
from sklearn.model_selection import train_test_split  # split data for training and testing
from sklearn.preprocessing import StandardScaler     # normalize features
import joblib                                # save and load the trained model
import os                                    # file paths

# ─────────────────────────────────────────────
# STEP 1 — Load the data
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # backend/ directory
DATA_DIR = os.path.join(BASE_DIR, 'data')              # backend/data/

print("Loading data...")
players = pd.read_csv(os.path.join(DATA_DIR, 'players.csv'))
appearances = pd.read_csv(os.path.join(DATA_DIR, 'appearances.csv'))

print(f"Players: {len(players)} rows")
print(f"Appearances: {len(appearances)} rows")

# ─────────────────────────────────────────────
# STEP 2 — Aggregate appearances per player (last 3 seasons only)
# ─────────────────────────────────────────────

print("Aggregating appearances...")

# convert date to datetime
appearances['date'] = pd.to_datetime(appearances['date'], errors='coerce')

# only use appearances from the last 3 seasons
cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=3)
recent_appearances = appearances[appearances['date'] >= cutoff_date]

print(f"Recent appearances (last 3 years): {len(recent_appearances)} rows")

player_stats = recent_appearances.groupby('player_id').agg(
    total_goals=('goals', 'sum'),
    total_assists=('assists', 'sum'),
    total_minutes=('minutes_played', 'sum'),
    total_appearances=('appearance_id', 'count')
).reset_index()

print(f"Aggregated stats for {len(player_stats)} players")
# ─────────────────────────────────────────────
# STEP 3 — Merge everything
# ─────────────────────────────────────────────

print("Merging data...")
df = players.merge(player_stats, on='player_id', how='inner')  # inner join — only players with appearances
print(f"Merged dataset: {len(df)} players")

# ─────────────────────────────────────────────
# STEP 4 — Clean the data
# ─────────────────────────────────────────────

print("Cleaning data...")

# drop players with no market value — can't train without a target
df = df.dropna(subset=['market_value_in_eur'])

# calculate age from date_of_birth
df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
df['age'] = ((pd.Timestamp.now() - df['date_of_birth']).dt.days / 365.25).astype(float)

# drop players with no age
df = df.dropna(subset=['age'])

# convert position to numbers — model can't understand strings
position_map = {
    'Attack': 4,       # forwards have highest value potential
    'Midfield': 3,
    'Defender': 2,
    'Goalkeeper': 1,
    'Missing': 0       # unknown position
}
df['position_num'] = df['position'].map(position_map).fillna(0)

# drop obvious outliers — players with 0 minutes or unrealistic values
df = df[df['total_minutes'] > 0]
df = df[df['market_value_in_eur'] > 0]

print(f"Clean dataset: {len(df)} players")
print(f"Market value range: €{df['market_value_in_eur'].min():,.0f} to €{df['market_value_in_eur'].max():,.0f}")

# ─────────────────────────────────────────────
# STEP 5 — Train the model
# ─────────────────────────────────────────────

print("Training model...")

# features — what we feed into the model
features = ['total_goals', 'total_assists', 'total_minutes', 'total_appearances', 'age', 'position_num']
X = df[features]

# target — what we want to predict
y = df['market_value_in_eur']

# split 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale features — puts all numbers on the same scale so no feature dominates
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on training data only
X_test_scaled = scaler.transform(X_test)          # transform test data with same scaler

# train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# evaluate — R² score: 1.0 = perfect, 0.0 = random, negative = worse than random
score = model.score(X_test_scaled, y_test)
print(f"Model R² score: {score:.3f}")

# show feature importance — which stats matter most
feature_importance = pd.DataFrame({
    'feature': features,
    'coefficient': model.coef_
}).sort_values('coefficient', ascending=False)
print("\nFeature importance:")
print(feature_importance)

# ─────────────────────────────────────────────
# STEP 6 — Save the model
# ─────────────────────────────────────────────

print("\nSaving model...")
model_path = os.path.join(BASE_DIR, 'model.pkl')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')

joblib.dump(model, model_path)    # save the trained model
joblib.dump(scaler, scaler_path)  # save the scaler — need it to scale predictions too

print(f"Model saved to {model_path}")
print(f"Scaler saved to {scaler_path}")
print("Done!")

# also save the clean dataframe for player search
df_path = os.path.join(DATA_DIR, 'clean_players.csv')
df[['player_id', 'name', 'position', 'position_num', 'age', 'current_club_name',
    'market_value_in_eur', 'total_goals', 'total_assists', 'total_minutes',
    'total_appearances', 'country_of_citizenship']].to_csv(df_path, index=False)
print(f"Clean player data saved to {df_path}")