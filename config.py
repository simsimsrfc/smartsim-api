"""
Smart Sim — Configuration centrale
API-FOOTBALL PRO (7 500 req/jour)
"""

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent

# Charger .env si présent (local), sinon les secrets viennent de l'environnement (HF)
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass  # python-dotenv optionnel en production HF

# ──────────────────────────────────────────────
# API — Clé lue depuis le secret HF FOOTBALL_API_KEY ou le .env local
# ──────────────────────────────────────────────
_raw_key = os.environ.get("FOOTBALL_API_KEY", "")
API_KEY = _raw_key.strip() if _raw_key else None
API_HOST = "v3.football.api-sports.io"
BASE_URL = f"https://{API_HOST}"
HEADERS = {
    "x-apisports-key": API_KEY,
}

# ──────────────────────────────────────────────
# QUOTAS & CACHING
# ──────────────────────────────────────────────
DAILY_QUOTA = 7500   # Plan Pro API-Football : 7 500 requêtes/jour (vrai quota)
CACHE_DIR = str(_PROJECT_ROOT / "cache")
CACHE_TTL_FIXTURES = 3600        # 1 h  — matchs du jour
CACHE_TTL_STATS = 86400          # 24 h — stats historiques (changent rarement)
CACHE_TTL_ODDS = 1800            # 30 min — cotes (sensibles au timing)
CACHE_TTL_REFEREE = 604800       # 7 j  — profil arbitre

# Batching : nb max de matchs traités par run (75k quota = ~500 matchs/jour)
BATCH_SIZE = 200

# ──────────────────────────────────────────────
# LIGUES — 34 compétitions (MONDIAL + COUPES D'EUROPE)
# Format : league_id (API-Football) → metadata
# ──────────────────────────────────────────────
LEAGUES = {
    # ── FRANCE ──
    61:  {"name": "Ligue 1",           "country": "France",      "season": 2026, "flag": "🇫🇷"},
    62:  {"name": "Ligue 2",           "country": "France",      "season": 2026, "flag": "🇫🇷"},
    63:  {"name": "National 1",        "country": "France",      "season": 2026, "flag": "🇫🇷"},

    # ── ANGLETERRE ──
    39:  {"name": "Premier League",    "country": "England",     "season": 2026, "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
    40:  {"name": "Championship",      "country": "England",     "season": 2026, "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
    41:  {"name": "League One",        "country": "England",     "season": 2026, "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
    42:  {"name": "League Two",        "country": "England",     "season": 2026, "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
    45:  {"name": "National League",   "country": "England",     "season": 2026, "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿"},

    # ── ESPAGNE ──
    140: {"name": "La Liga",           "country": "Spain",       "season": 2026, "flag": "🇪🇸"},
    141: {"name": "Segunda División",  "country": "Spain",       "season": 2026, "flag": "🇪🇸"},

    # ── ITALIE ──
    135: {"name": "Serie A",           "country": "Italy",       "season": 2026, "flag": "🇮🇹"},
    136: {"name": "Serie B",           "country": "Italy",       "season": 2026, "flag": "🇮🇹"},

    # ── ALLEMAGNE ──
    78:  {"name": "Bundesliga",        "country": "Germany",     "season": 2026, "flag": "🇩🇪"},
    79:  {"name": "2. Bundesliga",     "country": "Germany",     "season": 2026, "flag": "🇩🇪"},

    # ── PORTUGAL ──
    94:  {"name": "Primeira Liga",     "country": "Portugal",    "season": 2026, "flag": "🇵🇹"},

    # ── PAYS-BAS ──
    88:  {"name": "Eredivisie",        "country": "Netherlands", "season": 2026, "flag": "🇳🇱"},

    # ── SUISSE ──
    207: {"name": "Super League",      "country": "Switzerland", "season": 2026, "flag": "🇨🇭"},

    # ── TURQUIE ──
    203: {"name": "Süper Lig",         "country": "Turkey",      "season": 2026, "flag": "🇹🇷"},

    # ── SLOVÉNIE ──
    373: {"name": "Prva Liga",         "country": "Slovenia",    "season": 2026, "flag": "🇸🇮"},

    # ── SLOVAQUIE ──
    332: {"name": "Super Liga",        "country": "Slovakia",    "season": 2026, "flag": "🇸🇰"},

    # ── BRÉSIL (saison calendaire 2026) ──
    71:  {"name": "Série A",           "country": "Brazil",      "season": 2026, "flag": "🇧🇷"},

    # ── JAPON (saison calendaire 2026) ──
    98:  {"name": "J1 League",         "country": "Japan",       "season": 2026, "flag": "🇯🇵"},
    99:  {"name": "J2 League",         "country": "Japan",       "season": 2026, "flag": "🇯🇵"},

    # ── AUSTRALIE (saison calendaire 2025-26) ──
    188: {"name": "A-League",          "country": "Australia",   "season": 2026, "flag": "🇦🇺"},

    # ── MEXIQUE (saison calendaire 2026) ──
    262: {"name": "Liga MX",           "country": "Mexico",      "season": 2026, "flag": "🇲🇽"},

    # ── BELGIQUE ──
    144: {"name": "Pro League",        "country": "Belgium",    "season": 2026, "flag": "🇧🇪"},
    145: {"name": "Challenger Pro League", "country": "Belgium", "season": 2026, "flag": "🇧🇪"},

    # ── GRÈCE ──
    197: {"name": "Super League",      "country": "Greece",     "season": 2026, "flag": "🇬🇷"},

    # ── CROATIE ──
    210: {"name": "HNL",              "country": "Croatia",     "season": 2026, "flag": "🇭🇷"},

    # ── COUPES D'EUROPE ──
    2:   {"name": "Champions League",         "country": "Europe", "season": 2026, "flag": "🏆"},
    3:   {"name": "Europa League",            "country": "Europe", "season": 2026, "flag": "🏆"},
    848: {"name": "Europa Conference League", "country": "Europe", "season": 2026, "flag": "🏆"},
}

# IDs des coupes européennes (double analyse : compétition + forme)
EUROPEAN_CUP_IDS = {2, 3, 848}

# Nombre de saisons passées à analyser pour l'historique européen
EURO_CUP_HISTORY_SEASONS = 3

# ──────────────────────────────────────────────
# LINEUPS & ABSENCES (pour matchs à venir)
# ──────────────────────────────────────────────
FETCH_LINEUPS = True
CACHE_TTL_LINEUPS = 3600  # 1 h

# ──────────────────────────────────────────────
# PARAMÈTRES DU MODÈLE
# ──────────────────────────────────────────────
LAST_N_MATCHES = 10          # Historique récent par équipe
H2H_LIMIT = 10               # Confrontations directes
MIN_CONFIDENCE = 0.55         # Seuil minimal pour afficher une prédiction
SMART_BET_THRESHOLD = 0.70    # Seuil "SMART BET" (IA + xG convergent)

# Historique affiché : on conserve les anciennes lignes Supabase,
# mais l'API /api/history ne retourne que les analyses créées à partir de ce point.
HISTORY_START_DATE = os.environ.get(
    "HISTORY_START_DATE",
    "2026-05-12T20:54:11+02:00",
).strip()

# ──────────────────────────────────────────────
# BOOKMAKERS (pour cotes Over 2.5)
# IDs API-Football
# ──────────────────────────────────────────────
BOOKMAKERS = {
    6:  "Bet365",
    11: "1xBet",
    8:  "Unibet",
    1:  "Bwin",
    5:  "William Hill",
}
# On récupère les 3 premiers disponibles pour chaque match
BOOKMAKERS_MIN = 3

# ──────────────────────────────────────────────
# STATS À EXTRAIRE (mapping API-Football)
# ──────────────────────────────────────────────
STATS_KEYS = [
    # Offensive
    "expected_goals",
    "Shots on Goal",
    "Shots insidebox",
    "Total Shots",

    # Possession & Pression
    "Ball Possession",
    "Passes %",
    "Corner Kicks",
    "Offsides",

    # Défense
    "Goalkeeper Saves",
    "Blocked Shots",
    "Tackles",
    "Interceptions",

    # Discipline
    "Fouls",
    "Yellow Cards",
    "Red Cards",
]

# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
APP_TITLE = "Smart Sim"
APP_ICON = "S"
APP_LAYOUT = "wide"
THEME_PRIMARY = "#0C1014"      # Dark profond (fond principal)
THEME_ACCENT = "#22C55E"       # Vert emerald (Smart Sim signature)
THEME_ACCENT_GLOW = "#16A34A"  # Vert foncé (hover/glow)
THEME_GOLD = "#FBBF24"         # Or (highlights premium)
THEME_DANGER = "#EF4444"       # Rouge alerte
THEME_TEXT = "#FFFFFF"
THEME_CARD_BG = "#161A20"      # Anthracite mat (cartes)
THEME_SECONDARY_TEXT = "#94A3B8"  # Gris bleuté (texte secondaire)
