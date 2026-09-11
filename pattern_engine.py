"""Pattern engine — encode recurring cross-signal patterns as prediction adjustments.

Every rule takes the full match dict (already enriched by data_fetcher) and
returns zero or one PatternHit:

    PatternHit(
        name        : short slug (e.g. "squad_asymmetry"),
        reason      : human-readable French sentence,
        lam_home_mul: multiplicative adjustment on home Poisson lambda,
        lam_away_mul: multiplicative adjustment on away Poisson lambda,
        winner_bias : additive log-odds nudge on the 1X2 vector
                      (+ favours home, - favours away, 0 neutral),
        confidence  : 0..1 how much we trust the rule for this match,
    )

`apply_patterns(match, lam_home, lam_away, probs_1x2)` scans all rules,
combines the hits, and returns adjusted (lam_home, lam_away, probs_1x2, hits).

Design principles:
    - Multipliers are BOUNDED (0.85..1.15 typical, 0.75..1.25 max).
    - Winner bias is a small logit shift capped at ±0.35 total.
    - Multiple hits compound multiplicatively for lambdas, additively for bias
      but total absolute bias is clamped once at the end.
    - All rules degrade gracefully to no-op when their input signals are missing.
"""
from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

log = logging.getLogger("SmartSim.patterns")

# ────────────────────────────────────────────────────────────
# Types
# ────────────────────────────────────────────────────────────
@dataclass
class PatternHit:
    name: str
    reason: str
    lam_home_mul: float = 1.0
    lam_away_mul: float = 1.0
    winner_bias: float = 0.0
    confidence: float = 1.0

    def scaled(self) -> "PatternHit":
        """Blend the hit by its confidence (mul → 1 at conf=0, bias → 0 at conf=0)."""
        c = max(0.0, min(1.0, self.confidence))
        return PatternHit(
            name=self.name,
            reason=self.reason,
            lam_home_mul=1.0 + (self.lam_home_mul - 1.0) * c,
            lam_away_mul=1.0 + (self.lam_away_mul - 1.0) * c,
            winner_bias=self.winner_bias * c,
            confidence=c,
        )


# ────────────────────────────────────────────────────────────
# Helpers to safely pull nested fields
# ────────────────────────────────────────────────────────────
def _safe(v, cast=float, default=None):
    try:
        if v is None: return default
        return cast(v)
    except (TypeError, ValueError):
        return default


def _form_wdl(form_str: str) -> dict:
    """Parse standing.form string like 'WWDLW' into {w, d, l, last3, streak_w, streak_l}."""
    if not form_str:
        return {"w": 0, "d": 0, "l": 0, "last3_pts": 0, "streak_w": 0, "streak_l": 0, "n": 0}
    s = form_str.strip().upper()[-5:]  # keep 5 most recent
    w = s.count("W"); d = s.count("D"); l = s.count("L")
    last3 = s[-3:] if len(s) >= 3 else s
    last3_pts = last3.count("W") * 3 + last3.count("D")
    # streaks are anchored at the MOST RECENT match (right-most char)
    streak_w = 0
    for ch in reversed(s):
        if ch == "W": streak_w += 1
        else: break
    streak_l = 0
    for ch in reversed(s):
        if ch == "L": streak_l += 1
        else: break
    return {"w": w, "d": d, "l": l, "last3_pts": last3_pts,
            "streak_w": streak_w, "streak_l": streak_l, "n": len(s)}


def _per_game(x, n):
    n = _safe(n, int, 0) or 0
    x = _safe(x, float, 0.0) or 0.0
    return x / n if n > 0 else None


# ════════════════════════════════════════════════════════════
# INDIVIDUAL RULES — each returns None or a PatternHit
# ════════════════════════════════════════════════════════════

def rule_squad_asymmetry(m: dict) -> Optional[PatternHit]:
    """L'exemple du user : équipe A a des absents lourds, équipe B non → shift vers B."""
    h_w = _safe(m.get("home_missing_weighted"), float, 0.0)
    a_w = _safe(m.get("away_missing_weighted"), float, 0.0)
    h_n = _safe(m.get("home_missing_players"), int, 0)
    a_n = _safe(m.get("away_missing_players"), int, 0)
    if h_w is None or a_w is None:
        return None
    gap_w = h_w - a_w
    gap_n = h_n - a_n
    # Threshold: at least 0.4 weighted diff AND 2+ raw player diff to trigger
    if abs(gap_w) < 0.4 or abs(gap_n) < 2:
        return None
    # Direction: positive gap → home more damaged → bias against home
    strength = min(1.0, abs(gap_w) / 1.5)  # 0.4 → 0.27 conf, 1.5+ → 1.0 conf
    if gap_w > 0:
        return PatternHit(
            name="squad_asymmetry_favours_away",
            reason=f"Domicile amputé (poids d'absences {h_w:.1f} vs {a_w:.1f}) — banc adverse plus complet",
            lam_home_mul=1.0 - 0.08 * strength,
            lam_away_mul=1.0 + 0.03 * strength,
            winner_bias=-0.28 * strength,
            confidence=strength,
        )
    return PatternHit(
        name="squad_asymmetry_favours_home",
        reason=f"Extérieur amputé (poids d'absences {a_w:.1f} vs {h_w:.1f}) — banc du domicile plus complet",
        lam_home_mul=1.0 + 0.03 * strength,
        lam_away_mul=1.0 - 0.08 * strength,
        winner_bias=+0.28 * strength,
        confidence=strength,
    )


def rule_standing_gap(m: dict) -> Optional[PatternHit]:
    """Écart au classement : top-tier vs bottom-tier → shift ferme."""
    hs = m.get("home_standing") or {}
    as_ = m.get("away_standing") or {}
    h_rank = _safe(hs.get("rank"), int)
    a_rank = _safe(as_.get("rank"), int)
    league_meta = m.get("league_meta") or {}
    total = _safe(league_meta.get("teams_count"), int) or 20
    if h_rank is None or a_rank is None:
        return None
    gap = a_rank - h_rank  # positive → home is better ranked
    # Only trigger meaningful gap (>= 30% of table size)
    if abs(gap) < max(4, total * 0.30):
        return None
    strength = min(1.0, abs(gap) / (total * 0.6))
    bias = 0.25 * strength * (1 if gap > 0 else -1)
    lam_fav = 1.0 + 0.06 * strength
    lam_dog = 1.0 - 0.05 * strength
    if gap > 0:
        return PatternHit(
            name="standing_home_stronger",
            reason=f"Écart de classement : dom. #{h_rank} vs ext. #{a_rank}",
            lam_home_mul=lam_fav, lam_away_mul=lam_dog, winner_bias=bias, confidence=strength,
        )
    return PatternHit(
        name="standing_away_stronger",
        reason=f"Écart de classement : ext. #{a_rank} vs dom. #{h_rank}",
        lam_home_mul=lam_dog, lam_away_mul=lam_fav, winner_bias=bias, confidence=strength,
    )


def rule_venue_specialists(m: dict) -> Optional[PatternHit]:
    """Home team excellent à domicile + Away team piètre à l'extérieur → boost home.
    Miroir inversé sinon. Utilise home_wins/played et away_wins/played du classement."""
    hs = m.get("home_standing") or {}
    as_ = m.get("away_standing") or {}
    h_home_pg = _per_game(hs.get("home_wins"), (hs.get("home_wins", 0) + hs.get("home_draws", 0) + hs.get("home_losses", 0)))
    a_away_pg = _per_game(as_.get("away_wins"), (as_.get("away_wins", 0) + as_.get("away_draws", 0) + as_.get("away_losses", 0)))
    if h_home_pg is None or a_away_pg is None:
        return None
    # h_home_pg > 0.65 = très fort à dom (12+/20 wins), a_away_pg < 0.20 = piètre dehors
    home_advantage_case = h_home_pg >= 0.65 and a_away_pg <= 0.20
    away_advantage_case = h_home_pg <= 0.20 and a_away_pg >= 0.55
    if home_advantage_case:
        return PatternHit(
            name="venue_specialists_home",
            reason=f"Domicile ultra-dominant chez lui ({int(h_home_pg*100)}% V) vs adv. sans réussite dehors ({int(a_away_pg*100)}% V)",
            lam_home_mul=1.10, lam_away_mul=0.90, winner_bias=+0.30, confidence=0.85,
        )
    if away_advantage_case:
        return PatternHit(
            name="venue_specialists_away",
            reason=f"Extérieur redoutable en déplacement ({int(a_away_pg*100)}% V) vs adv. fragile chez lui ({int(h_home_pg*100)}% V)",
            lam_home_mul=0.90, lam_away_mul=1.10, winner_bias=-0.30, confidence=0.85,
        )
    return None


def rule_home_side_scoring(m: dict) -> Optional[PatternHit]:
    """Attaque du domicile à domicile vs défense de l'extérieur à l'extérieur — signal direct."""
    hs = m.get("home_standing") or {}
    as_ = m.get("away_standing") or {}
    h_home_games = (hs.get("home_wins", 0) or 0) + (hs.get("home_draws", 0) or 0) + (hs.get("home_losses", 0) or 0)
    a_away_games = (as_.get("away_wins", 0) or 0) + (as_.get("away_draws", 0) or 0) + (as_.get("away_losses", 0) or 0)
    if h_home_games < 5 or a_away_games < 5:
        return None
    h_attack_home = _safe(hs.get("home_goals_for"), float, 0) / h_home_games
    a_defense_away = _safe(as_.get("away_goals_against"), float, 0) / a_away_games
    # Very high scoring + very leaky defense → boost home lambda
    if h_attack_home >= 2.0 and a_defense_away >= 1.8:
        return PatternHit(
            name="home_attack_vs_leaky_away",
            reason=f"Dom. inscrit {h_attack_home:.1f}b/match chez lui vs ext. encaisse {a_defense_away:.1f}b/match dehors",
            lam_home_mul=1.12, lam_away_mul=1.00, winner_bias=+0.15, confidence=0.75,
        )
    # Very low scoring + rock-solid defense → damping home lambda + lean under
    if h_attack_home <= 1.0 and a_defense_away <= 0.8:
        return PatternHit(
            name="home_attack_stalled_vs_wall",
            reason=f"Dom. n'inscrit que {h_attack_home:.1f}b/match, ext. n'encaisse que {a_defense_away:.1f}b/match dehors",
            lam_home_mul=0.85, lam_away_mul=0.95, confidence=0.75,
        )
    return None


def rule_away_side_scoring(m: dict) -> Optional[PatternHit]:
    """Miroir de home_side_scoring pour l'attaque extérieur / défense domicile."""
    hs = m.get("home_standing") or {}
    as_ = m.get("away_standing") or {}
    a_away_games = (as_.get("away_wins", 0) or 0) + (as_.get("away_draws", 0) or 0) + (as_.get("away_losses", 0) or 0)
    h_home_games = (hs.get("home_wins", 0) or 0) + (hs.get("home_draws", 0) or 0) + (hs.get("home_losses", 0) or 0)
    if a_away_games < 5 or h_home_games < 5:
        return None
    a_attack_away = _safe(as_.get("away_goals_for"), float, 0) / a_away_games
    h_defense_home = _safe(hs.get("home_goals_against"), float, 0) / h_home_games
    if a_attack_away >= 1.7 and h_defense_home >= 1.7:
        return PatternHit(
            name="away_attack_vs_leaky_home",
            reason=f"Ext. inscrit {a_attack_away:.1f}b/match dehors vs dom. encaisse {h_defense_home:.1f}b/match chez lui",
            lam_home_mul=1.00, lam_away_mul=1.10, winner_bias=-0.12, confidence=0.70,
        )
    return None


def rule_form_asymmetry(m: dict) -> Optional[PatternHit]:
    """Standing.form: forme 5 derniers matchs — écart significatif de points."""
    hs = m.get("home_standing") or {}
    as_ = m.get("away_standing") or {}
    hf = _form_wdl(hs.get("form") or "")
    af = _form_wdl(as_.get("form") or "")
    if hf["n"] < 4 or af["n"] < 4:
        return None
    h_pts = hf["w"] * 3 + hf["d"]
    a_pts = af["w"] * 3 + af["d"]
    diff = h_pts - a_pts
    if abs(diff) < 6:  # need clear separation (>2 wins gap)
        return None
    strength = min(1.0, abs(diff) / 10.0)
    if diff > 0:
        return PatternHit(
            name="form_home_hot",
            reason=f"Forme récente : dom. {h_pts}pts vs ext. {a_pts}pts sur 5 matchs",
            lam_home_mul=1.0 + 0.04 * strength, lam_away_mul=1.0 - 0.03 * strength,
            winner_bias=+0.18 * strength, confidence=strength,
        )
    return PatternHit(
        name="form_away_hot",
        reason=f"Forme récente : ext. {a_pts}pts vs dom. {h_pts}pts sur 5 matchs",
        lam_home_mul=1.0 - 0.03 * strength, lam_away_mul=1.0 + 0.04 * strength,
        winner_bias=-0.18 * strength, confidence=strength,
    )


def rule_streak_reversal(m: dict) -> Optional[PatternHit]:
    """Streak V ou L de 4+ dans le form standing → régression à la moyenne (léger damping)."""
    hs = m.get("home_standing") or {}; as_ = m.get("away_standing") or {}
    hf = _form_wdl(hs.get("form") or ""); af = _form_wdl(as_.get("form") or "")
    reasons = []
    lam_h = lam_a = 1.0
    bias = 0.0
    if hf["streak_w"] >= 4:
        lam_h *= 0.96; bias -= 0.06
        reasons.append(f"dom. sur {hf['streak_w']}V — régression probable")
    if hf["streak_l"] >= 4:
        lam_h *= 1.03; bias += 0.04
        reasons.append(f"dom. sur {hf['streak_l']}D — rebond attendu")
    if af["streak_w"] >= 4:
        lam_a *= 0.96; bias += 0.06
        reasons.append(f"ext. sur {af['streak_w']}V — régression probable")
    if af["streak_l"] >= 4:
        lam_a *= 1.03; bias -= 0.04
        reasons.append(f"ext. sur {af['streak_l']}D — rebond attendu")
    if not reasons:
        return None
    return PatternHit(
        name="streak_reversal",
        reason=" ; ".join(reasons),
        lam_home_mul=lam_h, lam_away_mul=lam_a, winner_bias=bias, confidence=0.55,
    )


def rule_h2h_dominance(m: dict) -> Optional[PatternHit]:
    """Domination historique en H2H sur les 5+ derniers matchs."""
    h2h = m.get("h2h") or []
    if len(h2h) < 4:
        return None
    hid = (m.get("home_team") or {}).get("id")
    aid = (m.get("away_team") or {}).get("id")
    home_wins = away_wins = draws = 0
    for match in h2h[:6]:
        teams = match.get("teams") or {}
        winner_home = (teams.get("home") or {}).get("winner")
        winner_away = (teams.get("away") or {}).get("winner")
        g = match.get("goals") or {}
        hg, ag = g.get("home"), g.get("away")
        if hg is None or ag is None: continue
        # winner_home/away can be True/False/None ; use goals for safety
        if hg > ag: won_id = (teams.get("home") or {}).get("id")
        elif hg < ag: won_id = (teams.get("away") or {}).get("id")
        else: won_id = None
        if won_id == hid: home_wins += 1
        elif won_id == aid: away_wins += 1
        else: draws += 1
    total = home_wins + away_wins + draws
    if total < 4:
        return None
    if home_wins - away_wins >= 3:
        return PatternHit(
            name="h2h_home_dominates",
            reason=f"H2H : {home_wins}V dom. contre {away_wins} sur {total} confrontations",
            winner_bias=+0.20, lam_home_mul=1.04, confidence=0.65,
        )
    if away_wins - home_wins >= 3:
        return PatternHit(
            name="h2h_away_dominates",
            reason=f"H2H : {away_wins}V ext. contre {home_wins} sur {total} confrontations",
            winner_bias=-0.20, lam_away_mul=1.04, confidence=0.65,
        )
    return None


def rule_h2h_goals_pattern(m: dict) -> Optional[PatternHit]:
    """H2H tendance buts : matchs constamment prolifiques ou constamment fermés."""
    h2h = m.get("h2h") or []
    if len(h2h) < 4:
        return None
    totals = []
    for match in h2h[:6]:
        g = match.get("goals") or {}
        hg, ag = g.get("home"), g.get("away")
        if hg is None or ag is None: continue
        totals.append(hg + ag)
    if len(totals) < 4:
        return None
    avg = sum(totals) / len(totals)
    # >= 3.4 goals/match on H2H → boost O2.5 ; <= 1.8 → damp
    if avg >= 3.4:
        return PatternHit(
            name="h2h_high_scoring",
            reason=f"H2H {len(totals)} matchs : {avg:.1f} buts/match — historiquement prolifique",
            lam_home_mul=1.05, lam_away_mul=1.05, confidence=0.60,
        )
    if avg <= 1.8:
        return PatternHit(
            name="h2h_low_scoring",
            reason=f"H2H {len(totals)} matchs : {avg:.1f} buts/match — historiquement fermé",
            lam_home_mul=0.92, lam_away_mul=0.92, confidence=0.60,
        )
    return None


def rule_underdog_full_squad(m: dict) -> Optional[PatternHit]:
    """Le pattern-signature du user :
    outsider (cotes indiquent que c'est le dogue) mais banc plus complet → shift value bet."""
    market = m.get("odds") or {}
    ext = market.get("extended_markets") or {}
    def _imp(x):
        try: v = float(x)
        except (TypeError, ValueError): return None
        return 1.0 / v if v and v > 1 else None
    ph = _imp(ext.get("home")); pa = _imp(ext.get("away"))
    if ph is None or pa is None:
        return None
    fav = "home" if ph > pa else "away"
    fav_odds_prob = max(ph, pa)
    if fav_odds_prob < 0.55:
        return None  # not a clear favourite
    h_w = _safe(m.get("home_missing_weighted"), float, 0.0)
    a_w = _safe(m.get("away_missing_weighted"), float, 0.0)
    if h_w is None or a_w is None:
        return None
    # Favourite injured (>= 0.8) while underdog healthy (<= 0.2) → value on underdog
    if fav == "home" and h_w >= 0.8 and a_w <= 0.2:
        return PatternHit(
            name="underdog_full_squad_away",
            reason=f"Favori dom. touché ({h_w:.1f}) mais outsider ext. au complet ({a_w:.1f}) — value bet extérieur",
            winner_bias=-0.32, lam_home_mul=0.92, lam_away_mul=1.06, confidence=0.85,
        )
    if fav == "away" and a_w >= 0.8 and h_w <= 0.2:
        return PatternHit(
            name="underdog_full_squad_home",
            reason=f"Favori ext. touché ({a_w:.1f}) mais outsider dom. au complet ({h_w:.1f}) — value bet domicile",
            winner_bias=+0.32, lam_home_mul=1.06, lam_away_mul=0.92, confidence=0.85,
        )
    return None


def rule_fatigue_asymmetry(m: dict) -> Optional[PatternHit]:
    """Rest days : un côté <3j alors que l'autre >= 5j → shift vers le reposé."""
    hr = _safe(m.get("home_rest_days"), float)
    ar = _safe(m.get("away_rest_days"), float)
    if hr is None or ar is None:
        return None
    diff = hr - ar
    if abs(diff) < 3:
        return None
    strength = min(1.0, abs(diff) / 5.0)
    if diff > 0:  # home is more rested
        return PatternHit(
            name="rest_advantage_home",
            reason=f"Domicile {hr:.0f}j de repos contre {ar:.0f}j pour l'ext. — asymétrie de fraîcheur",
            lam_home_mul=1.0 + 0.05 * strength, lam_away_mul=1.0 - 0.05 * strength,
            winner_bias=+0.15 * strength, confidence=strength,
        )
    return PatternHit(
        name="rest_advantage_away",
        reason=f"Extérieur {ar:.0f}j de repos contre {hr:.0f}j pour dom. — asymétrie de fraîcheur",
        lam_home_mul=1.0 - 0.05 * strength, lam_away_mul=1.0 + 0.05 * strength,
        winner_bias=-0.15 * strength, confidence=strength,
    )


def rule_european_hangover(m: dict) -> Optional[PatternHit]:
    """Équipe qui a joué en coupe d'Europe récemment (euro_history récent proche de today)
    → fatigue rotation potentielle."""
    from datetime import datetime, timezone, timedelta
    def _recent_euro(team_hist):
        if not team_hist: return False
        try:
            most_recent = team_hist[0].get("fixture", {}).get("date", "")
            if not most_recent: return False
            dt = datetime.fromisoformat(most_recent.replace("Z", "+00:00"))
            return (datetime.now(timezone.utc) - dt) < timedelta(days=5)
        except Exception:
            return False
    h_euro = _recent_euro(m.get("home_euro_history"))
    a_euro = _recent_euro(m.get("away_euro_history"))
    if not (h_euro or a_euro):
        return None
    if h_euro and not a_euro:
        return PatternHit(
            name="euro_hangover_home",
            reason="Dom. a joué en coupe d'Europe récemment — rotation possible",
            lam_home_mul=0.94, winner_bias=-0.08, confidence=0.55,
        )
    if a_euro and not h_euro:
        return PatternHit(
            name="euro_hangover_away",
            reason="Ext. a joué en coupe d'Europe récemment — rotation possible",
            lam_away_mul=0.94, winner_bias=+0.08, confidence=0.55,
        )
    # both played → cancel out
    return None


def rule_formation_offensive(m: dict) -> Optional[PatternHit]:
    """Alignements offensifs (3-4-3, 4-2-4) → +buts. Alignements défensifs (5-4-1) → -buts."""
    hf = _safe(m.get("home_formation_score"), float, 1.0)
    af = _safe(m.get("away_formation_score"), float, 1.0)
    if hf is None or af is None:
        return None
    # formation_offensive_score is roughly 0.7 (défensif) → 1.3 (offensif)
    if hf >= 1.15 and af >= 1.15:
        return PatternHit(
            name="both_offensive_formations",
            reason=f"Les deux entraîneurs ont aligné des systèmes offensifs (scores {hf:.2f} / {af:.2f})",
            lam_home_mul=1.05, lam_away_mul=1.05, confidence=0.55,
        )
    if hf <= 0.85 and af <= 0.85:
        return PatternHit(
            name="both_defensive_formations",
            reason=f"Les deux entraîneurs ont aligné des systèmes défensifs (scores {hf:.2f} / {af:.2f})",
            lam_home_mul=0.92, lam_away_mul=0.92, confidence=0.55,
        )
    return None


def rule_cup_upset_tolerance(m: dict) -> Optional[PatternHit]:
    """Match de coupe européenne : réduit la conviction 1X2 (upsets plus fréquents)
    en tirant les proba vers le nul, sans changer les buts."""
    if not m.get("is_european_cup"):
        return None
    # Slight bias toward draw (bias 0 by construction) but reduces winner spread
    return PatternHit(
        name="cup_upset_tolerance",
        reason="Match de coupe européenne — davantage d'aléatoire dans le résultat, on tempère les convictions",
        lam_home_mul=0.98, lam_away_mul=0.98,  # marginally fewer goals in cup KO
        winner_bias=0.0, confidence=0.60,
    )


def rule_relegation_battle_urgency(m: dict) -> Optional[PatternHit]:
    """Les deux équipes sont en zone rouge → matchs très serrés, tendance nul ou 1-1."""
    hs = m.get("home_standing") or {}
    as_ = m.get("away_standing") or {}
    league_meta = m.get("league_meta") or {}
    total = _safe(league_meta.get("teams_count"), int)
    if not total:
        return None
    h_rank = _safe(hs.get("rank"), int)
    a_rank = _safe(as_.get("rank"), int)
    if h_rank is None or a_rank is None:
        return None
    # Both in bottom 25% of table
    if h_rank >= total * 0.75 and a_rank >= total * 0.75:
        return PatternHit(
            name="mutual_relegation_battle",
            reason=f"Duel du bas de tableau (#{h_rank} vs #{a_rank}) — match tendu, tendance nul",
            lam_home_mul=0.90, lam_away_mul=0.90, confidence=0.65,
        )
    return None


def rule_top_clash_defensive(m: dict) -> Optional[PatternHit]:
    """Deux top-5 → match plus fermé qu'attendu (les grosses ne se découvrent pas)."""
    hs = m.get("home_standing") or {}
    as_ = m.get("away_standing") or {}
    h_rank = _safe(hs.get("rank"), int)
    a_rank = _safe(as_.get("rank"), int)
    if h_rank is None or a_rank is None:
        return None
    if h_rank <= 5 and a_rank <= 5:
        return PatternHit(
            name="top_clash_defensive",
            reason=f"Top-5 vs top-5 (#{h_rank} vs #{a_rank}) — historiquement plus fermé",
            lam_home_mul=0.94, lam_away_mul=0.94, confidence=0.60,
        )
    return None


def rule_lopsided_form_trajectory(m: dict) -> Optional[PatternHit]:
    """Trajectoire opposée : dom. en LLWWW (rebond récent) vs ext. en WWWLL (chute récente)."""
    hs = m.get("home_standing") or {}
    as_ = m.get("away_standing") or {}
    h_form = (hs.get("form") or "").upper()
    a_form = (as_.get("form") or "").upper()
    if len(h_form) < 5 or len(a_form) < 5:
        return None
    def _trend(f):
        # >0 means improving, <0 means declining
        first_half = f[:2]
        last_half = f[-2:]
        pts_first = first_half.count("W") * 3 + first_half.count("D")
        pts_last = last_half.count("W") * 3 + last_half.count("D")
        return pts_last - pts_first
    h_trend = _trend(h_form)
    a_trend = _trend(a_form)
    # Look for strong opposite trends
    if h_trend >= 3 and a_trend <= -3:
        return PatternHit(
            name="trajectory_home_rising",
            reason=f"Trajectoires opposées : dom. en progression ({h_form}), ext. en déclin ({a_form})",
            lam_home_mul=1.05, lam_away_mul=0.95, winner_bias=+0.14, confidence=0.65,
        )
    if a_trend >= 3 and h_trend <= -3:
        return PatternHit(
            name="trajectory_away_rising",
            reason=f"Trajectoires opposées : ext. en progression ({a_form}), dom. en déclin ({h_form})",
            lam_home_mul=0.95, lam_away_mul=1.05, winner_bias=-0.14, confidence=0.65,
        )
    return None


def rule_referee_disciplinary(m: dict) -> Optional[PatternHit]:
    """Arbitre à cartons élevés + coupe européenne ou derby → matchs hachés → -buts."""
    ref = m.get("referee") or {}
    yellows = _safe(ref.get("avg_yellows_per_match"), float)
    if yellows is None or yellows < 5.5:
        return None
    is_cup = m.get("is_european_cup") or False
    if is_cup:
        return PatternHit(
            name="strict_ref_in_cup",
            reason=f"Arbitre sévère ({yellows:.1f} jaunes/match) en coupe — jeu plus haché",
            lam_home_mul=0.94, lam_away_mul=0.94, confidence=0.55,
        )
    return None


def rule_high_penalty_referee(m: dict) -> Optional[PatternHit]:
    """Arbitre siffle beaucoup de péno → +goals."""
    ref = m.get("referee") or {}
    pen = _safe(ref.get("penalty_rate"), float)
    if pen is None:
        return None
    if pen >= 0.5:  # 0.5+ pen/match tendency
        return PatternHit(
            name="high_penalty_referee",
            reason=f"Arbitre à haute fréquence de pénalty ({pen:.2f}/match) — +buts probable",
            lam_home_mul=1.04, lam_away_mul=1.04, confidence=0.55,
        )
    return None


def rule_h2h_btts_streak(m: dict) -> Optional[PatternHit]:
    """H2H : si 4 des 5 derniers matchs ont eu BTTS → boost BTTS."""
    h2h = m.get("h2h") or []
    if len(h2h) < 4:
        return None
    btts_count = 0
    total = 0
    for match in h2h[:5]:
        g = match.get("goals") or {}
        hg, ag = g.get("home"), g.get("away")
        if hg is None or ag is None: continue
        total += 1
        if hg >= 1 and ag >= 1: btts_count += 1
    if total < 4:
        return None
    if btts_count >= 4:
        return PatternHit(
            name="h2h_btts_streak",
            reason=f"BTTS dans {btts_count}/{total} derniers h2h — les deux équipes trouvent le but",
            lam_home_mul=1.05, lam_away_mul=1.05, confidence=0.60,
        )
    if btts_count <= 1 and total >= 4:
        return PatternHit(
            name="h2h_clean_sheet_pattern",
            reason=f"BTTS seulement {btts_count}/{total} derniers h2h — un cadenas fréquent",
            lam_home_mul=0.94, lam_away_mul=0.94, confidence=0.55,
        )
    return None


def rule_openness_double_leaky(m: dict) -> Optional[PatternHit]:
    """Les deux équipes prennent beaucoup de buts + les deux attaquent → match ouvert."""
    hs = m.get("home_standing") or {}
    as_ = m.get("away_standing") or {}
    def _all_side_avg(s):
        played = _safe(s.get("played"), int, 0) or 0
        if played < 5:
            return None, None
        gf = _safe(s.get("goals_for"), float, 0) / played
        ga = _safe(s.get("goals_against"), float, 0) / played
        return gf, ga
    h_gf, h_ga = _all_side_avg(hs)
    a_gf, a_ga = _all_side_avg(as_)
    if None in (h_gf, h_ga, a_gf, a_ga):
        return None
    # Both attack decent + both defense leaky → openness
    if h_gf >= 1.5 and a_gf >= 1.3 and h_ga >= 1.4 and a_ga >= 1.5:
        return PatternHit(
            name="double_openness",
            reason=f"Attaques prolifiques et défenses friables des deux côtés ({h_gf:.1f}⚽/{h_ga:.1f}∅ vs {a_gf:.1f}⚽/{a_ga:.1f}∅)",
            lam_home_mul=1.08, lam_away_mul=1.08, confidence=0.65,
        )
    # Both stingy attacks + both solid defenses → low-scoring
    if h_gf <= 1.0 and a_gf <= 1.0 and h_ga <= 1.0 and a_ga <= 1.0:
        return PatternHit(
            name="double_lockdown",
            reason=f"Attaques stériles et défenses de fer des deux côtés — match fermé",
            lam_home_mul=0.87, lam_away_mul=0.87, confidence=0.65,
        )
    return None


def rule_form_last3_asymmetry(m: dict) -> Optional[PatternHit]:
    """Un côté a fait le plein sur les 3 derniers (9pts) et l'autre 0-1pt : signal fort et récent."""
    hs = m.get("home_standing") or {}; as_ = m.get("away_standing") or {}
    hf = _form_wdl(hs.get("form") or ""); af = _form_wdl(as_.get("form") or "")
    if hf["n"] < 5 or af["n"] < 5:
        return None
    if hf["last3_pts"] >= 9 and af["last3_pts"] <= 1:
        return PatternHit(
            name="home_flying_visitors_crashing",
            reason=f"Dom. 9/9pts sur les 3 derniers, ext. {af['last3_pts']}pts — momentum très net",
            winner_bias=+0.22, lam_home_mul=1.05, lam_away_mul=0.96, confidence=0.75,
        )
    if af["last3_pts"] >= 9 and hf["last3_pts"] <= 1:
        return PatternHit(
            name="away_flying_home_crashing",
            reason=f"Ext. 9/9pts sur les 3 derniers, dom. {hf['last3_pts']}pts — momentum très net",
            winner_bias=-0.22, lam_home_mul=0.96, lam_away_mul=1.05, confidence=0.75,
        )
    return None


def rule_defensive_formation_vs_prolific(m: dict) -> Optional[PatternHit]:
    """Formation défensive contre une attaque très prolifique = risque contenu."""
    hs = m.get("home_standing") or {}; as_ = m.get("away_standing") or {}
    hf_score = _safe(m.get("home_formation_score"), float, 1.0)
    af_score = _safe(m.get("away_formation_score"), float, 1.0)
    played_h = _safe(hs.get("played"), int, 0) or 0
    played_a = _safe(as_.get("played"), int, 0) or 0
    if played_h < 5 or played_a < 5:
        return None
    h_gf = _safe(hs.get("goals_for"), float, 0) / played_h
    a_gf = _safe(as_.get("goals_for"), float, 0) / played_a
    # Away sets up defensive against home prolific attack → damp home
    if af_score <= 0.85 and h_gf >= 2.0:
        return PatternHit(
            name="away_bunker_vs_home_attack",
            reason=f"Ext. aligné en système défensif ({af_score:.2f}) face à une attaque dom. à {h_gf:.1f}b/match",
            lam_home_mul=0.92, confidence=0.55,
        )
    if hf_score <= 0.85 and a_gf >= 1.7:
        return PatternHit(
            name="home_bunker_vs_away_attack",
            reason=f"Dom. aligné en système défensif ({hf_score:.2f}) face à une attaque ext. à {a_gf:.1f}b/match",
            lam_away_mul=0.92, confidence=0.55,
        )
    return None


def rule_unbeaten_run(m: dict) -> Optional[PatternHit]:
    """Série d'invincibilité de 5 matchs (aucune L dans la forme récente)."""
    hs = m.get("home_standing") or {}; as_ = m.get("away_standing") or {}
    h_form = (hs.get("form") or "").upper()[-5:]
    a_form = (as_.get("form") or "").upper()[-5:]
    if len(h_form) < 5 or len(a_form) < 5:
        return None
    h_unbeaten = "L" not in h_form
    a_unbeaten = "L" not in a_form
    if h_unbeaten and not a_unbeaten:
        return PatternHit(
            name="home_unbeaten_run",
            reason=f"Dom. invaincu sur 5 matchs ({h_form}) — cadre solide",
            lam_home_mul=1.04, winner_bias=+0.10, confidence=0.60,
        )
    if a_unbeaten and not h_unbeaten:
        return PatternHit(
            name="away_unbeaten_run",
            reason=f"Ext. invaincu sur 5 matchs ({a_form}) — cadre solide",
            lam_away_mul=1.04, winner_bias=-0.10, confidence=0.60,
        )
    return None


def rule_league_meta_lopsided(m: dict) -> Optional[PatternHit]:
    """La ligue elle-même est très asymétrique (grand écart top/median) : le favori gagne plus souvent."""
    hs = m.get("home_standing") or {}; as_ = m.get("away_standing") or {}
    meta = m.get("league_meta") or {}
    max_pts = _safe(meta.get("max_points"), float)
    median_pts = _safe(meta.get("median_points"), float)
    if max_pts is None or median_pts is None or median_pts <= 0:
        return None
    lopsided = (max_pts - median_pts) / median_pts
    if lopsided < 0.8:  # league not that skewed
        return None
    h_pts = _safe(hs.get("points"), float, 0)
    a_pts = _safe(as_.get("points"), float, 0)
    if h_pts >= median_pts * 1.4 and a_pts <= median_pts * 0.6:
        return PatternHit(
            name="league_lopsided_home_top",
            reason=f"Ligue très hiérarchisée : dom. {int(h_pts)}pts (au-dessus) vs ext. {int(a_pts)}pts (fond)",
            lam_home_mul=1.05, lam_away_mul=0.95, winner_bias=+0.14, confidence=0.55,
        )
    if a_pts >= median_pts * 1.4 and h_pts <= median_pts * 0.6:
        return PatternHit(
            name="league_lopsided_away_top",
            reason=f"Ligue très hiérarchisée : ext. {int(a_pts)}pts (au-dessus) vs dom. {int(h_pts)}pts (fond)",
            lam_home_mul=0.95, lam_away_mul=1.05, winner_bias=-0.14, confidence=0.55,
        )
    return None


# ════════════════════════════════════════════════════════════
# Registry + apply
# ════════════════════════════════════════════════════════════
RULES: list[Callable[[dict], Optional[PatternHit]]] = [
    rule_squad_asymmetry,
    rule_standing_gap,
    rule_venue_specialists,
    rule_home_side_scoring,
    rule_away_side_scoring,
    rule_form_asymmetry,
    rule_streak_reversal,
    rule_h2h_dominance,
    rule_h2h_goals_pattern,
    rule_h2h_btts_streak,
    rule_underdog_full_squad,
    rule_fatigue_asymmetry,
    rule_european_hangover,
    rule_formation_offensive,
    rule_cup_upset_tolerance,
    rule_relegation_battle_urgency,
    rule_top_clash_defensive,
    rule_lopsided_form_trajectory,
    rule_referee_disciplinary,
    rule_high_penalty_referee,
    rule_openness_double_leaky,
    rule_form_last3_asymmetry,
    rule_defensive_formation_vs_prolific,
    rule_unbeaten_run,
    rule_league_meta_lopsided,
]

# Bounds on total adjustment applied
LAM_MUL_MIN, LAM_MUL_MAX = 0.72, 1.35
WINNER_BIAS_CLAMP = 0.45


def apply_patterns(match: dict, lam_home: float, lam_away: float,
                   probs_1x2: Optional[tuple[float, float, float]] = None) -> dict:
    """Run every rule, combine hits, return adjusted lambdas and 1X2 probs.

    probs_1x2 : (p_home, p_draw, p_away) — used to apply winner_bias.
                If None, only lambdas are returned and the caller re-derives 1X2 from lambdas.

    Returns dict with:
        lam_home, lam_away  : adjusted lambdas (bounded)
        probs_1x2           : shifted (p_home, p_draw, p_away) — or None if input was None
        hits                : list of {name, reason, confidence, adjustments}
        adjustments_summary : compact string of active patterns
    """
    hits_out = []
    total_lam_h = 1.0
    total_lam_a = 1.0
    total_bias = 0.0
    for rule in RULES:
        try:
            hit = rule(match)
        except Exception as e:
            log.warning("rule %s crashed: %s", rule.__name__, e)
            continue
        if hit is None:
            continue
        h = hit.scaled()
        total_lam_h *= h.lam_home_mul
        total_lam_a *= h.lam_away_mul
        total_bias += h.winner_bias
        hits_out.append({
            "name": h.name,
            "reason": h.reason,
            "lam_home_mul": round(h.lam_home_mul, 3),
            "lam_away_mul": round(h.lam_away_mul, 3),
            "winner_bias": round(h.winner_bias, 3),
            "confidence": round(h.confidence, 2),
        })

    # Bound overall multiplication
    total_lam_h = max(LAM_MUL_MIN, min(LAM_MUL_MAX, total_lam_h))
    total_lam_a = max(LAM_MUL_MIN, min(LAM_MUL_MAX, total_lam_a))
    total_bias = max(-WINNER_BIAS_CLAMP, min(WINNER_BIAS_CLAMP, total_bias))

    new_lam_h = lam_home * total_lam_h
    new_lam_a = lam_away * total_lam_a

    # Apply winner bias to 1X2 (log-odds nudge, then renormalize)
    new_probs = probs_1x2
    if probs_1x2 is not None and abs(total_bias) > 1e-4:
        ph, pd, pa = probs_1x2
        # bias > 0 favours home, < 0 favours away
        # Use additive log adjustment on H and A, keep draw invariant (renormalize)
        eps = 1e-6
        lh = math.log(max(eps, ph)) + total_bias
        la = math.log(max(eps, pa)) - total_bias
        ld = math.log(max(eps, pd))
        ph2 = math.exp(lh); pd2 = math.exp(ld); pa2 = math.exp(la)
        z = ph2 + pd2 + pa2
        new_probs = (ph2 / z, pd2 / z, pa2 / z)

    return {
        "lam_home": new_lam_h,
        "lam_away": new_lam_a,
        "probs_1x2": new_probs,
        "hits": hits_out,
        "total_lam_home_mul": round(total_lam_h, 3),
        "total_lam_away_mul": round(total_lam_a, 3),
        "total_winner_bias": round(total_bias, 3),
    }
