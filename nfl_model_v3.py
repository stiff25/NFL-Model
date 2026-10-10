from __future__ import annotations

"""NFL Model V3.4 — market-first Pick’em with timestamped pre-kickoff tracking.

Design goals
------------
1. Strict 2021-2025 walk-forward validation of the core model.
2. Live weekly engine uses the full enrichment layer when data is available.
2. Current-week live picks from a configurable odds feed.
3. QB availability, injury impact, travel/time zones, weather, and line movement.
4. Separate ML and ATS outputs.
5. No claim of 68% without a genuine forward, pre-deadline market comparison.

Live workflow
-------------
- Pull schedule + historical PBP from nflverse.
- Build pre-game team features.
- Pull current sportsbook markets from The Odds API (optional API key).
- Pull current venue weather from Open-Meteo.
- Apply manual/verified QB + injury overrides from a CSV.
- Fit on completed games only and score upcoming games.

Strict historical features use only results available before each matchup.
Calibration is fit from earlier out-of-fold scores. V3.3 keeps the market
favourite unless model override rules were tested on timestamped deadline odds
and passed a 2024-25 holdout gate, or experimental mode was explicitly enabled.
No real-world 68%+ Pick'em or 64%+ ATS rate is guaranteed.
The closing sportsbook spread is used for RETROSPECTIVE ATS grading only,
not as an input to the core classifiers or margin regression. Current odds
are fetched live, so pre-deadline prices are not reproducible unless archived.
QB/injury/weather context is kept out of the trained model until timestamped
historical pregame records exist (otherwise training sees missing data).
"""

import json
import math
import re
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================ CONFIG ====================================

HISTORY_START = 2019
BACKTEST_START = 2021
BACKTEST_END = 2025
TARGET_ACCURACY = 0.68
TARGET_ATS_ACCURACY = 0.64  # Future research target only; NOT proven

# Live pick gates — conservative by design.
MIN_ML_CONFIDENCE = 0.60
MIN_ML_EDGE = 0.035          # model probability - no-vig market probability
MIN_ATS_EDGE_POINTS = 1.50   # projected margin - market requirement
PICKEM_STRONG_CONFIDENCE = 0.72
PICKEM_QUALIFIED_CONFIDENCE = 0.60
STRONG_ATS_EDGE_POINTS = 2.50

# Live/current-season learning weights. The 2026 season is now included in
# model training, with its influence ramping up as the season progresses.
# Older seasons remain useful as the stabilizing baseline.
LIVE_SEASON_DECAY = 0.82
WEIGHT_CANDIDATES = (0.70, 0.82, 0.93, 1.0)
LIVE_CURRENT_SEASON_MIN_WEIGHT = 0.65
LIVE_CURRENT_SEASON_MAX_WEIGHT = 1.20
LIVE_CURRENT_SEASON_RAMP_WEEKS = 10

HALF_LIVES = (4, 8, 16)
ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 55.0
ELO_SEASON_REGRESSION = 0.75

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_REGIONS = os.getenv("ODDS_REGIONS", "au")
ODDS_MARKETS = os.getenv("ODDS_MARKETS", "h2h,spreads,totals")
BOOKMAKERS = os.getenv("BOOKMAKERS", "")
PREFERRED_BOOKMAKERS = os.getenv("PREFERRED_BOOKMAKERS", "")  # ordered bookmaker keys, optional
QB_INJURY_CSV = os.getenv("NFL_QB_INJURY_CSV", "qb_injury_overrides.csv")

# LIVE markets are genuine quotes at run time. A 75% market blend is an
# EXPERIMENTAL choice, not validated on historical deadline-matched odds.
# Set NFL_MARKET_WEIGHT=0 to disable blending without editing Python.
MARKET_WEIGHT = float(np.clip(float(os.getenv("NFL_MARKET_WEIGHT", "0.75")), 0, 1))
# An optional historical archive unlocks a fair deadline-matched backtest.
HISTORICAL_ODDS_CSV = os.getenv("NFL_HISTORICAL_ODDS_CSV", "data/historical_deadline_odds.csv")
CALIBRATION_MIN_GAMES = 240
CALIBRATION_MIN_VALIDATION_GAIN = 0.0005

# V3.3: without independently validated timestamped pre-kickoff quotes, choose
# the market favourite and FLAG possible model upset overrides for research.
# NFL_EXPERIMENTAL_UPSETS=1 enables explicitly UNVALIDATED overrides if desired;
# the default is market-favourite first. Never mislabel these as proven.
V33_EXPERIMENTAL_UPSETS = os.getenv("NFL_EXPERIMENTAL_UPSETS", "0").strip() == "1"
V33_RESEARCH_MODEL_CONFIDENCE = float(os.getenv("NFL_OVERRIDE_MIN_MODEL_PROB", "0.60"))
V33_RESEARCH_PROB_EDGE = float(os.getenv("NFL_OVERRIDE_MIN_EDGE", "0.10"))
V33_RESEARCH_MIN_VOTES = int(os.getenv("NFL_OVERRIDE_MIN_VOTES", "3"))
V33_POLICY_REPORT = Path(os.getenv("NFL_OVERRIDE_REPORT", "nfl_model_v3_backtest.json"))
V33_SNAPSHOT_PATH = Path(os.getenv("NFL_SNAPSHOT_PATH", "data/pick_snapshots.csv"))
V33_SAVE_SNAPSHOTS = os.getenv("NFL_SAVE_SNAPSHOTS", "0").strip() == "1"
# A real deadline archive for "just before each kickoff" must not use stale
# snapshots from days earlier. Limits are configurable for a specific contest.
V33_MAX_QUOTE_AGE_MIN = int(os.getenv("NFL_MAX_QUOTE_AGE_MIN", "180"))
V33_MAX_DEADLINE_LEAD_MIN = int(os.getenv("NFL_MAX_DEADLINE_LEAD_MIN", "180"))



# Current betting cutoff in UTC if you want reproducibility. Empty means now.
BET_CUTOFF_UTC = os.getenv("NFL_BET_CUTOFF_UTC", "")


# ======================== TEAM / VENUE DATA =============================

VENUES = {
    "ARI": (33.5276, -112.2626, "America/Phoenix", "retractable"),
    "ATL": (33.7554, -84.4009, "America/New_York", "dome"),
    "BAL": (39.2779, -76.6227, "America/New_York", "outdoor"),
    "BUF": (42.7738, -78.7870, "America/New_York", "outdoor"),
    "CAR": (35.2258, -80.8528, "America/New_York", "outdoor"),
    "CHI": (41.8623, -87.6167, "America/Chicago", "outdoor"),
    "CIN": (39.0954, -84.5160, "America/New_York", "outdoor"),
    "CLE": (41.5061, -81.6995, "America/New_York", "outdoor"),
    "DAL": (32.7473, -97.0945, "America/Chicago", "dome"),
    "DEN": (39.7439, -105.0201, "America/Denver", "outdoor"),
    "DET": (42.3400, -83.0456, "America/Detroit", "dome"),
    "GB":  (44.5013, -88.0622, "America/Chicago", "outdoor"),
    "HOU": (29.6847, -95.4107, "America/Chicago", "dome"),
    "IND": (39.7601, -86.1639, "America/Indiana/Indianapolis", "dome"),
    "JAX": (30.3239, -81.6373, "America/New_York", "outdoor"),
    "KC":  (39.0489, -94.4839, "America/Chicago", "outdoor"),
    "LV":  (36.0909, -115.1833, "America/Los_Angeles", "dome"),
    "LAC": (33.9535, -118.3390, "America/Los_Angeles", "outdoor"),
    "LAR": (33.9535, -118.3390, "America/Los_Angeles", "outdoor"),
    "MIA": (25.9580, -80.2389, "America/New_York", "retractable"),
    "MIN": (44.9738, -93.2575, "America/Chicago", "dome"),
    "NE":  (42.0909, -71.2643, "America/New_York", "outdoor"),
    "NO":  (29.9511, -90.0812, "America/Chicago", "dome"),
    "NYG": (40.8135, -74.0745, "America/New_York", "outdoor"),
    "NYJ": (40.8135, -74.0745, "America/New_York", "outdoor"),
    "PHI": (39.9008, -75.1675, "America/New_York", "outdoor"),
    "PIT": (40.4468, -80.0158, "America/New_York", "outdoor"),
    "SF":  (37.4030, -121.9694, "America/Los_Angeles", "outdoor"),
    "SEA": (47.5952, -122.3316, "America/Los_Angeles", "outdoor"),
    "TB":  (27.9759, -82.5033, "America/New_York", "outdoor"),
    "TEN": (36.1665, -86.7713, "America/Chicago", "outdoor"),
    "WAS": (38.9076, -76.8645, "America/New_York", "outdoor"),
}


def haversine(a_lat, a_lon, b_lat, b_lon):
    r = 6371.0088
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = math.radians(b_lat - a_lat)
    dl = math.radians(b_lon - a_lon)
    x = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*r*math.asin(math.sqrt(x))


# ============================ DATA ======================================

SCHEDULE_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"


def load_schedule(start=HISTORY_START, end=2026):
    d = pd.read_csv(SCHEDULE_URL)
    d = d[d["season"].between(start, end) & (d["game_type"] == "REG")].copy()
    d["gameday"] = pd.to_datetime(d["gameday"], errors="coerce")
    for c in ["home_score", "away_score", "spread_line", "total_line"]:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def load_pbp(seasons: Iterable[int]):
    frames = []
    cols = [
        "game_id", "season", "week", "season_type", "posteam", "defteam",
        "play_type", "epa", "success", "interception", "fumble_lost",
        "passer_player_id", "passer_player_name", "pass_attempt"
    ]
    for s in seasons:
        url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{s}.parquet"
        p = pd.read_parquet(url)
        keep = [c for c in cols if c in p.columns]
        p = p[keep]
        p = p[p["season_type"].eq("REG")].copy()
        frames.append(p)
    return pd.concat(frames, ignore_index=True)


def load_injuries_legacy(seasons: Iterable[int]):
    frames = []
    for s in seasons:
        url = f"https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{s}.csv"
        try:
            frames.append(pd.read_csv(url))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_depth_charts(seasons: Iterable[int]):
    frames = []
    for s in seasons:
        url = f"https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{s}.csv"
        try:
            d = pd.read_csv(url)
        except Exception:
            try:
                d = pd.read_parquet(url.replace(".csv", ".parquet"))
            except Exception:
                continue
        d["season"] = s
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_snap_counts(seasons: Iterable[int]):
    frames = []
    for s in seasons:
        url = f"https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{s}.parquet"
        try:
            frames.append(pd.read_parquet(url))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ======================= TEAM FORM + ELO ===============================


def build_team_history(pbp, schedule):
    p = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    p = p[p["posteam"].notna() & p["defteam"].notna()].copy()
    p["turnover"] = p.get("interception", 0).fillna(0) + p.get("fumble_lost", 0).fillna(0)

    off = (p.groupby(["season", "week", "game_id", "posteam"], as_index=False)
             .agg(off_epa=("epa", "mean"), off_success=("success", "mean"),
                  turnover=("turnover", "sum")))
    off = off.rename(columns={"posteam": "team"})

    pas = (p[p["play_type"].eq("pass")]
           .groupby(["season", "week", "game_id", "posteam"])["epa"].mean()
           .rename("pass_epa"))
    rush = (p[p["play_type"].eq("run")]
            .groupby(["season", "week", "game_id", "posteam"])["epa"].mean()
            .rename("rush_epa"))
    off = off.join(pas, on=["season", "week", "game_id", "team"])
    off = off.join(rush, on=["season", "week", "game_id", "team"])

    deff = (p.groupby(["season", "week", "game_id", "defteam"], as_index=False)
              .agg(def_epa_allowed=("epa", "mean"), def_success_allowed=("success", "mean"),
                   turnovers_forced=("turnover", "sum"))
              .rename(columns={"defteam": "team"}))

    g = schedule[["game_id", "season", "week", "gameday", "home_team", "away_team",
                  "home_score", "away_score"]].copy()
    h = g.assign(team=g.home_team, points_for=g.home_score, points_against=g.away_score, home_flag=1)
    a = g.assign(team=g.away_team, points_for=g.away_score, points_against=g.home_score, home_flag=0)
    long = pd.concat([h, a], ignore_index=True)
    long["point_diff"] = long.points_for - long.points_against
    long["win"] = (long.point_diff > 0).astype(float)

    out = long.merge(off, on=["season", "week", "game_id", "team"], how="left")
    out = out.merge(deff, on=["season", "week", "game_id", "team"], how="left")
    out["turnover_margin"] = out.turnovers_forced.fillna(0) - out.turnover.fillna(0)
    out = out.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)

    base = ["off_epa", "off_success", "pass_epa", "rush_epa", "def_epa_allowed",
            "def_success_allowed", "point_diff", "win", "turnover_margin"]
    for c in base:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        for hl in HALF_LIVES:
            alpha = 1 - math.exp(math.log(0.5) / hl)
            # Pregame history is shifted: strict walk-forward features never
            # contain the game being predicted. Postgame state is separate and
            # is used ONLY for live predictions after that game is completed.
            out[f"{c}_ewm{hl}"] = (
                out.groupby("team")[c]
                   .transform(lambda s: s.shift(1).ewm(alpha=alpha, adjust=False, min_periods=1).mean())
            )
            out[f"{c}_post_ewm{hl}"] = (
                out.groupby("team")[c]
                   .transform(lambda s: s.ewm(alpha=alpha, adjust=False, min_periods=1).mean())
            )

    # Sequential Elo
    ratings = {}
    prev_season = None
    elo_rows = []
    for _, r in schedule.sort_values(["gameday", "game_id"]).iterrows():
        s = int(r.season)
        if prev_season is not None and s != prev_season:
            ratings = {t: ELO_START + ELO_SEASON_REGRESSION*(v-ELO_START) for t,v in ratings.items()}
        hr = ratings.get(r.home_team, ELO_START)
        ar = ratings.get(r.away_team, ELO_START)
        elo_rows += [{"game_id": r.game_id, "team": r.home_team, "pre_elo": hr},
                     {"game_id": r.game_id, "team": r.away_team, "pre_elo": ar}]
        if pd.notna(r.home_score) and pd.notna(r.away_score):
            exp = 1/(1+10**(-((hr+ELO_HOME_ADV)-ar)/400))
            actual = 1.0 if r.home_score > r.away_score else 0.0 if r.home_score < r.away_score else 0.5
            mov = abs(float(r.home_score)-float(r.away_score))
            mov_mult = math.log(max(mov,1)+1)*(2.2/(0.001*abs(hr-ar)+2.2))
            chg = ELO_K*mov_mult*(actual-exp)
            ratings[r.home_team] = hr+chg
            ratings[r.away_team] = ar-chg
        # Capture the rating AFTER this completed game for the current live slate.
        elo_rows[-2]["post_elo"] = ratings.get(r.home_team, ELO_START)
        elo_rows[-1]["post_elo"] = ratings.get(r.away_team, ELO_START)
        prev_season = s
    elo = pd.DataFrame(elo_rows)
    return out.merge(elo, on=["game_id", "team"], how="left")


def to_games(team_hist, schedule):
    h = team_hist[team_hist.home_flag.eq(1)].copy().add_prefix("h_")
    a = team_hist[team_hist.home_flag.eq(0)].copy().add_prefix("a_")
    h["game_id"] = h.h_game_id
    a["game_id"] = a.a_game_id
    g = h.merge(a, on="game_id", how="inner")
    core = schedule[["game_id","season","week","gameday","home_team","away_team",
                     "home_score","away_score","spread_line","total_line","gametime"]].copy()
    g = g.merge(core, on="game_id", how="left")
    g["actual_margin"] = g.home_score - g.away_score
    g["home_win"] = (g.actual_margin > 0).astype(int)
    # nflverse convention: spread_line is POSITIVE when the home club is favoured.
    # Home ATS cover is actual margin > spread_line (NOT margin + spread_line).
    # An exactly equal margin is a PUSH and must not count as an ATS loss.
    cover_margin = g.actual_margin - g.spread_line
    g["home_cover"] = np.where(
        g.actual_margin.notna() & g.spread_line.notna() & cover_margin.ne(0),
        (cover_margin > 0).astype(float), np.nan
    )
    roots = ["off_epa","off_success","pass_epa","rush_epa","def_epa_allowed",
             "def_success_allowed","point_diff","win","turnover_margin"]
    for root in roots:
        for hl in HALF_LIVES:
            c = f"{root}_ewm{hl}"
            g[f"diff_{c}"] = g[f"h_{c}"].fillna(0)-g[f"a_{c}"].fillna(0)
    g["elo_diff"] = g.h_pre_elo.fillna(ELO_START)-g.a_pre_elo.fillna(ELO_START)
    return g.sort_values(["gameday","game_id"]).reset_index(drop=True)


# ====================== QB + INJURY LAYER ===============================


def build_qb_features(pbp, schedule, depth=None, injury=None, override=None):
    if pbp.empty:
        out = schedule[["game_id","home_team","away_team","season","week"]].copy()
        for s in ["home","away"]:
            for c in ["qb_form","qb_change","qb_injury","qb_uncertainty"]:
                out[f"{s}_{c}"] = 0.0
        return out

    q = pbp[pbp.get("passer_player_id", pd.Series(index=pbp.index)).notna()].copy()
    if q.empty:
        return build_qb_features(pd.DataFrame(), schedule)
    q["pass_attempt"] = pd.to_numeric(q.get("pass_attempt"), errors="coerce").fillna(1)
    q["qb_epa"] = pd.to_numeric(q.get("epa"), errors="coerce")
    game_qb = (q.groupby(["game_id","season","week","posteam","passer_player_id","passer_player_name"], as_index=False)
                 .agg(dropbacks=("pass_attempt","sum"), epa=("qb_epa","mean")))
    starters = (game_qb.sort_values(["game_id","posteam","dropbacks"], ascending=[True,True,False])
                .groupby("game_id", as_index=False).head(1))
    starters = starters.rename(columns={"posteam":"team","passer_player_id":"qb_id","passer_player_name":"qb_name"})

    q = game_qb.sort_values(["passer_player_id","season","week","game_id"])
    q["qb_form"] = q.groupby("passer_player_id")["epa"].transform(lambda s: s.shift(1).rolling(8, min_periods=1).mean())

    d = depth.copy() if depth is not None else pd.DataFrame()
    if not d.empty:
        d["position_norm"] = np.where(d.get("pos_abb", pd.Series(index=d.index)).notna(),
                                       d["pos_abb"].astype(str),
                                       d.get("position", d.get("depth_position", "")).astype(str))
        d = d[d.position_norm.str.upper().eq("QB")].copy()
        if "team" not in d.columns:
            d["team"] = d.get("depth_team", d.get("club_code", ""))
        if "full_name" in d.columns:
            d["qb_name"] = d.full_name
        elif "player_name" in d.columns:
            d["qb_name"] = d.player_name
        else:
            d["qb_name"] = ""
        d["rank"] = pd.to_numeric(d.get("depth_chart_position", d.get("pos_rank", 99)), errors="coerce").fillna(99)

    ov = override.copy() if override is not None else pd.DataFrame()
    if not ov.empty:
        ov["game_id"] = ov["game_id"].astype(str)

    rows=[]
    for _, game in schedule.iterrows():
        rec={"game_id":game.game_id}
        for side in ("home","away"):
            team=game[f"{side}_team"]
            prior=starters[(starters.team==team)&((starters.season<game.season)|((starters.season==game.season)&(starters.week<game.week)))]
            prev=prior.iloc[-1] if not prior.empty else None
            exp_id = prev.qb_id if prev is not None else None
            exp_name = prev.qb_name if prev is not None else None
            uncertain=1.0
            # Current/known depth chart QB, if present.
            if not d.empty:
                dd=d[d.team==team].copy()
                if "season" in dd.columns:
                    dd=dd[pd.to_numeric(dd.season,errors="coerce").fillna(0) <= int(game.season)]
                if "week" in dd.columns:
                    wk=pd.to_numeric(dd.week,errors="coerce")
                    dd=dd[(pd.to_numeric(dd.season,errors="coerce") < int(game.season)) | (wk <= int(game.week))]
                if not dd.empty:
                    dd=dd.sort_values("rank")
                    exp_name=dd.iloc[0].qb_name
                    uncertain=0.0
            # Override takes highest priority.
            if not ov.empty:
                x=ov[(ov.game_id==str(game.game_id))&(ov.team==team)]
                if not x.empty:
                    z=x.iloc[-1]
                    if pd.notna(z.get("expected_qb")) and str(z.expected_qb).strip(): exp_name=str(z.expected_qb)
                    uncertain=float(z.get("qb_uncertainty", uncertain))
                    rec[f"{side}_qb_injury"]=float(z.get("qb_injury_score",0.0))
                else:
                    rec[f"{side}_qb_injury"]=0.0
            else:
                rec[f"{side}_qb_injury"]=0.0
            # Form by expected name first, then prior id.
            if exp_name and q["passer_player_name"].notna().any():
                qq=q[(q.posteam==team)&(q.passer_player_name.astype(str).str.lower()==str(exp_name).lower())&
                     ((q.season<game.season)|((q.season==game.season)&(q.week<game.week)))]
            else: qq=pd.DataFrame()
            if qq.empty and exp_id is not None:
                qq=q[(q.passer_player_id.astype(str)==str(exp_id))&
                     ((q.season<game.season)|((q.season==game.season)&(q.week<game.week)))]
            rec[f"{side}_qb_form"]=float(qq.tail(8).qb_form.mean()) if not qq.empty else 0.0
            rec[f"{side}_qb_change"]=float(prev is not None and exp_name is not None and str(prev.qb_name).lower()!=str(exp_name).lower())
            rec[f"{side}_qb_uncertainty"]=uncertain
        rows.append(rec)
    return pd.DataFrame(rows)


POS_IMP={"QB":2.5,"T":1.5,"G":1.3,"C":1.3,"OL":1.5,"WR":1.0,"TE":0.9,"RB":0.8,
         "DL":1.0,"DE":1.0,"DT":1.0,"EDGE":1.0,"LB":0.9,"CB":0.9,"S":0.8,"K":0.25,"P":0.15}
STATUS_W={"OUT":1.0,"IR":1.0,"DOUBTFUL":0.8,"QUESTIONABLE":0.4,"SUSPENDED":1.0}


def injury_features(schedule, legacy_inj=None, snaps=None, override=None):
    rows=[]
    inj=legacy_inj.copy() if legacy_inj is not None else pd.DataFrame()
    sn=snaps.copy() if snaps is not None else pd.DataFrame()
    ov=override.copy() if override is not None else pd.DataFrame()
    if not inj.empty:
        inj["status_weight"]=inj.get("report_status","").fillna("").astype(str).str.upper().map(STATUS_W).fillna(0.0)
    for _,g in schedule.iterrows():
        r={"game_id":g.game_id}
        for side in ("home","away"):
            team=g[f"{side}_team"]
            a=b=count=0.0
            if not ov.empty:
                x=ov[(ov.game_id.astype(str)==str(g.game_id))&(ov.team==team)]
                if not x.empty:
                    z=x.iloc[-1]
                    a=float(z.get("offense_impact",0.0)); b=float(z.get("defense_impact",0.0)); count=float(z.get("injury_count",0.0))
            if (a+b)==0 and not inj.empty:
                ii=inj[(inj.season==g.season)&(inj.week==g.week)&(inj.team==team)].copy()
                if not ii.empty:
                    pos=ii.get("position","").astype(str).str.upper()
                    ii["imp"]=pos.map(POS_IMP).fillna(0.5)
                    # Use a simple role proxy if snap counts are present.
                    ii["usage"] = 1.0
                    if not sn.empty and "player" in sn.columns:
                        # Schema-dependent; use a conservative fallback if names do not line up.
                        pass
                    a=float((ii.status_weight*ii.imp*ii.usage).sum())
                    b=0.0
                    count=float((ii.status_weight>0).sum())
            r[f"{side}_injury_offense_impact"]=a
            r[f"{side}_injury_defense_impact"]=b
            r[f"{side}_injury_count"]=count
        rows.append(r)
    return pd.DataFrame(rows)


# ======================== TRAVEL + WEATHER ==============================


def travel_features(schedule):
    s=schedule.sort_values(["gameday","game_id"]).copy()
    events=[]
    for _,g in s.iterrows():
        events.append((g.game_id,g.gameday,g.home_team,g.home_team,1))
        events.append((g.game_id,g.gameday,g.away_team,g.home_team,0))
    e=pd.DataFrame(events,columns=["game_id","gameday","team","venue","home_flag"])
    rows=[]
    for _,g in s.iterrows():
        r={"game_id":g.game_id}
        for side in ("home","away"):
            team=g[f"{side}_team"]
            hist=e[(e.team==team)&((e.gameday<g.gameday)|((e.gameday==g.gameday)&(e.game_id<g.game_id)))]
            if hist.empty:
                r[f"{side}_travel_km"]=0.; r[f"{side}_road_games_14d"]=0.; r[f"{side}_days_rest_proxy"]=7.; continue
            prev=hist.iloc[-1]
            prevvenue=prev.team if prev.home_flag==1 else prev.venue
            curvenue=g.home_team
            if prevvenue in VENUES and curvenue in VENUES:
                r[f"{side}_travel_km"]=haversine(*VENUES[prevvenue][:2],*VENUES[curvenue][:2])
                r[f"{side}_tz_change"]=_tz_offset_delta(prevvenue,curvenue,g.gameday)
            else:
                r[f"{side}_travel_km"]=0.; r[f"{side}_tz_change"]=0.
            r[f"{side}_road_games_14d"]=float((hist[(hist.gameday>=g.gameday-pd.Timedelta(days=14))].home_flag==0).sum())
            r[f"{side}_days_rest_proxy"]=max((g.gameday-prev.gameday).days,0)
        r["away_travel_penalty"]=r.get("away_travel_km",0)/2000 + abs(r.get("away_tz_change",0))/3 + .10*r.get("away_road_games_14d",0) - .08*np.clip(r.get("away_days_rest_proxy",7)-7,-4,4)
        rows.append(r)
    return pd.DataFrame(rows)


def _tz_offset_delta(prev_team, cur_team, dt):
    try:
        from zoneinfo import ZoneInfo
        prevz=ZoneInfo(VENUES[prev_team][2]); curz=ZoneInfo(VENUES[cur_team][2])
        prev_off=dt.to_pydatetime().replace(tzinfo=prevz).utcoffset().total_seconds()/3600
        cur_off=dt.to_pydatetime().replace(tzinfo=curz).utcoffset().total_seconds()/3600
        return float(cur_off-prev_off)
    except Exception:
        return 0.0


def fetch_weather(schedule):
    import requests
    rows=[]
    for _,g in schedule.iterrows():
        meta=VENUES.get(g.home_team)
        if not meta or meta[3] in {"dome"}:
            rows.append({"game_id":g.game_id,"temp_f":65,"wind_mph":0,"precip_in":0,"snow_in":0,"weather_severity":0,"indoor_flag":1})
            continue
        url="https://api.open-meteo.com/v1/forecast"
        params={"latitude":meta[0],"longitude":meta[1],"hourly":"temperature_2m,precipitation,snowfall,wind_speed_10m,weather_code",
                "temperature_unit":"fahrenheit","wind_speed_unit":"mph","precipitation_unit":"inch","forecast_days":16,"timezone":meta[2]}
        try:
            p=requests.get(url,params=params,timeout=20).json()
            times=pd.to_datetime(p["hourly"]["time"])
            kickoff=pd.Timestamp(g.gameday)
            if pd.notna(g.get("gametime")):
                try:
                    hh,mm=str(g.gametime)[:5].split(":"); kickoff=kickoff.replace(hour=int(hh),minute=int(mm))
                except Exception: pass
            idx=int(np.argmin(np.abs(times-kickoff)))
            temp=float(p["hourly"]["temperature_2m"][idx]); wind=float(p["hourly"]["wind_speed_10m"][idx]); precip=float(p["hourly"]["precipitation"][idx]); snow=float(p["hourly"]["snowfall"][idx])
            sev=wind/20 + precip/.20 + snow/1.0
            rows.append({"game_id":g.game_id,"temp_f":temp,"wind_mph":wind,"precip_in":precip,"snow_in":snow,"weather_severity":sev,"indoor_flag":0})
        except Exception:
            rows.append({"game_id":g.game_id,"temp_f":65,"wind_mph":0,"precip_in":0,"snow_in":0,"weather_severity":0,"indoor_flag":0})
    return pd.DataFrame(rows)


# ============================= ODDS =====================================


def moneyline_implied(p):
    """American moneylines; zero, missing or invalid prices aren't tradable."""
    try:
        v = float(p)
    except (ValueError, TypeError):
        return np.nan
    if not np.isfinite(v) or v == 0 or abs(v) < 100:
        return np.nan
    return 100.0/(v+100.0) if v > 0 else -v/(-v+100.0)

def no_vig_probs(home_ml, away_ml):
    a=moneyline_implied(home_ml); b=moneyline_implied(away_ml)
    if not np.isfinite(a) or not np.isfinite(b): return (np.nan,np.nan)
    if a <= 0 or b <= 0: return (np.nan,np.nan)
    z=a+b
    if z <= 0 or not np.isfinite(z): return (np.nan,np.nan)
    return a/z,b/z


def _norm_team_name(name):
    """Normalize sportsbook team names so they can be matched to nflverse abbreviations."""
    if name is None:
        return ""
    s = str(name).strip().lower()
    aliases = {
        "arizona cardinals":"ari", "atlanta falcons":"atl",
        "baltimore ravens":"bal", "buffalo bills":"buf",
        "carolina panthers":"car", "chicago bears":"chi",
        "cincinnati bengals":"cin", "cleveland browns":"cle",
        "dallas cowboys":"dal", "denver broncos":"den",
        "detroit lions":"det", "green bay packers":"gb",
        "houston texans":"hou", "indianapolis colts":"ind",
        "jacksonville jaguars":"jax", "kansas city chiefs":"kc",
        "las vegas raiders":"lv", "oakland raiders":"lv",
        "los angeles chargers":"lac", "la chargers":"lac",
        "los angeles rams":"lar", "la rams":"lar",
        "miami dolphins":"mia", "minnesota vikings":"min",
        "new england patriots":"ne", "new orleans saints":"no",
        "new york giants":"nyg", "new york jets":"nyj",
        "philadelphia eagles":"phi", "pittsburgh steelers":"pit",
        "san francisco 49ers":"sf", "seattle seahawks":"sea",
        "tampa bay buccaneers":"tb", "tennessee titans":"ten",
        "washington commanders":"was",
    }
    if s in aliases:
        return aliases[s]
    # Handle abbreviations and common punctuation variants.
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    reverse = {v:k for k,v in aliases.items()}
    if s in reverse:
        return s
    return s

def _safe_market_number(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _real_spread(v):
    """Real half-point/whole-point bookmaker spread, never an averaged quarter point."""
    f = _safe_market_number(v)
    return np.isfinite(f) and abs(f*2 - round(f*2)) < 1e-7 and abs(f) <= 40


def select_bookmaker_market(offerings, preferred=()):
    """Select ONE sportsbook's actual spread & h2h prices (no invented medians).

    `offerings` is a list of individual bookmaker dictionaries for one game.
    If its preferred book has no spread, the next book with a genuine half-point
    line is selected. Moneylines are paired from the same bookmaker if present.
    """
    prefs = [str(x).strip().lower() for x in preferred if str(x).strip()]
    def order(x):
        key = str(x.get("bookmaker", "")).lower()
        return (prefs.index(key) if key in prefs else len(prefs), key)
    offers = sorted(offerings, key=order)
    valid_spreads = [x for x in offers if _real_spread(x.get("spread_home"))]
    chosen = valid_spreads[0] if valid_spreads else (offers[0] if offers else {})
    spread = _safe_market_number(chosen.get("spread_home")) if chosen and _real_spread(chosen.get("spread_home")) else np.nan
    # Try same bookmaker first, then fall back to another *complete* pair.
    valid_ml = lambda b: np.isfinite(moneyline_implied(b.get("home_ml"))) and np.isfinite(moneyline_implied(b.get("away_ml")))
    odds_pick = chosen if chosen and valid_ml(chosen) else next((b for b in offers if valid_ml(b)), {})
    return {
        "spread_home": spread,
        "spread_price_home": _safe_market_number(chosen.get("spread_price_home")) if np.isfinite(spread) else np.nan,
        "spread_bookmaker": str(chosen.get("bookmaker", "")) if np.isfinite(spread) else "",
        "home_ml": _safe_market_number(odds_pick.get("home_ml")),
        "away_ml": _safe_market_number(odds_pick.get("away_ml")),
        "ml_bookmaker": str(odds_pick.get("bookmaker", "")),
        "bookmakers_used": len(offers),
    }


def odds_api_current(schedule, api_key=ODDS_API_KEY):
    if not api_key:
        print("Odds API key not set; market fields will be unavailable.")
        return pd.DataFrame()
    import requests
    params={"apiKey":api_key,"regions":ODDS_REGIONS,"markets":ODDS_MARKETS,
            "oddsFormat":"american","dateFormat":"iso"}
    if BOOKMAKERS:
        params["bookmakers"] = BOOKMAKERS
    # Record when the quote was actually requested, not when the ML fit completed.
    quote_observed_utc=datetime.now(timezone.utc).isoformat(timespec="seconds")
    res=requests.get("https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds",
                     params=params,timeout=30)
    res.raise_for_status()
    events=res.json()
    by_teams={}
    for _,g in schedule.iterrows():
        key=(_norm_team_name(g.home_team), _norm_team_name(g.away_team))
        by_teams.setdefault(key, []).append(g)
    rows=[]; matched=0
    preferred=PREFERRED_BOOKMAKERS.split(",") if PREFERRED_BOOKMAKERS else []
    for event in events:
        ht=event.get("home_team"); at=event.get("away_team")
        matches=by_teams.get((_norm_team_name(ht),_norm_team_name(at)), [])
        if not matches:
            continue
        # If these opponents meet twice, don't silently assign odds to the wrong fixture.
        event_day=pd.to_datetime(event.get("commence_time"),utc=True,errors="coerce")
        if pd.notna(event_day):
            g=min(matches,key=lambda x: abs((pd.Timestamp(x.gameday)-event_day.tz_localize(None)).total_seconds()))
        elif len(matches)==1:
            g=matches[0]
        else:
            continue
        matched+=1
        offers=[]
        for book in event.get("bookmakers",[]):
            bm={"bookmaker":book.get("key", "")}
            for market in book.get("markets",[]):
                outcomes=market.get("outcomes",[])
                if market.get("key")=="spreads":
                    for x in outcomes:
                        if _norm_team_name(x.get("name"))==_norm_team_name(ht):
                            bm["spread_home"]=x.get("point")
                            bm["spread_price_home"]=x.get("price")
                            break
                if market.get("key")=="h2h":
                    o={_norm_team_name(x.get("name")):x.get("price") for x in outcomes}
                    bm["home_ml"]=o.get(_norm_team_name(ht))
                    bm["away_ml"]=o.get(_norm_team_name(at))
            offers.append(bm)
        selected=select_bookmaker_market(offers,preferred)
        rows.append({"game_id":g.game_id, "kickoff_utc":event.get("commence_time"),
                     "odds_observed_utc":quote_observed_utc, **selected})
    print(f"Odds API events received: {len(events)}; matched to schedule: {matched}")
    if not rows: return pd.DataFrame()
    result=pd.DataFrame(rows)
    return result.drop_duplicates("game_id",keep="last")

def attach_market(g, odds):
    out=g.copy()
    numeric=["spread_home","spread_price_home","total","home_ml","away_ml"]
    textual=["spread_bookmaker","ml_bookmaker","kickoff_utc","odds_observed_utc"]
    if odds.empty:
        for c in numeric: out[c]=np.nan
        for c in textual: out[c]=""
        out["bookmakers_used"]=0
        return out
    out=out.merge(odds,on="game_id",how="left",validate="one_to_one")
    for c in numeric:
        if c not in out: out[c]=np.nan
    for c in textual:
        if c not in out: out[c]=""
    if "bookmakers_used" not in out: out["bookmakers_used"]=0
    return out

# =========================== MODEL ======================================


def feature_sets():
    """Only pregame signals that exist consistently in the historical data.

    Never train on nflverse historical *closing* spread_line or on API data that
    did not exist at the pick deadline. QB / injuries / weather remain displayed
    as context; they're not silently treated as trained features with zero
    historical coverage. Add them back only with archived, timestamped reports.
    """
    result={}
    for label, hl in zip(("FAST","MEDIUM","SLOW"), HALF_LIVES):
        base=[f"diff_{x}_ewm{hl}" for x in (
            "off_epa","off_success","pass_epa","rush_epa","def_epa_allowed",
            "def_success_allowed","point_diff","win","turnover_margin")]
        result[label] = base + ["elo_diff"]
    return result

def prep(df, cols):
    x=df.copy()
    for c in cols:
        if c not in x: x[c]=0.0
    return x[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0)


def classifier():
    return Pipeline([("scale",StandardScaler()),("logit",LogisticRegression(C=.35,max_iter=2000))])


def margin_model():
    return Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=8.0))])


def live_training_weights(train, asof_season=2026, asof_week=None, season_decay=None):
    """Return recency/season weights for the live model training set.

    Current-season games are included and gradually gain influence as the
    season progresses. Previous seasons decay by season so recent historical
    performance remains more relevant than older history.
    """
    t=train.copy()
    season_num=pd.to_numeric(t.get("season"),errors="coerce").fillna(asof_season).astype(int)
    decay=LIVE_SEASON_DECAY if season_decay is None else float(season_decay)
    base=np.power(decay, np.maximum(asof_season-season_num, 0))

    if asof_week is None or pd.isna(asof_week):
        progress=1.0
    else:
        progress=float(np.clip((float(asof_week)-1.0)/max(LIVE_CURRENT_SEASON_RAMP_WEEKS-1,1),0.0,1.0))
    current_weight=LIVE_CURRENT_SEASON_MIN_WEIGHT + (LIVE_CURRENT_SEASON_MAX_WEIGHT-LIVE_CURRENT_SEASON_MIN_WEIGHT)*progress
    weights=np.array(base,dtype=float,copy=True)
    weights[season_num.to_numpy()==asof_season]=current_weight
    return np.maximum(weights,0.05)


def add_model_diffs(df):
    d=df.copy()
    d["qb_form_diff"]=d.get("home_qb_form",0)-d.get("away_qb_form",0)
    d["qb_change_diff"]=d.get("away_qb_change",0)-d.get("home_qb_change",0)
    d["qb_injury_diff"]=d.get("away_qb_injury",0)-d.get("home_qb_injury",0)
    d["qb_uncertainty_diff"]=d.get("away_qb_uncertainty",0)-d.get("home_qb_uncertainty",0)
    d["injury_offense_diff"]=d.get("away_injury_offense_impact",0)-d.get("home_injury_offense_impact",0)
    d["injury_defense_diff"]=d.get("away_injury_defense_impact",0)-d.get("home_injury_defense_impact",0)
    d["weather_temp_extreme"]=np.maximum(0,45-d.get("temp_f",65))/20 + np.maximum(0,d.get("temp_f",65)-85)/20
    if "spread_move_open_to_cutoff" not in d: d["spread_move_open_to_cutoff"]=0.0
    if "opening_spread" not in d: d["opening_spread"]=d.get("spread_line",0)
    if "cutoff_spread" not in d: d["cutoff_spread"]=d.get("spread_line",0)
    return d


def tune_pre_holdout_decay(games):
    """Tune on 2022-23 annual held-out predictions ONLY; 2024-25 untouched.

    Annual holdouts are an efficient independent checkpoint, not a claim that
    the eventual selected parameter beats the default in forward evaluation.
    """
    fs=feature_sets()
    scores=[]
    for decay in WEIGHT_CANDIDATES:
        preds=[]; labels=[]
        for season in (2022,2023):
            tr=games[(games.season<season)&games.home_score.notna()].copy()
            te=games[(games.season==season)&games.home_score.notna()].copy()
            if len(tr)<250 or te.empty:continue
            w=live_training_weights(tr,asof_season=season,asof_week=1,season_decay=decay)
            this=[]
            for cols in fs.values():
                fitted=classifier().fit(prep(tr,cols),tr.home_win,logit__sample_weight=w)
                this.append(fitted.predict_proba(prep(te,cols))[:,1])
            preds.extend(np.vstack(this).mean(axis=0).tolist())
            labels.extend(te.home_win.to_numpy(dtype=int).tolist())
        if preds:
            p=np.asarray(preds,dtype=float);y=np.asarray(labels,dtype=float)
            scores.append({"season_decay":float(decay),"validation_games":len(p),
                           "validation_brier":float(np.mean((p-y)**2)),
                           "validation_accuracy":float(np.mean((p>=.5)==y))})
    selected=min(scores,key=lambda x:(x["validation_brier"],-x["validation_accuracy"])) if scores else None
    return float(selected["season_decay"]) if selected else LIVE_SEASON_DECAY, {
        "selection_years":[2022,2023],"untouched_holdout_years":[2024,2025],
        "metric":"Brier score (primary), accuracy tie-break",
        "selected_decay":float(selected["season_decay"]) if selected else LIVE_SEASON_DECAY,
        "tuning_available":bool(scores),"candidate_scores":scores}


def walk_forward(games, start=BACKTEST_START, end=BACKTEST_END, holdout_decay=None):
    """Week-by-week walk forward: fit models AND calibrator using only prior games.

    Evaluated probability and winner are calibrated if and only if an earlier
    validation slice proved calibration beneficial on Brier score.
    """
    gs=games.copy(); fs=feature_sets(); rows=[]
    for season in range(start,end+1):
        for week in sorted(gs.loc[gs.season.eq(season),"week"].dropna().unique()):
            tr=gs[((gs.season<season)|((gs.season==season)&(gs.week<week))) & gs.home_score.notna()].copy()
            te=gs[(gs.season==season)&(gs.week==week)&gs.home_score.notna()].copy()
            if len(tr)<250 or te.empty: continue
            weights=live_training_weights(tr,asof_season=int(season),asof_week=float(week),
                season_decay=holdout_decay if season>=2024 else LIVE_SEASON_DECAY)
            scores=[]; votes=[]
            for cols in fs.values():
                fitted=classifier().fit(prep(tr,cols),tr.home_win,logit__sample_weight=weights)
                pp=fitted.predict_proba(prep(te,cols))[:,1]
                scores.append(pp);votes.append((pp>=.5).astype(int))
            raw=np.vstack(scores).mean(axis=0)
            prior=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
            cal,meta=fit_past_oof_calibrator(prior)
            p=calibrated_probs(raw,cal)
            n_votes=np.vstack(votes).sum(axis=0)
            cols=list(dict.fromkeys(fs["MEDIUM"]+fs["SLOW"]))
            reg=margin_model().fit(prep(tr,cols),tr.actual_margin,ridge__sample_weight=weights)
            margins=reg.predict(prep(te,cols))
            o=te[["game_id","season","week","gameday","home_team","away_team","spread_line","home_win","home_cover","actual_margin"]].copy()
            o["raw_home_prob"]=raw
            o["home_win_prob"]=p
            o["calibrator_enabled"]=bool(cal is not None)
            o["pick_home"]=(p>=.5).astype(int)
            o["model_votes_home"]=n_votes
            o["confidence"]=np.maximum(p,1-p)
            o["projected_margin"]=margins
            # nflverse CLOSE spread positive if HOME favoured; only used to grade.
            o["ats_edge_points_home"]=margins-o.spread_line
            o["ats_pick_home"]=np.where(o.spread_line.notna(),(o.ats_edge_points_home>0).astype(int),np.nan)
            o["ats_signal"]=np.select([
                o.spread_line.notna() & (o.ats_edge_points_home.abs()>=STRONG_ATS_EDGE_POINTS),
                o.spread_line.notna() & (o.ats_edge_points_home.abs()>=MIN_ATS_EDGE_POINTS),
            ], ["STRONG","QUALIFIED"], default="PASS")
            rows.append(o)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()

def _safe_logit(p):
    p=np.clip(np.asarray(p,dtype=float),1e-5,1-1e-5)
    return np.log(p/(1-p)).reshape(-1,1)


def fit_past_oof_calibrator(history, min_games=CALIBRATION_MIN_GAMES):
    """Only chronological, previously-scored OUT-OF-FOLD games may train this.

    It is enabled only when it improves Brier score on a later, untouched
    validation slice of this *past* history, then refit on all past OOF rows.
    Avoids using the outcome of the game being scored.
    """
    if history is None or len(history)<min_games:
        return None, {"reason":"insufficient_past_oof_games","n":0 if history is None else len(history)}
    h=history.sort_values(["gameday","game_id"]).copy()
    p=pd.to_numeric(h["raw_home_prob"],errors="coerce").to_numpy(dtype=float)
    y=pd.to_numeric(h["home_win"],errors="coerce").to_numpy(dtype=float)
    ok=np.isfinite(p)&np.isfinite(y)
    p=p[ok]; y=y[ok]
    if len(p)<min_games or len(np.unique(y))<2:
        return None, {"reason":"not_enough_valid_class_data","n":len(p)}
    cut=int(len(p)*0.7)
    if cut<150 or len(p)-cut<60 or len(np.unique(y[:cut]))<2:
        return None, {"reason":"too_few_valid_validation_games","n":len(p)}
    check=LogisticRegression(C=2.0,max_iter=300).fit(_safe_logit(p[:cut]), y[:cut])
    validated=check.predict_proba(_safe_logit(p[cut:]))[:,1]
    raw_brier=float(np.mean((p[cut:]-y[cut:])**2))
    calibrated_brier=float(np.mean((validated-y[cut:])**2))
    gain=raw_brier-calibrated_brier
    if gain < CALIBRATION_MIN_VALIDATION_GAIN:
        return None, {"reason":"did_not_improve_past_validation_brier","n":len(p),
                      "validation_brier_gain":gain}
    model=LogisticRegression(C=2.0,max_iter=300).fit(_safe_logit(p),y)
    return model, {"reason":"enabled_from_past_oof_only","n":len(p),
                   "validation_brier_gain":gain}


def calibrated_probs(raw, calibrator):
    raw=np.asarray(raw,dtype=float)
    return calibrator.predict_proba(_safe_logit(raw))[:,1] if calibrator is not None else raw.copy()


def load_verified_deadline_odds(path=HISTORICAL_ODDS_CSV):
    """Optional archive with verified UTC snapshot/deadline/kickoff timestamps.

    Mandatory CSV headers: game_id,snapshot_utc,deadline_utc,kickoff_utc,
    home_ml,away_ml. All timestamps must be ISO8601 UTC with offset/Z.
    The LATEST valid market quote no later than the deadline is chosen.
    Missing archives are NOT inferred from historical closing spreads.
    """
    file=Path(path)
    if not file.exists():
        return pd.DataFrame(columns=["game_id","deadline_market_home_prob","deadline_snapshot_utc"])
    d=pd.read_csv(file)
    required=["game_id","snapshot_utc","deadline_utc","kickoff_utc","home_ml","away_ml"]
    missing=[c for c in required if c not in d]
    if missing: raise ValueError(f"Historical odds archive missing headers: {missing}")
    for c in ("snapshot_utc","deadline_utc","kickoff_utc"):
        # Reject timestamps lacking an explicit timezone, even if pandas might
        # otherwise assume UTC silently.
        tzexplicit=d[c].astype(str).str.contains(r'(?:Z|[+-]\d{2}:?\d{2})$',regex=True,case=False)
        d.loc[~tzexplicit,c]=pd.NA
        d[c]=pd.to_datetime(d[c],errors="coerce",utc=True)
    d=d[d.snapshot_utc.notna() & d.deadline_utc.notna() & d.kickoff_utc.notna() &
        (d.snapshot_utc<=d.deadline_utc) & (d.deadline_utc<d.kickoff_utc)].copy()
    # Do not count week-old lines as "just before kickoff" evidence.
    mins_to_kickoff=(d.kickoff_utc-d.deadline_utc).dt.total_seconds()/60
    quote_age=(d.deadline_utc-d.snapshot_utc).dt.total_seconds()/60
    d=d[mins_to_kickoff.between(0,V33_MAX_DEADLINE_LEAD_MIN) &
        quote_age.between(0,V33_MAX_QUOTE_AGE_MIN)].copy()
    if d.empty:
        return pd.DataFrame(columns=["game_id","deadline_market_home_prob","deadline_snapshot_utc"])
    d["deadline_market_home_prob"]=[no_vig_probs(h,a)[0] for h,a in zip(d.home_ml,d.away_ml)]
    d=d[np.isfinite(d.deadline_market_home_prob)].sort_values(["game_id","snapshot_utc"])
    d=d.drop_duplicates("game_id",keep="last")
    result=d[["game_id","deadline_market_home_prob","snapshot_utc"]].copy()
    return result.rename(columns={"snapshot_utc":"deadline_snapshot_utc"})


# =================== V3.3 MARKET-FIRST PICK'EM ==========================

def v33_candidate_override(model_home_prob, market_home_prob, votes_home,
                           min_confidence=V33_RESEARCH_MODEL_CONFIDENCE,
                           min_edge=V33_RESEARCH_PROB_EDGE,
                           required_votes=V33_RESEARCH_MIN_VOTES):
    """A *candidate* underdog selection, not a verified profitable override.

    Works on scalars or arrays. The market must have a non-tied, valid quote.
    Candidates disagree with the market, meet a model probability threshold,
    exceed the market probability for that underdog, and have model agreement.
    """
    m=np.asarray(model_home_prob,dtype=float)
    q=np.asarray(market_home_prob,dtype=float)
    v=np.asarray(votes_home,dtype=float)
    market_valid=np.isfinite(q)&(q>0)&(q<1)&(np.abs(q-.5)>1e-9)
    model_valid=np.isfinite(m)&(m>=0)&(m<=1)&np.isfinite(v)
    fav=q>.5; modeled=m>.5
    underdog_model_prob=np.where(modeled,m,1-m)
    underdog_market_prob=np.where(modeled,q,1-q)
    votes_for=np.where(modeled,v,3-v)
    return (market_valid & model_valid & (fav!=modeled) &
            (underdog_model_prob+1e-10>=min_confidence) &
            ((underdog_model_prob-underdog_market_prob)+1e-10>=min_edge) &
            (votes_for>=required_votes))


def v33_policy_metrics(pred, quotes, params):
    """Exact-fixture comparison on only verified, pre-deadline odds games."""
    if pred is None or pred.empty or quotes is None or quotes.empty:
        return {"games":0,"market_correct":0,"strategy_correct":0,"flips":0,"net_correct_over_market":0}
    x=pred.merge(quotes[["game_id","deadline_market_home_prob"]],on="game_id",how="inner")
    x=x[x.home_win.notna() & x.deadline_market_home_prob.notna() & x.home_win_prob.notna()].copy()
    if x.empty:
        return {"games":0,"market_correct":0,"strategy_correct":0,"flips":0,"net_correct_over_market":0}
    market=x.deadline_market_home_prob.to_numpy(dtype=float)>.5
    model=x.home_win_prob.to_numpy(dtype=float)>.5
    eligible=np.isfinite(x.deadline_market_home_prob.to_numpy(dtype=float)) & (x.deadline_market_home_prob.to_numpy(dtype=float)!=.5)
    override=v33_candidate_override(x.home_win_prob.to_numpy(dtype=float),
             x.deadline_market_home_prob.to_numpy(dtype=float),
             x.model_votes_home.to_numpy(dtype=float),
             min_confidence=params["min_confidence"],min_edge=params["min_edge"],
             required_votes=params["min_votes"])
    pick=np.where(override,model,market)
    # With a true no-vig 50/50 market there is no favourite; choose the model.
    pick=np.where(eligible,pick,model)
    baseline=np.where(eligible,market,model)
    actual=x.home_win.to_numpy(dtype=int).astype(bool)
    return {"games":int(len(x)),"market_correct":int(np.sum(baseline==actual)),
            "strategy_correct":int(np.sum(pick==actual)),"flips":int(np.sum(override)),
            "net_correct_over_market":int(np.sum(pick==actual)-np.sum(baseline==actual)),
            "market_accuracy":float(np.mean(baseline==actual)),
            "strategy_accuracy":float(np.mean(pick==actual)),
            "model_accuracy":float(np.mean(model==actual))}


def v33_select_override_policy(pred, quotes):
    """Tune ONLY on 2021-23, then check the *one* winner on 2024-25.

    Do not create a fictional market backtest using the closing lines.
    A holdout validation gate is not a future-proof guarantee and is clearly
    labelled as such. A policy needs a real archive with sufficient coverage.
    """
    unavailable={"method":"MARKET_FIRST_V33","validated":False,"policy":None,
      "reason":"No adequate timestamped historical pre-deadline moneyline quotes",
      "selection_years":[2021,2022,2023],"validation_years":[2024,2025],
      "note":"Closing lines are never substituted for pre-deadline prices"}
    if pred is None or pred.empty or quotes is None or quotes.empty:
        return unavailable
    train=pred[pred.season.isin([2021,2022,2023])]
    test=pred[pred.season.isin([2024,2025])]
    options=[]
    for conf in (0.54,0.57,0.60,0.63):
        for edge in (0.05,0.08,0.11,0.14):
            for votes in (2,3):
                settings={"min_confidence":conf,"min_edge":edge,"min_votes":votes}
                result=v33_policy_metrics(train,quotes,settings)
                # minimum candidate count; otherwise an apparent advantage is noise
                if result["games"]>=150 and result["flips"]>=8:
                    options.append((result["net_correct_over_market"],-result["flips"],settings,result))
    if not options:
        unavailable["reason"]="Insufficient verified training coverage or upset candidates (minimum 150 games and 8 candidates)"
        return unavailable
    # The held-out years do not participate in parameter selection.
    best=max(options,key=lambda t:(t[0],t[1]))
    policy=best[2]
    train_metrics=best[3]
    held=v33_policy_metrics(test,quotes,policy)
    # Conservative admission gate — not a claim of statistical proof.
    pass_gate=(train_metrics["net_correct_over_market"]>=3 and
               held["games"]>=100 and held["flips"]>=8 and
               held["net_correct_over_market"]>=2)
    return {"method":"MARKET_FIRST_V33","validated":bool(pass_gate),
            "policy":policy,"selection_years":[2021,2022,2023],
            "validation_years":[2024,2025],"training":train_metrics,
            "heldout":held,"reason":("Met conservative holdout admission gate" if pass_gate else
                    "No demonstrated, adequately sampled holdout improvement; stay with favourites"),
            "note":"Holdout gates are checks, not proof of future accuracy; prices must be truly pre-deadline"}


def v33_load_deployment_policy(path=V33_POLICY_REPORT):
    """Live reads the V3.3 backtest report; never learns an override from closing lines."""
    try:
        obj=json.loads(Path(path).read_text())
        pol=obj.get("market_first_override_policy",{})
        if pol.get("method")!="MARKET_FIRST_V33" or pol.get("validated") is not True:
            return None
        config=pol.get("policy")
        if not isinstance(config,dict) or not all(k in config for k in ("min_confidence","min_edge","min_votes")):
            return None
        if not (0.5<=float(config["min_confidence"])<=0.9 and 0<=float(config["min_edge"])<=0.5 and int(config["min_votes"]) in (2,3)):
            return None
        return {"min_confidence":float(config["min_confidence"]),
                "min_edge":float(config["min_edge"]),"min_votes":int(config["min_votes"])}
    except (OSError,ValueError,TypeError,AttributeError):
        return None


def v33_log_live_snapshots(picks, path=V33_SNAPSHOT_PATH):
    """Append transparent as-run snapshots; not a guarantee of a pre-kickoff lock.

    The GitHub workflow must stage this data file for it to persist across runs.
    """
    if picks is None or picks.empty:
        return
    keep=["game_id","season","week","gameday","kickoff_utc","snapshot_utc",
          "home_team","away_team","home_ml","away_ml","market_home_prob",
          "calibrated_model_home_prob","market_favorite_pick_home","candidate_upset",
          "validated_override","pick_home","pickem_pick","pickem_method",
          "confidence","pickem_signal"]
    new=picks[[c for c in keep if c in picks.columns]].copy()
    new["game_id"]=new["game_id"].astype(str)
    kickoff=pd.to_datetime(new.get("kickoff_utc"),errors="coerce",utc=True)
    observed=pd.to_datetime(new.get("snapshot_utc"),errors="coerce",utc=True)
    new["verified_pre_kickoff_snapshot"]=kickoff.notna() & observed.notna() & (observed<kickoff)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        try:
            old=pd.read_csv(path,dtype={"game_id":str})
            old["game_id"]=old["game_id"].astype(str)
            new=pd.concat([old,new],ignore_index=True)
            if "snapshot_utc" in new.columns:
                new=new.drop_duplicates(subset=["game_id","snapshot_utc"],keep="last")
        except (OSError,ValueError,pd.errors.ParserError) as e:
            print(f"Skipped snapshot history overwrite because existing CSV could not be read: {e}")
            return
    new.to_csv(path,index=False)
    print(f"Saved {len(picks)} pick snapshots to {path}. Ensure workflow commits this file.")


def fair_historical_market_diagnostics(pred, deadline_quotes, market_weight=MARKET_WEIGHT):
    """Score all methods on exactly the same games with real deadline odds."""
    if deadline_quotes is None or deadline_quotes.empty or pred.empty:
        return {"deadline_market_games":0,"deadline_market_status":"Unavailable: no audited pre-deadline prices; closing lines are excluded"}
    m=pred.merge(deadline_quotes,on="game_id",how="inner")
    m=m[m.home_win.notna() & m.deadline_market_home_prob.notna()].copy()
    if m.empty: return {"deadline_market_games":0,"deadline_market_status":"No matching verified pre-deadline odds"}
    m["hybrid_home_prob"]=(1-market_weight)*m.home_win_prob+market_weight*m.deadline_market_home_prob
    actual=m.home_win.astype(int)
    return {
        "deadline_market_games":int(len(m)),
        "deadline_market_status":"Valid timestamped pre-deadline quotes; compared on identical fixtures",
        "deadline_market_favorite_accuracy":float((m.deadline_market_home_prob>=.5).astype(int).eq(actual).mean()),
        "model_accuracy_on_deadline_games":float((m.home_win_prob>=.5).astype(int).eq(actual).mean()),
        "fixed_hybrid_accuracy_on_deadline_games":float((m.hybrid_home_prob>=.5).astype(int).eq(actual).mean()),
        "fixed_hybrid_market_weight":float(market_weight),
        "market_brier_on_deadline_games":float(np.mean((m.deadline_market_home_prob-actual)**2)),
        "model_brier_on_deadline_games":float(np.mean((m.home_win_prob-actual)**2)),
        "hybrid_brier_on_deadline_games":float(np.mean((m.hybrid_home_prob-actual)**2)),
        "selection_note":"Fixed predeclared blend; no weight tuned on these games",
    }


def live_calibration_oof(hist_games, seasons=(2024,2025)):
    """Build a *small*, past-season-only out-of-fold calibrator for live 2026.

    Each validation season is predicted by a model fitted on all prior seasons.
    No game outcomes from the held-out season enter the fitting data.
    """
    fs=feature_sets(); records=[]
    for season in seasons:
        train=hist_games[(hist_games.season<season)&hist_games.home_score.notna()].copy()
        test=hist_games[(hist_games.season==season)&hist_games.home_score.notna()].copy()
        if len(train)<250 or test.empty:continue
        w=live_training_weights(train,asof_season=season,asof_week=1)
        scores=[]
        for cols in fs.values():
            model=classifier().fit(prep(train,cols),train.home_win,logit__sample_weight=w)
            scores.append(model.predict_proba(prep(test,cols))[:,1])
        oo=test[["game_id","gameday","home_win"]].copy()
        oo["raw_home_prob"]=np.vstack(scores).mean(axis=0)
        records.append(oo)
    past=pd.concat(records,ignore_index=True) if records else pd.DataFrame()
    return fit_past_oof_calibrator(past)


def _calibration_error(prob, label, buckets=10):
    """Expected calibration error (ECE): *diagnostic*, not a fitted calibrator."""
    p=np.asarray(prob,dtype=float); y=np.asarray(label,dtype=float)
    mask=np.isfinite(p)&np.isfinite(y)
    if not mask.any(): return None
    p=p[mask]; y=y[mask]
    result=0.0
    for i in range(buckets):
        lo=i/buckets; hi=(i+1)/buckets
        keep=(p>=lo)&((p<=hi) if i==buckets-1 else (p<hi))
        if keep.any():result+=float(keep.mean())*abs(float(y[keep].mean())-float(p[keep].mean()))
    return float(result)


def evaluate(pred):
    if pred.empty: return {}
    games=pred.copy()
    decided=games[games.home_win.notna()].copy()
    decided=decided[decided.actual_margin.ne(0)]  # No outright ties in Pick'em evaluation.
    su_correct=decided.pick_home.eq(decided.home_win)
    unanimous=decided.model_votes_home.isin([0,3])
    ats_games=games[games.home_cover.notna() & games.ats_pick_home.notna()]
    ats_games=ats_games[ats_games.ats_edge_points_home.ne(0)]
    # Grade the distinct *ATS* selection. The old implementation graded pick_home
    # (straight-up) and the wrong-sign cover and was severely misleading.
    ats_correct=ats_games.ats_pick_home.eq(ats_games.home_cover)
    # Closing-market favourite is a retrospective benchmark, NOT an odds quote
    # known at a fixed earlier Pick'em deadline.
    favourite=decided[decided.spread_line.notna() & decided.spread_line.ne(0)]
    fav_accuracy=(favourite.spread_line.gt(0).astype(int).eq(favourite.home_win).mean() if not favourite.empty else None)
    p=decided.home_win_prob.to_numpy(dtype=float); y=decided.home_win.to_numpy(dtype=float)
    su=float(su_correct.mean()) if len(su_correct) else None
    ats=float(ats_correct.mean()) if len(ats_correct) else None
    by_season=[]
    for season,g in decided.groupby("season",sort=True):
        s=ats_games[ats_games.season.eq(season)]
        by_season.append({
            "season":int(season),"games":int(len(g)),
            "pickem_accuracy":float(g.pick_home.eq(g.home_win).mean()),
            "ats_games":int(len(s)),
            "ats_accuracy":float(s.ats_pick_home.eq(s.home_cover).mean()) if len(s) else None,
        })
    return {
        "games":int(len(decided)),
        "straight_up_accuracy":su,
        "unanimous_accuracy":float(su_correct[unanimous].mean()) if unanimous.any() else None,
        "unanimous_games":int(unanimous.sum()),
        "ats_rate":ats,
        "ats_games":int(len(ats_games)),
        "ats_target_hit":bool(ats is not None and ats>=TARGET_ATS_ACCURACY),
        "ats_research_target":TARGET_ATS_ACCURACY,
        "target_hit":bool(su is not None and su>=TARGET_ACCURACY),
        "target_pickem":TARGET_ACCURACY,
        "closing_market_favorite_accuracy":float(fav_accuracy) if fav_accuracy is not None else None,
        "closing_market_favorite_games":int(len(favourite)),
        "model_accuracy_on_closing_favorite_games":float(favourite.pick_home.eq(favourite.home_win).mean()) if not favourite.empty else None,
        "model_minus_closing_favorite_accuracy":float(favourite.pick_home.eq(favourite.home_win).mean()-fav_accuracy) if fav_accuracy is not None else None,
        "brier_score":float(np.mean((p-y)**2)) if len(p) else None,
        "expected_calibration_error":_calibration_error(p,y),
        "seasons":by_season,
        "ats_line_timing":"Historical closing lines, NOT archived lines at Pick'em deadline",
        "historical_odds_used_as_predictor":False,
        "raw_model_accuracy":float((decided.raw_home_prob.ge(.5).astype(int)==decided.home_win).mean()) if "raw_home_prob" in decided else None,
        "raw_model_brier":float(np.mean((decided.raw_home_prob.to_numpy(dtype=float)-y)**2)) if "raw_home_prob" in decided else None,
        "calibrated_predictions":int(decided.calibrator_enabled.sum()) if "calibrator_enabled" in decided else 0,
        "calibration_strategy":"Only prior out-of-fold predictions; 70/30 earlier-history gate by Brier",
        "ats_method":"Compare separate projected-margin ATS side with nflverse closing line; omit pushes",
    }

# =========================== LIVE PICKS ================================


def build_live_dataset(schedule, hist_games, qb, inj, travel, weather, odds):
    upcoming=schedule[(schedule.home_score.isna()) & (schedule.away_score.isna())].copy()
    if upcoming.empty:
        return pd.DataFrame()

    # Reconstruct last pregame state for each team from the LONG team-history
    # columns embedded in hist_games. For a home row use h_*; for an away row use a_*.
    teams=set(upcoming.home_team)|set(upcoming.away_team)
    team_state={}
    root_names=["off_epa","off_success","pass_epa","rush_epa","def_epa_allowed",
                "def_success_allowed","point_diff","win","turnover_margin"]
    completed=hist_games[hist_games.home_score.notna()].sort_values(["gameday","game_id"])
    for team in teams:
        parts=[]
        h=completed[completed.home_team.eq(team)].copy()
        if not h.empty:
            h=h.tail(1).iloc[0]
            for hl in HALF_LIVES:
                parts.append({f"{r}_ewm{hl}": h.get(f"h_{r}_ewm{hl}",np.nan) for r in root_names})
        a=completed[completed.away_team.eq(team)].copy()
        if not a.empty:
            a=a.tail(1).iloc[0]
            for hl in HALF_LIVES:
                parts.append({f"{r}_ewm{hl}": a.get(f"a_{r}_ewm{hl}",np.nan) for r in root_names})
        # Latest game representation wins, rather than averaging home/away rows.
        src_h=h if not completed[completed.home_team.eq(team)].empty else None
        src_a=a if not completed[completed.away_team.eq(team)].empty else None
        if src_h is not None and src_a is not None:
            src=src_h if pd.Timestamp(src_h.gameday) >= pd.Timestamp(src_a.gameday) else src_a
            prefix='h_' if src_h is src else 'a_'
        elif src_h is not None:
            src=src_h; prefix='h_'
        elif src_a is not None:
            src=src_a; prefix='a_'
        else:
            src=None; prefix=''
        state={}
        for r in root_names:
            for hl in HALF_LIVES:
                c=f"{r}_ewm{hl}"
                post_name=prefix+r+f"_post_ewm{hl}"
                state[c]=float(src[post_name]) if src is not None and pd.notna(src.get(post_name,np.nan)) else 0.0
        if src is not None:
            state['elo']=float(src.get(prefix+'post_elo',ELO_START)) if pd.notna(src.get(prefix+'post_elo',np.nan)) else ELO_START
        else:
            state['elo']=ELO_START
        team_state[team]=state

    rows=[]
    for _,g in upcoming.iterrows():
        r=g.to_dict(); hs=team_state[g.home_team]; aws=team_state[g.away_team]
        for root in root_names:
            for hl in HALF_LIVES:
                c=f"{root}_ewm{hl}"
                r[f"diff_{c}"]=hs[c]-aws[c]
        r["elo_diff"]=hs['elo']-aws['elo']
        rows.append(r)
    live=pd.DataFrame(rows)
    for src,name in [(qb,'qb'),(inj,'inj'),(travel,'travel'),(weather,'weather')]:
        if src is not None and not src.empty:
            live=live.merge(src,on='game_id',how='left')
    live=attach_market(live,odds)
    if "cutoff_spread" not in live:
        live["cutoff_spread"]=live.get("spread_home",np.nan)
    if "opening_spread" not in live:
        live["opening_spread"]=live.get("spread_home",np.nan)
    if "spread_move_open_to_cutoff" not in live:
        live["spread_move_open_to_cutoff"]=0.0
    live=add_model_diffs(live)
    # Neutralize missing live enrichment features rather than allowing NaN to
    # become an accidental advantage/disadvantage.
    market_cols={"spread_home","spread_price_home","total","over_price","home_ml","away_ml","bookmakers_used"}
    for c in live.columns:
        if c not in ['game_id','gameday','home_team','away_team'] and live[c].dtype.kind in 'fc':
            live[c]=live[c].replace([np.inf,-np.inf],np.nan)
            if c not in market_cols:
                live[c]=live[c].fillna(0.0)
    return live


def make_live_picks(live, hist_games, current_season=2026, current_week=None):
    fs=feature_sets(); probs=[]; votes=[]
    train=hist_games[hist_games.home_score.notna()].copy()
    if train.empty:raise ValueError("No completed games available for live model training.")
    selected_decay,weight_info=tune_pre_holdout_decay(hist_games)
    print(f"Season-decay tuning (2022-23 only): selected {selected_decay}; full details in optional backtest")
    train_w=live_training_weights(train,asof_season=current_season,asof_week=current_week,season_decay=selected_decay)
    for cols in fs.values():
        classifier_model=classifier().fit(prep(train,cols),train.home_win,logit__sample_weight=train_w)
        prediction=classifier_model.predict_proba(prep(live,cols))[:,1]
        probs.append(prediction); votes.append((prediction>=.5).astype(int))
    raw_p=np.vstack(probs).mean(axis=0); v=np.vstack(votes).sum(axis=0)
    calibrator,cal_meta=live_calibration_oof(hist_games)
    model_p=calibrated_probs(raw_p,calibrator)
    print(f"2026 calibration: {cal_meta}")
    margin_cols=list(dict.fromkeys(fs["MEDIUM"]+fs["SLOW"]))
    ridge=margin_model().fit(prep(train,margin_cols),train.actual_margin,ridge__sample_weight=train_w)
    projected=ridge.predict(prep(live,margin_cols))
    fields=["game_id","season","week","gameday","home_team","away_team",
            "spread_home","spread_price_home","spread_bookmaker","ml_bookmaker",
            "home_ml","away_ml","bookmakers_used","kickoff_utc","odds_observed_utc",
            "temp_f","wind_mph","precip_in","weather_severity"]
    out=live[[c for c in fields if c in live.columns]].copy()
    out["raw_model_home_prob"]=raw_p
    out["calibrated_model_home_prob"]=model_p
    out["calibrator_enabled"]=bool(calibrator is not None)
    nv=[no_vig_probs(h,a) for h,a in zip(out.home_ml,out.away_ml)]
    out["market_home_prob"]=[x[0] for x in nv]; out["market_away_prob"]=[x[1] for x in nv]
    marketp=out.market_home_prob.to_numpy(dtype=float)
    kick_pre=pd.to_datetime(out.get("kickoff_utc",pd.Series([pd.NaT]*len(out))),
                            errors="coerce",utc=True)
    now_utc=pd.Timestamp.now(tz="UTC")
    # Some feeds may include in-play lines. Never use those as a pre-kickoff quote.
    not_started=(kick_pre.isna() | (kick_pre>now_utc)).to_numpy(dtype=bool)
    has_market=np.isfinite(marketp) & (np.abs(marketp-0.5)>1e-9) & not_started
    market_fav=marketp>0.5
    model_fav=model_p>0.5
    deployed=v33_load_deployment_policy()
    print("V3.3 override mode:", "VALIDATED POLICY" if deployed is not None else
          ("EXPERIMENTAL (NOT VALIDATED)" if V33_EXPERIMENTAL_UPSETS else "MARKET FIRST; UNVALIDATED UPSETS BLOCKED"))
    settings=deployed if deployed is not None else {
        "min_confidence":V33_RESEARCH_MODEL_CONFIDENCE,
        "min_edge":V33_RESEARCH_PROB_EDGE,"min_votes":V33_RESEARCH_MIN_VOTES}
    candidates=v33_candidate_override(model_p,marketp,v,
          min_confidence=settings["min_confidence"],min_edge=settings["min_edge"],
          required_votes=settings["min_votes"])
    validated_flip=candidates if deployed is not None else np.zeros(len(live),dtype=bool)
    experimental_flip=(candidates & V33_EXPERIMENTAL_UPSETS & (deployed is None))
    actual_flip=validated_flip | experimental_flip
    chosen=np.where(has_market,np.where(actual_flip,model_fav,market_fav),model_fav)
    # A probability readout must NEVER contradict the displayed picked team.
    # The market-first probability normally blends market+model (75/25).
    # If the blend crosses to the wrong side of the chosen pick, use the
    # unadjusted market estimate; on actual upset overrides, use model probability.
    mix=np.where(has_market,MARKET_WEIGHT,0.0)
    baselinep=np.where(has_market,marketp,model_p)
    blended=np.clip(mix*baselinep+(1-mix)*model_p,0.0,1.0)
    fallback=has_market & ((blended>0.5)!=market_fav)
    effective=np.where(actual_flip,model_p,np.where(fallback,baselinep,blended))
    # Model-only means there is no bookmaker favourite; use calibrated model.
    effective=np.where(has_market,effective,model_p)
    out["home_win_prob"]=effective
    out["away_win_prob"]=1-effective
    out["pick_home"]=chosen.astype(int)
    out["model_votes_home"]=v
    out["model_only_pick_home"]=model_fav.astype(int)
    out["market_favorite_pick_home"]=np.where(has_market,market_fav.astype(int),np.nan)
    out["market_available"]=has_market
    out["market_blend_weight"]=mix
    out["candidate_upset"]=candidates
    out["validated_override"]=validated_flip
    out["experimental_override"]=experimental_flip
    out["pickem_method"]=np.select([
       ~has_market,validated_flip,experimental_flip,fallback
    ],["MODEL_ONLY_NO_MARKET","VALIDATED_MODEL_OVERRIDE",
       "EXPERIMENTAL_MODEL_OVERRIDE","MARKET_FAVOURITE_ODDS"],default="MARKET_FIRST_BLEND")
    out["probability_source"]=np.select([
       ~has_market,actual_flip,fallback
    ],["CALIBRATED_MODEL","CALIBRATED_MODEL_OVERRIDE","NO_VIG_MARKET"],default="MARKET_MODEL_BLEND")
    out["pickem_pick"]=np.where(out.pick_home.eq(1),out.home_team,out.away_team)
    out["confidence"]=np.where(out.pick_home.eq(1),effective,1-effective)
    out["projected_margin"]=projected
    out["snapshot_utc"]=datetime.now(timezone.utc).isoformat(timespec="seconds")
    if "kickoff_utc" not in out.columns: out["kickoff_utc"]=""
    kick=pd.to_datetime(out["kickoff_utc"],errors="coerce",utc=True)
    snap=pd.Timestamp(out["snapshot_utc"].iloc[0])
    out["pregame_snapshot_verified"]=kick.notna() & (snap<kick)
    out["pick_submission_status"]=np.where(kick.isna(),"KICKOFF_TIME_UNVERIFIED",
        np.where(snap<kick,"PRE_KICKOFF","AFTER_KICKOFF_NOT_SUBMITTABLE"))

    # ML betting edge uses the calibrated CORE-model probability versus the
    # sportsbook. Never call the blend itself evidence of a betting edge.
    model_edge=np.where(out.pick_home.eq(1),model_p-out.market_home_prob,1-model_p-out.market_away_prob)
    out["ml_edge"]=model_edge
    market_fav=np.where(out.market_home_prob>out.market_away_prob,1,np.where(out.market_home_prob<out.market_away_prob,0,np.nan))
    is_upset=actual_flip
    votes_for=np.where(out.pick_home.eq(1),v,3-v)
    out["pickem_signal"]=np.select([
        is_upset,
        (out.confidence>=PICKEM_STRONG_CONFIDENCE)&(votes_for==3),
        (out.confidence>=PICKEM_QUALIFIED_CONFIDENCE)&(votes_for>=2),
    ],["UPSET","STRONG","QUALIFIED"],default="LEAN")
    out["pickem_action"]="TAKE PICK"  # Every contest game always has an outright winner.
    out["ml_action"]="PASS ML"  # No validated wager profitability
    out["ml_research_signal"]=np.where((np.maximum(model_p,1-model_p)>=MIN_ML_CONFIDENCE)&
        np.isfinite(out.ml_edge)&(out.ml_edge>=MIN_ML_EDGE),"RESEARCH ML","NO ML EDGE")
    valid_spread=pd.to_numeric(out.spread_home,errors="coerce").notna()
    out["ats_edge_points_home"]=np.where(valid_spread,out.projected_margin+out.spread_home,np.nan)
    out["ats_pick_home"]=np.where(valid_spread,(out.ats_edge_points_home>=0).astype(int),np.nan)
    out["ats_pick"]=np.where(~valid_spread,"",np.where(out.ats_pick_home==1,out.home_team,out.away_team))
    out["ats_signal"]=np.select([
        valid_spread & (out.ats_edge_points_home.abs()>=STRONG_ATS_EDGE_POINTS),
        valid_spread & (out.ats_edge_points_home.abs()>=MIN_ATS_EDGE_POINTS),
    ],["STRONG","QUALIFIED"],default="PASS")
    out["ats_action"]=np.where(out.ats_signal.eq("PASS"),"PASS ATS",np.where(out.ats_pick_home==1,"BET HOME SPREAD","BET AWAY SPREAD"))
    out["signal_strength"]=out.pickem_signal  # preserve all four dashboard strength labels
    return out.sort_values(["gameday","confidence"],ascending=[True,False])

# ============================= PIPELINE =================================


def load_override_csv(path=QB_INJURY_CSV):
    p=Path(path)
    if not p.exists(): return pd.DataFrame()
    d=pd.read_csv(p)
    for c in d.columns:
        if c not in ["game_id","team","expected_qb"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    return d


def run_backtest():
    schedule=load_schedule(HISTORY_START,BACKTEST_END)
    pbp=load_pbp(range(HISTORY_START,BACKTEST_END+1))
    hist=to_games(build_team_history(pbp,schedule),schedule)
    # Closing spread is kept only to grade against the historical market.
    # It is NOT allowed to enter the classifier or margin regressions.
    hist=add_model_diffs(hist)
    chosen_decay,weight_info=tune_pre_holdout_decay(hist)
    pred=walk_forward(hist,holdout_decay=chosen_decay)
    rep=evaluate(pred)
    rep["season_weight_tuning"]=weight_info
    hold=pred[pred.season.isin([2024,2025])]
    rep["held_out_2024_2025_accuracy"]=float((hold.pick_home==hold.home_win).mean()) if len(hold) else None
    rep["held_out_2024_2025_games"]=int(len(hold))
    rep["held_out_2024_2025_brier"]=float(np.mean((hold.home_win_prob-hold.home_win)**2)) if len(hold) else None
    deadline_quotes=load_verified_deadline_odds()
    rep.update(fair_historical_market_diagnostics(pred,deadline_quotes))
    rep["market_first_override_policy"]=v33_select_override_policy(pred,deadline_quotes)
    rep["market_blend_method"]="Market-first with candidate model overrides; never tested on closing market prices"
    Path("nfl_model_v3_backtest.json").write_text(json.dumps(rep,indent=2,allow_nan=False))
    pred.to_csv("nfl_model_v3_backtest_predictions.csv",index=False)
    print(json.dumps(rep,indent=2))
    return rep,pred


def run_live():
    schedule=load_schedule(HISTORY_START,2026)
    current=schedule[(schedule.season==2026)&(schedule.home_score.isna())&(schedule.away_score.isna())].copy()
    # Keep the live feed focused on the current/upcoming weekly slate rather than
    # the entire remaining season. Use the first upcoming week in the schedule.
    if not current.empty:
        # Whole NFL week, including all Thursday-to-Monday games.
        first_week=int(current.week.min())
        current=current[current.week.eq(first_week)].copy()
    if current.empty:
        print("No upcoming regular-season games found in the schedule feed.")
        return pd.DataFrame()
    seasons=range(HISTORY_START,2027)
    pbp=load_pbp(seasons)
    team_hist=build_team_history(pbp,schedule)
    hist=to_games(team_hist,schedule)
    legacy=load_injuries_legacy(range(BACKTEST_START,min(BACKTEST_END,2024)+1))
    snaps=load_snap_counts(range(HISTORY_START,2026))
    depth=load_depth_charts(range(HISTORY_START,2027))
    override=load_override_csv()
    qb=build_qb_features(pbp,schedule,depth,legacy,override)
    inj=injury_features(schedule,legacy,snaps,override)
    tr=travel_features(schedule)
    weather=fetch_weather(current)
    odds=odds_api_current(current)
    live=build_live_dataset(current,hist,qb,inj,tr,weather,odds)
    current_week_num=int(pd.to_numeric(current["week"],errors="coerce").dropna().min()) if current["week"].notna().any() else 1
    completed=hist[hist.home_score.notna()].copy()
    picks=make_live_picks(live,completed,current_season=2026,current_week=current_week_num)
    picks.to_csv("nfl_model_v3_current_picks.csv",index=False)
    if V33_SAVE_SNAPSHOTS:
        v33_log_live_snapshots(picks)
    print(picks.to_string(index=False))
    return picks


def write_templates():
    if not Path("qb_injury_overrides.csv").exists():
        pd.DataFrame(columns=["game_id","team","expected_qb","qb_uncertainty","qb_injury_score",
                              "offense_impact","defense_impact","injury_count"]).to_csv("qb_injury_overrides.csv",index=False)
    print("Wrote qb_injury_overrides.csv template.")


if __name__=="__main__":
    mode=os.getenv("NFL_MODEL_MODE","live").lower()
    write_templates()
    if mode=="backtest": run_backtest()
    elif mode=="live": run_live()
    else: raise SystemExit("NFL_MODEL_MODE must be backtest or live")
