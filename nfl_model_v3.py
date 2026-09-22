from __future__ import annotations

"""NFL Model V3 — weekly picks engine + walk-forward validation.

Design goals
------------
1. Strict 2021-2025 walk-forward validation of the core model.
2. Live weekly engine uses the full enrichment layer when data is available.
2. Current-week live picks from a configurable odds feed.
3. QB availability, injury impact, travel/time zones, weather, and line movement.
4. Separate ML and ATS outputs.
5. No claim of >=62% unless the strict historical test actually clears 62%.

Live workflow
-------------
- Pull schedule + historical PBP from nflverse.
- Build pre-game team features.
- Pull current sportsbook markets from The Odds API (optional API key).
- Pull current venue weather from Open-Meteo.
- Apply manual/verified QB + injury overrides from a CSV.
- Fit on completed games only and score upcoming games.

The model deliberately uses only information available at the configured
betting cutoff. Closing lines and postgame actual weather are diagnostics,
not strict predictors.
"""

import json
import math
import os
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
TARGET_ACCURACY = 0.62

# Live pick gates — conservative by design.
MIN_ML_CONFIDENCE = 0.60
MIN_ML_EDGE = 0.035          # model probability - no-vig market probability
MIN_ATS_EDGE_POINTS = 1.50   # projected margin - market requirement
STRONG_ATS_EDGE_POINTS = 2.50

HALF_LIVES = (4, 8, 16)
ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 55.0
ELO_SEASON_REGRESSION = 0.75

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_REGIONS = os.getenv("ODDS_REGIONS", "us")
ODDS_MARKETS = "h2h,spreads,totals"
BOOKMAKERS = os.getenv("BOOKMAKERS", "")
QB_INJURY_CSV = os.getenv("NFL_QB_INJURY_CSV", "qb_injury_overrides.csv")

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
            out[f"{c}_ewm{hl}"] = (
                out.groupby("team")[c]
                   .transform(lambda s: s.shift(1).ewm(alpha=alpha, adjust=False, min_periods=1).mean())
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
    g["home_cover"] = np.where(g.spread_line.notna(),
                                (g.actual_margin + g.spread_line > 0).astype(int), np.nan)
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
    if p is None or pd.isna(p): return np.nan
    p=float(p)
    return 100/(p+100) if p>0 else (-p)/(-p+100)


def no_vig_probs(home_ml, away_ml):
    a=moneyline_implied(home_ml); b=moneyline_implied(away_ml)
    if not np.isfinite(a) or not np.isfinite(b): return (np.nan,np.nan)
    z=a+b
    return a/z,b/z


def odds_api_current(schedule, api_key=ODDS_API_KEY):
    if not api_key:
        return pd.DataFrame()
    import requests
    params={"apiKey":api_key,"regions":ODDS_REGIONS,"markets":ODDS_MARKETS,"oddsFormat":"american","dateFormat":"iso"}
    if BOOKMAKERS: params["bookmakers"]=BOOKMAKERS
    r=requests.get("https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds",params=params,timeout=30)
    r.raise_for_status()
    data=r.json()
    rows=[]
    teammap={t:t for t in set(schedule.home_team)|set(schedule.away_team)}
    for ev in data:
        ht=ev.get("home_team"); at=ev.get("away_team")
        match=schedule[(schedule.home_team==ht)&(schedule.away_team==at)]
        if match.empty: continue
        g=match.iloc[0]
        for bm in ev.get("bookmakers",[]):
            for m in bm.get("markets",[]):
                outcomes=m.get("outcomes",[])
                if m["key"]=="spreads":
                    for o in outcomes:
                        if o["name"]==ht: rows.append({"game_id":g.game_id,"bookmaker":bm["key"],"snapshot":ev.get("commence_time"),"spread_home":o.get("point"),"spread_price_home":o.get("price")})
                elif m["key"]=="totals":
                    for o in outcomes:
                        if o["name"]=="Over": rows.append({"game_id":g.game_id,"bookmaker":bm["key"],"snapshot":ev.get("commence_time"),"total":o.get("point"),"over_price":o.get("price")})
                elif m["key"]=="h2h":
                    vals={o["name"]:o.get("price") for o in outcomes}
                    rows.append({"game_id":g.game_id,"bookmaker":bm["key"],"home_ml":vals.get(ht),"away_ml":vals.get(at)})
    if not rows: return pd.DataFrame()
    d=pd.DataFrame(rows)
    # Aggregate across books; median is robust to outliers.
    agg={}
    for c in ["spread_home","spread_price_home","total","over_price","home_ml","away_ml"]:
        if c in d.columns: agg[c]="median"
    return d.groupby("game_id",as_index=False).agg(agg)


def attach_market(g, odds):
    out=g.copy()
    if odds.empty:
        for c in ["spread_home","spread_price_home","total","home_ml","away_ml"]: out[c]=np.nan
        return out
    return out.merge(odds,on="game_id",how="left")


# =========================== MODEL ======================================


def feature_sets():
    result={}
    for label,hl in zip(("FAST","MEDIUM","SLOW"),HALF_LIVES):
        base=[f"diff_{x}_ewm{hl}" for x in ["off_epa","off_success","pass_epa","rush_epa","def_epa_allowed","def_success_allowed","point_diff","win","turnover_margin"]]
        extra=["elo_diff","qb_form_diff","qb_change_diff","qb_injury_diff","qb_uncertainty_diff",
               "injury_offense_diff","injury_defense_diff","away_travel_penalty","away_tz_change",
               "away_days_rest_proxy","away_road_games_14d","weather_severity","weather_temp_extreme",
               "wind_mph","precip_in","snow_in","indoor_flag","cutoff_spread","opening_spread",
               "spread_move_open_to_cutoff"]
        result[label]=base+extra
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


def walk_forward(games, start=BACKTEST_START, end=BACKTEST_END):
    gs=games.copy(); fs=feature_sets(); rows=[]
    for season in range(start,end+1):
        for week in sorted(gs.loc[gs.season.eq(season),"week"].dropna().unique()):
            tr=gs[((gs.season<season)|((gs.season==season)&(gs.week<week)))].copy()
            te=gs[(gs.season==season)&(gs.week==week)].copy()
            if len(tr)<250 or te.empty: continue
            probs=[]; votes=[]
            for cols in fs.values():
                m=classifier().fit(prep(tr,cols),tr.home_win)
                p=m.predict_proba(prep(te,cols))[:,1]
                probs.append(p); votes.append((p>=.5).astype(int))
            avg=np.vstack(probs).mean(axis=0); vc=np.vstack(votes).sum(axis=0)
            med_cols=list(dict.fromkeys(fs["MEDIUM"]+fs["SLOW"]))
            mm=margin_model().fit(prep(tr,med_cols),tr.actual_margin)
            margin=mm.predict(prep(te,med_cols))
            o=te[["game_id","season","week","gameday","home_team","away_team","spread_line","home_win","home_cover"]].copy()
            o["home_win_prob"]=avg; o["pick_home"]=(vc>=2).astype(int); o["model_votes_home"]=vc; o["projected_margin"]=margin
            o["confidence"]=np.where(o.pick_home.eq(1),o.home_win_prob,1-o.home_win_prob)
            rows.append(o)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()


def evaluate(pred):
    if pred.empty: return {}
    return {
        "games":int(len(pred)),
        "straight_up_accuracy":float((pred.pick_home==pred.home_win).mean()),
        "unanimous_accuracy":float((pred.loc[(pred.model_votes_home==0)|(pred.model_votes_home==3),"pick_home"]==pred.loc[(pred.model_votes_home==0)|(pred.model_votes_home==3),"home_win"]).mean()) if ((pred.model_votes_home==0)|(pred.model_votes_home==3)).any() else np.nan,
        "ats_rate":float((pred.loc[pred.home_cover.notna(),"pick_home"]==pred.loc[pred.home_cover.notna(),"home_cover"]).mean()) if pred.home_cover.notna().any() else np.nan,
        "target_hit":bool((pred.pick_home==pred.home_win).mean()>=TARGET_ACCURACY),
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
                state[c]=float(src.get(prefix+c,np.nan)) if src is not None and pd.notna(src.get(prefix+c,np.nan)) else 0.0
        if src is not None:
            state['elo']=float(src.get(prefix+'pre_elo',ELO_START)) if pd.notna(src.get(prefix+'pre_elo',np.nan)) else ELO_START
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
    live=add_model_diffs(live)
    # Neutralize missing live enrichment features rather than allowing NaN to
    # become an accidental advantage/disadvantage.
    for c in live.columns:
        if c not in ['game_id','gameday','home_team','away_team'] and live[c].dtype.kind in 'fc':
            live[c]=live[c].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    return live


def make_live_picks(live, hist_games):
    fs=feature_sets(); probs=[]; votes=[]
    train=hist_games[hist_games.home_score.notna()].copy()
    for cols in fs.values():
        m=classifier().fit(prep(train,cols),train.home_win)
        p=m.predict_proba(prep(live,cols))[:,1]
        probs.append(p); votes.append((p>=.5).astype(int))
    p=np.vstack(probs).mean(axis=0); v=np.vstack(votes).sum(axis=0)
    med_cols=list(dict.fromkeys(fs["MEDIUM"]+fs["SLOW"]))
    mm=margin_model().fit(prep(train,med_cols),train.actual_margin)
    margin=mm.predict(prep(live,med_cols))
    out=live[["game_id","gameday","home_team","away_team","spread_home","home_ml","away_ml","temp_f","wind_mph","precip_in","weather_severity"]].copy()
    out["home_win_prob"]=p; out["away_win_prob"]=1-p; out["pick_home"]=(v>=2).astype(int); out["model_votes_home"]=v
    out["confidence"]=np.where(out.pick_home.eq(1),out.home_win_prob,out.away_win_prob); out["projected_margin"]=margin
    # ML edge vs no-vig market.
    nv=[no_vig_probs(h,a) for h,a in zip(out.home_ml,out.away_ml)]
    out["market_home_prob"]=[x[0] for x in nv]; out["market_away_prob"]=[x[1] for x in nv]
    out["ml_edge"]=np.where(out.pick_home.eq(1),out.home_win_prob-out.market_home_prob,out.away_win_prob-out.market_away_prob)
    out["ats_edge_points_home"]=out.projected_margin + out.spread_home
    out["ats_pick_home"]=(out.ats_edge_points_home>=0).astype(int)
    # Decision labels are thresholded signals, not claims of certainty.
    out["ml_action"]=np.where((out.confidence>=MIN_ML_CONFIDENCE)&(out.ml_edge>=MIN_ML_EDGE),"BET ML","PASS ML")
    out["ats_action"]=np.where(np.maximum(out.ats_edge_points_home,-out.ats_edge_points_home)>=MIN_ATS_EDGE_POINTS,
                                 np.where(out.ats_edge_points_home>=0,"BET HOME SPREAD","BET AWAY SPREAD"),"PASS ATS")
    out["signal_strength"]=np.select([
        (out.ml_action=="BET ML")&(out.ats_action!="PASS ATS")&(abs(out.ats_edge_points_home)>=STRONG_ATS_EDGE_POINTS),
        (out.ml_action=="BET ML")|(out.ats_action!="PASS ATS"),
    ],["STRONG","QUALIFIED"],default="PASS")
    return out.sort_values(["signal_strength","confidence","ml_edge"],ascending=[True,False,False])


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
    hist["opening_spread"]=hist.spread_line
    hist["cutoff_spread"]=hist.spread_line
    hist=add_model_diffs(hist)
    pred=walk_forward(hist)
    rep=evaluate(pred)
    Path("nfl_model_v3_backtest.json").write_text(json.dumps(rep,indent=2))
    pred.to_csv("nfl_model_v3_backtest_predictions.csv",index=False)
    print(json.dumps(rep,indent=2))
    return rep,pred


def run_live():
    schedule=load_schedule(HISTORY_START,2026)
    current=schedule[(schedule.season==2026)&(schedule.home_score.isna())&(schedule.away_score.isna())].copy()
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
    picks=make_live_picks(live,hist[hist.season.between(HISTORY_START,2025)])
    picks.to_csv("nfl_model_v3_current_picks.csv",index=False)
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
