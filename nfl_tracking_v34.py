#!/usr/bin/env python3
"""NFL Model V3.4: immutable pre-kickoff observations and fair market comparison.

Runs AFTER nfl_model_v3.py, or in preflight mode before the expensive live model.
Only quotes and picks demonstrably observed before a known kickoff are archived.
The most recent valid pre-kickoff pick is the locked pick; no after-kickoff swaps.
GitHub Actions must commit these files; local runner storage is not persistent.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime, time, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

SCHEDULE_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
ROOT = Path(os.getenv("NFL_DATA_DIR", "data"))
MODEL_CSV = Path(os.getenv("NFL_LIVE_CSV", "nfl_model_v3_current_picks.csv"))
NEAR_KICKOFF_MIN = int(os.getenv("NFL_NEAR_KICKOFF_MIN", "50"))
MIN_REFRESH_MIN = int(os.getenv("NFL_MIN_REFRESH_MIN", "24"))
QUOTE_AGE_LIMIT_MIN = int(os.getenv("NFL_QUOTE_AGE_LIMIT_MIN", "180"))
MATCH_YEAR = int(os.getenv("NFL_SEASON", "2026"))
VALID_LABELS = {"STRONG", "QUALIFIED", "UPSET", "LEAN"}


def utc_timestamp(value):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        x = pd.Timestamp(value)
        if pd.isna(x) or x.tzinfo is None:
            return None
        return x.tz_convert("UTC").to_pydatetime()
    except (ValueError, TypeError, OverflowError):
        return None


def iso(t):
    return t.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z") if t else None


def finite(v):
    if v is None: return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def moneyline_prob(ml):
    ml = finite(ml)
    if ml is None or ml == 0: return None
    return (100.0 / (100.0 + ml)) if ml > 0 else ((-ml) / (-ml + 100.0))


def market_prob(home_ml, away_ml):
    a, b = moneyline_prob(home_ml), moneyline_prob(away_ml)
    return a / (a + b) if a is not None and b is not None and a + b > 0 else None


def clean(obj):
    if isinstance(obj, dict): return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [clean(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, str): return obj
    if isinstance(obj, int): return int(obj)
    if isinstance(obj, float): return obj if math.isfinite(obj) else None
    if isinstance(obj, pd.Timestamp): return obj.isoformat()
    if hasattr(obj, "item"): return clean(obj.item())
    return str(obj)


def read_json(path, default):
    if not path.exists(): return default
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError(f"Refusing to overwrite unreadable archive: {path}: {exc}") from exc


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def read_jsonl(path):
    if not path.exists(): return []
    out = []
    for num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try: out.append(json.loads(line))
            except ValueError as exc:
                raise RuntimeError(f"Refusing to modify invalid archive {path}, line {num}") from exc
    return out


def append_jsonl(path, new_rows, key_fields):
    old = read_jsonl(path)
    keys = {tuple(row.get(k) for k in key_fields) for row in old}
    added = []
    for row in new_rows:
        key = tuple(row.get(k) for k in key_fields)
        if key not in keys:
            added.append(clean(row))
            keys.add(key)
    if added:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for row in added: f.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
    return len(added)


def schedule_kickoff(g):
    """nflverse gameday + gametime convention uses US/Eastern calendar time.

    Do NOT assume midnight on gameday. If `gametime` is missing, no lock.
    An exact, matched Odds API commence_time takes precedence when available.
    """
    try:
        day = pd.Timestamp(g["gameday"])
        if pd.isna(day): return None
        clock = str(g.get("gametime", "")).strip()
        if len(clock) < 4 or clock.lower() in ("nan", "none", "nat"): return None
        hh, mm = clock.split(":")[:2]
        if not (0 <= int(hh) <= 23 and 0 <= int(mm) <= 59): return None
        t = datetime.combine(day.date(), time(hour=int(hh), minute=int(mm)), tzinfo=ZoneInfo("America/New_York"))
        return t.astimezone(timezone.utc)
    except (KeyError, ValueError, TypeError):
        return None


def kickoff_for_pick(pick, game_by_id):
    direct = utc_timestamp(pick.get("kickoff_utc"))
    if direct is not None: return direct, "odds_api_commence_time"
    source = game_by_id.get(str(pick.get("game_id")))
    fallback = schedule_kickoff(source) if source else None
    return fallback, "nflverse_eastern_kickoff" if fallback else "UNVERIFIED"


def schedule_games(schedule):
    if schedule is None: return {}
    if isinstance(schedule, pd.DataFrame):
        return {str(r.get("game_id")): r.to_dict() for _, r in schedule.iterrows()}
    return {str(r.get("game_id")): dict(r) for r in schedule}


def fetch_schedule():
    return pd.read_csv(os.getenv("NFL_SCHEDULE_URL", SCHEDULE_URL), low_memory=False)


def upcoming_kickoffs(schedule, now):
    matches=[]
    for game in schedule_games(schedule).values():
        if finite(game.get("season")) != MATCH_YEAR or str(game.get("game_type")) != "REG": continue
        if finite(game.get("home_score")) is not None and finite(game.get("away_score")) is not None: continue
        ko = schedule_kickoff(game)
        if ko and ko > now:
            matches.append((str(game["game_id"]), ko))
    return sorted(matches, key=lambda x:x[1])


def decide_preflight(schedule, now, is_manual, is_daily, snap_rows):
    if is_manual: return True, "manual"
    future=upcoming_kickoffs(schedule, now)
    if not future:return False, "no upcoming 2026 regular-season games"
    if is_daily: return True, "daily-refresh"
    recent={}
    for row in snap_rows:
        t=utc_timestamp(row.get("prediction_observed_utc"))
        if t and (gid := row.get("game_id")):
            if gid not in recent or t > recent[gid]: recent[gid]=t
    for game_id, kick in future:
        mins=(kick-now).total_seconds()/60
        if 0 < mins <= NEAR_KICKOFF_MIN:
            previous=recent.get(game_id)
            if previous is None or (now-previous).total_seconds()/60 >= MIN_REFRESH_MIN:
                return True, f"pregame {game_id} {mins:.1f}m before kickoff"
    return False, "no games within pre-kickoff window"


def preflight(args):
    schedule=fetch_schedule()
    now=datetime.now(timezone.utc)
    snap_rows=read_jsonl(ROOT/"prediction_snapshots.jsonl")
    do_it, why=decide_preflight(schedule,now,args.event=="workflow_dispatch",args.schedule=="19 8 * * *",snap_rows)
    print(f"V3.4 preflight: {'RUN' if do_it else 'SKIP'} — {why}")
    envpath=os.getenv("GITHUB_OUTPUT")
    if envpath:
        with open(envpath,"a",encoding="utf-8") as f:
            f.write(f"run_model={'true' if do_it else 'false'}\n")
            f.write(f"reason={why}\n")
    return do_it


def snapshot_from_pick(p, games, processing_time):
    gid=str(p.get("game_id") or "")
    if not gid:return None
    ko, source=kickoff_for_pick(p,games)
    if ko is None:return None  # Can't truthfully determine whether the pick was pregame.
    decision=utc_timestamp(p.get("snapshot_utc")) or processing_time
    if not (decision < ko) or decision > processing_time + timedelta(minutes=2):return None
    quote=utc_timestamp(p.get("odds_observed_utc"))
    valid_quote=bool(quote and quote <= processing_time + timedelta(minutes=2) and
                     quote <= decision and quote < ko and
                     (decision-quote).total_seconds()/60 <= QUOTE_AGE_LIMIT_MIN)
    pmarket=market_prob(p.get("home_ml"),p.get("away_ml")) if valid_quote else None
    if pmarket is not None and abs(pmarket-.5) < 1e-12: pmarket=None  # No unique market favourite.
    picked=int(finite(p.get("pick_home")) or 0)
    modelp=finite(p.get("calibrated_model_home_prob"))
    lead=(ko-decision).total_seconds()/60
    quote_lead=(ko-quote).total_seconds()/60 if valid_quote else None
    quality="NEAR_KICKOFF" if lead <= 60 and (pmarket is None or (quote_lead is not None and quote_lead <= 60)) else "EARLY_PREGAME"
    signal=str(p.get("pickem_signal") or "LEAN").upper()
    if signal not in VALID_LABELS: signal="LEAN"
    return {
      "game_id":gid,"season":p.get("season"),"week":p.get("week"),"gameday":str(p.get("gameday") or ""),
      "home_team":p.get("home_team"),"away_team":p.get("away_team"),
      "kickoff_utc":iso(ko),"kickoff_source":source,
      "prediction_observed_utc":iso(decision),"quote_observed_utc":iso(quote) if valid_quote else None,
      "lead_minutes":round(lead,2),"lock_quality":quality,
      "pick_home":picked,"pickem_pick":p.get("pickem_pick"),
      "pickem_signal":signal,"pickem_method":p.get("pickem_method"),
      "confidence":finite(p.get("confidence")),"model_home_prob":modelp,
      "market_home_prob":pmarket,"market_favorite_pick_home":int(pmarket>.5) if pmarket is not None else None,
      "market_quote_usable":pmarket is not None,"home_ml":finite(p.get("home_ml")) if valid_quote else None,
      "away_ml":finite(p.get("away_ml")) if valid_quote else None,
      "ml_bookmaker":p.get("ml_bookmaker") if valid_quote else None,
      "spread_home":finite(p.get("spread_home")) if valid_quote else None,
      "spread_bookmaker":p.get("spread_bookmaker") if valid_quote else None,
      "projected_margin":finite(p.get("projected_margin")),
      "ats_action":str(p.get("ats_action") or "PASS ATS"),
      "ats_signal":p.get("ats_signal"),
      "candidate_upset":bool(p.get("candidate_upset")) if isinstance(p.get("candidate_upset"),bool) else str(p.get("candidate_upset")).lower()=="true",
      "verified_pre_kickoff":True,
    }


def rate(records, field):
    statuses=[x.get(field) for x in records if x.get(field) in ("WIN","LOSS","PUSH")]
    w=sum(x=="WIN" for x in statuses); l=sum(x=="LOSS" for x in statuses); p=sum(x=="PUSH" for x in statuses)
    return {"wins":w,"losses":l,"pushes":p,"bets":w+l,"pct":(w/(w+l) if w+l else None)}


def score_locked(locked, games):
    for r in locked.values():
        gid=str(r.get("game_id"))
        game=games.get(gid)
        if not game:continue
        home,away=finite(game.get("home_score")),finite(game.get("away_score"))
        if home is None or away is None:continue
        margin=home-away
        r["actual_home_score"]=home;r["actual_away_score"]=away
        if margin != 0:
            r["pickem_result"]="WIN" if bool(r.get("pick_home")) == (margin>0) else "LOSS"
            fav=r.get("market_favorite_pick_home")
            if fav is not None and r.get("market_quote_usable"):
                r["market_result"]="WIN" if bool(fav)==(margin>0) else "LOSS"
            if r.get("model_home_prob") is not None:
                r["model_only_result"]="WIN" if (float(r["model_home_prob"])>=.5)==(margin>0) else "LOSS"
        action=r.get("ats_action")
        spread=finite(r.get("spread_home"))
        if spread is not None and action in ("BET HOME SPREAD","BET AWAY SPREAD"):
            cover=margin+spread
            if cover == 0:r["ats_result"]="PUSH"
            else:r["ats_result"]="WIN" if ((cover>0)==(action=="BET HOME SPREAD")) else "LOSS"


def make_record(locked, now):
    rows=list(locked.values())
    comparable=[r for r in rows if r.get("pickem_result") in ("WIN","LOSS") and r.get("market_result") in ("WIN","LOSS")]
    market_w=sum(r["market_result"]=="WIN" for r in comparable)
    model_w=sum(r["pickem_result"]=="WIN" for r in comparable)
    comparison={"games":len(comparable),"market_wins":market_w,"model_wins":model_w,
        "net_correct_vs_favorites":model_w-market_w,
        "market_accuracy":market_w/len(comparable) if comparable else None,
        "model_accuracy_same_games":model_w/len(comparable) if comparable else None,
        "note":"Identical game sample, frozen pre-kickoff bookmaker quotes and picks"}
    return {"targets":{"pickem":.68,"ats":.64,"ats_research":.64},"updated_utc":iso(now),
      "2026_live":{
        "pickem_all":rate(rows,"pickem_result"),
        "pickem_qualified":rate([r for r in rows if r.get("pickem_signal") in ("STRONG","QUALIFIED")],"pickem_result"),
        "pickem_strong":rate([r for r in rows if r.get("pickem_signal")=="STRONG"],"pickem_result"),
        "ats_qualified":rate([r for r in rows if r.get("ats_signal") in ("STRONG","QUALIFIED")],"ats_result"),
        "ats_strong":rate([r for r in rows if r.get("ats_signal")=="STRONG"],"ats_result"),
        "market_first_comparison":comparison,
      },
      "tracking":{"locked_games":len(rows),"archived_quotes":len([r for r in rows if r.get("market_quote_usable")]),
                  "near_kickoff_locks":len([r for r in rows if r.get("lock_quality")=="NEAR_KICKOFF"]),
                  "verified_comparison_games":len(comparable)}}


def export_verified_deadline_odds(root, locked):
    """Exports real observed quotes in V3.3's historical import schema.

    Prospective 2026 data only. It cannot validate historical 2021-25 runs.
    """
    path=root/"pregame_deadline_odds_2026.csv"
    cols=["game_id","snapshot_utc","deadline_utc","kickoff_utc","home_ml","away_ml"]
    rows=[]
    for row in locked.values():
        if not row.get("market_quote_usable"): continue
        quote=utc_timestamp(row.get("quote_observed_utc"))
        deadline=utc_timestamp(row.get("prediction_observed_utc"))
        kickoff=utc_timestamp(row.get("kickoff_utc"))
        if not all((quote,deadline,kickoff)) or not (quote<=deadline<kickoff):continue
        if finite(row.get("home_ml")) is None or finite(row.get("away_ml")) is None:continue
        rows.append({"game_id":row["game_id"],"snapshot_utc":iso(quote),
             "deadline_utc":iso(deadline),"kickoff_utc":iso(kickoff),
             "home_ml":row["home_ml"],"away_ml":row["away_ml"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(".csv.tmp")
    with temp.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=cols)
        writer.writeheader()
        writer.writerows(sorted(rows,key=lambda x:x["game_id"]))
    temp.replace(path)
    return len(rows)


def update_model_archive(df, schedule, now=None, directory=None):
    now=now or datetime.now(timezone.utc)
    root=Path(directory or ROOT);root.mkdir(parents=True,exist_ok=True)
    games=schedule_games(schedule)
    feed=clean(df.to_dict(orient="records"))
    write_json(root/"current_picks.json",feed)
    old=read_json(root/"locked_picks.json",[])
    if not isinstance(old,list):raise RuntimeError("locked_picks.json must be a list; refusing overwrite")
    locked={str(x["game_id"]):x for x in old}
    new_snaps=[];new_market=[]
    for p in feed:
        obs=snapshot_from_pick(p,games,now)
        if obs is None:continue
        new_snaps.append(obs)
        if obs["market_quote_usable"]:
            new_market.append({k:obs.get(k) for k in (
                "game_id","season","week","home_team","away_team","kickoff_utc",
                "quote_observed_utc","prediction_observed_utc","home_ml","away_ml",
                "ml_bookmaker","market_home_prob","spread_home","spread_bookmaker")})
        previous=locked.get(obs["game_id"])
        # Only a genuinely newer observation before kickoff can change the lock.
        if previous is None or (utc_timestamp(obs["prediction_observed_utc"]) > utc_timestamp(previous["prediction_observed_utc"])):
            locked[obs["game_id"]]=obs
    counts={
      "prediction_snapshots":append_jsonl(root/"prediction_snapshots.jsonl",new_snaps,("game_id","prediction_observed_utc")),
      "odds_snapshots":append_jsonl(root/"odds_snapshots.jsonl",new_market,("game_id","quote_observed_utc","ml_bookmaker"))}
    score_locked(locked,games)
    ordered=sorted(locked.values(),key=lambda r:(str(r.get("kickoff_utc")),str(r.get("game_id"))))
    write_json(root/"locked_picks.json",ordered)
    write_json(root/"verified_pick_history.json",ordered)
    write_json(root/"record.json",make_record(locked,now))
    exported=export_verified_deadline_odds(root,locked)
    return {**counts,"locked":len(ordered),"feed":len(feed),"deadline_odds_rows":exported}


def update(args):
    if not MODEL_CSV.exists():raise FileNotFoundError(f"Missing {MODEL_CSV}: run nfl_model_v3.py first")
    df=pd.read_csv(MODEL_CSV)
    schedule=fetch_schedule()
    counts=update_model_archive(df,schedule)
    print("V3.4 feed/archives:",json.dumps(counts))


def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="command",required=True)
    before=sub.add_parser("preflight")
    before.add_argument("--event",default="schedule")
    before.add_argument("--schedule",default="")
    sub.add_parser("update")
    args=ap.parse_args()
    if args.command=="preflight":preflight(args)
    else:update(args)


if __name__=="__main__":main()
