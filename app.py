"""Flask web UI for FPL analysis tool."""

import json
import time
from pathlib import Path

import numpy as np
from flask import Flask, render_template, jsonify, request, redirect, url_for


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app = Flask(__name__)

# Ensure we run from project root
import os
os.chdir(Path(__file__).parent)


# Shared data cache - loaded once, reused across all requests
_cache = {
    "fb": None,
    "predictor": None,
    "pred_features": None,
    "client": None,
    "team_code_map": None,
    "loaded_at": 0,
}

CACHE_TTL = 3600  # refresh shared data every hour


def _load_shared():
    """Load shared data (players, model, predictions) - cached across requests."""
    from fpl.api import FPLClient
    from fpl.features import FeatureBuilder
    from fpl.model import PointsPredictor

    now = time.time()
    if _cache["fb"] and (now - _cache["loaded_at"]) < CACHE_TTL:
        return

    fb = FeatureBuilder()
    fb.load_data()

    predictor = PointsPredictor()
    predictor.load()

    # Single-GW predictions (for display)
    pred_features = fb.build_prediction_features()
    pred_features["predicted_points"] = predictor.predict(pred_features)

    # Multi-GW predictions (for transfer decisions)
    multi_gw = fb.build_prediction_features_multi_gw(n_gws=5)
    gw_predictions = {}
    for gw, gw_df in multi_gw.items():
        gw_df["predicted_points"] = predictor.predict(gw_df)
        gw_predictions[gw] = gw_df[["player_id", "predicted_points"]].copy()

    # Sum across GWs for each player
    import pandas as pd
    all_gw_preds = pd.concat(gw_predictions.values())
    multi_gw_totals = all_gw_preds.groupby("player_id")["predicted_points"].sum()

    # Add multi-GW total to pred_features
    pred_features["predicted_points_5gw"] = pred_features["player_id"].map(multi_gw_totals).fillna(0)

    client = FPLClient()
    bootstrap = client.get_bootstrap()
    team_code_map = {t["id"]: t["code"] for t in bootstrap["teams"]}

    _cache["fb"] = fb
    _cache["predictor"] = predictor
    _cache["pred_features"] = pred_features
    _cache["gw_predictions"] = gw_predictions
    _cache["client"] = client
    _cache["team_code_map"] = team_code_map
    _cache["loaded_at"] = now


def get_data(team_id: int) -> dict:
    """Load all data needed for the dashboard."""
    from fpl.optimizer import SquadOptimizer
    from fpl.transfers import TransferRecommender
    from fpl.chips import ChipAdvisor

    _load_shared()
    fb = _cache["fb"]
    predictor = _cache["predictor"]
    pred_features_1gw = _cache["pred_features"]
    client = _cache["client"]

    # Build prediction DataFrames for both horizons
    pred_1gw = pred_features_1gw.copy()
    pred_5gw = pred_features_1gw.copy()
    pred_5gw["predicted_points"] = pred_5gw["predicted_points_5gw"]

    # Generate transfer recs for both horizons
    recommender_5gw = TransferRecommender(client, predictor, pred_5gw)
    recommender_1gw = TransferRecommender(client, predictor, pred_1gw)

    info = recommender_5gw.get_current_squad(team_id)
    squad = info["squad"]
    recs_5gw = recommender_5gw.recommend_transfers(team_id, max_transfers=2)
    recs_1gw = recommender_1gw.recommend_transfers(team_id, max_transfers=2)

    # Starting lineup uses 1GW (you play one GW at a time)
    optimizer = SquadOptimizer(squad)
    lineup = optimizer.select_starting_11(squad)

    next_gw = fb._get_next_gw()

    # Chip analysis
    history = client.get_entry_history(team_id)
    chip_advisor = ChipAdvisor(
        history=history,
        fixtures_df=fb.fixtures,
        gameweeks_df=fb.gameweeks,
        squad_df=squad,
        prediction_df=pred_features_1gw,
    )
    chip_analysis = chip_advisor.full_analysis(info["free_transfers"])

    team_code_map = _cache["team_code_map"]

    # Top players by position (use 1GW data, JS handles sorting by horizon)
    pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    top_players = {}
    for pos_id, pos_name in pos_map.items():
        pos_df = pred_features_1gw[pred_features_1gw["element_type"] == pos_id]
        top = pos_df.nlargest(10, "predicted_points")
        top_players[pos_name] = [
            {
                "name": row["web_name"],
                "team": row.get("team_name", ""),
                "cost": round(row["now_cost"] / 10, 1),
                "xpts": round(row["predicted_points"], 1),
                "xpts_5gw": round(row.get("predicted_points_5gw", row["predicted_points"] * 5), 1),
                "form": round(float(row.get("form", row.get("rolling_3_points", 0))), 1),
                "xgi90": round(row.get("xgi_per90", 0), 2),
                "penalty": bool(row.get("is_penalty_taker", 0)),
            }
            for _, row in top.iterrows()
        ]

    # Format starters/bench
    cap_id = lineup["captain"].get("id") if hasattr(lineup["captain"], "get") else None
    vc_id = lineup["vice_captain"].get("id") if hasattr(lineup["vice_captain"], "get") else None

    def player_dict(row, role=""):
        team_id = int(row.get("team", 0))
        tc = team_code_map.get(team_id, 0)
        pos = pos_map.get(int(row["element_type"]), "?")
        suffix = "_1" if pos == "GKP" else ""
        kit_url = f"https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_{tc}{suffix}-110.webp"
        return {
            "id": int(row["id"]),
            "name": row["web_name"],
            "team": row.get("team_name", ""),
            "position": pos,
            "cost": round(row["now_cost"] / 10, 1),
            "xpts": round(row["predicted_points"], 1),
            "xpts_5gw": round(row.get("predicted_points_5gw", row["predicted_points"] * 5), 1),
            "role": role,
            "kit_url": kit_url,
        }

    starters = []
    for _, p in lineup["starters"].iterrows():
        role = ""
        if p.get("id") == cap_id:
            role = "C"
        elif p.get("id") == vc_id:
            role = "VC"
        starters.append(player_dict(p, role))

    bench = [player_dict(p) for _, p in lineup["bench"].iterrows()]

    # Format transfers for both horizons
    def format_recs(recs):
        formatted = []
        for rec in recs:
            formatted.append({
                "n_transfers": rec["n_transfers"],
                "hits": rec["hits"],
                "hit_cost": rec["hit_cost"],
                "points_gain": rec["points_gain"],
                "bank_after": round(rec["bank_after"] / 10, 1),
                "raw_gain": round(rec["points_gain"] + rec["hit_cost"], 1),
                "worth_it": rec["points_gain"] > 0,
                "out": [
                    {
                        "name": p["web_name"],
                        "team": p.get("team_name", ""),
                        "position": pos_map.get(int(p["element_type"]), "?"),
                        "cost": round(p.get("selling_price", p["now_cost"]) / 10, 1),
                        "xpts": round(p["predicted_points"], 1),
                    }
                    for p in rec["transfers_out"]
                ],
                "in": [
                    {
                        "name": p["web_name"],
                        "team": p.get("team_name", ""),
                        "position": pos_map.get(int(p["element_type"]), "?"),
                        "cost": round(p["now_cost"] / 10, 1),
                        "xpts": round(p["predicted_points"], 1),
                    }
                    for p in rec["transfers_in"]
                ],
                "reasons": rec.get("reasons", []),
            })
        return formatted

    formatted_recs_1gw = format_recs(recs_1gw)
    formatted_recs_5gw = format_recs(recs_5gw)

    # Format chips
    chips_available = {}
    for key, chip in chip_analysis["chips_available"].items():
        chips_available[key] = {
            "name": chip["name"],
            "available": not chip["used"],
            "used_gw": chip.get("used_gw"),
        }

    chip_recs = {}
    for key, analysis in chip_analysis.get("analyses", {}).items():
        best = analysis.get("best_gw", analysis.get("recommendations", [{}])[0])
        chip_recs[key] = {
            "best_gw": best.get("gw"),
            "score": best.get("score", 0),
            "reasoning": best.get("reasoning", []),
        }

    # GW schedule (DGW/BGW)
    gw_schedule = []
    for gw_id, gw_info in sorted(chip_analysis.get("gw_schedule", {}).items()):
        if gw_info["type"] != "normal":
            gw_schedule.append({
                "gw": gw_id,
                "type": gw_info["type"],
                "double_teams": len(gw_info.get("double_teams", set())),
                "blank_teams": len(gw_info.get("blank_teams", set())),
            })

    return {
        "team_name": info["team_name"],
        "overall_rank": info["overall_rank"],
        "total_points": info["total_points"],
        "bank": round(info["bank"] / 10, 1),
        "free_transfers": info["free_transfers"],
        "next_gw": next_gw,
        "starters": starters,
        "bench": bench,
        "transfers_1gw": formatted_recs_1gw,
        "transfers_5gw": formatted_recs_5gw,
        "top_players": top_players,
        "chips_available": chips_available,
        "chip_recommendations": chip_recs,
        "chip_this_week": chip_analysis.get("this_week", {}),
        "gw_schedule": gw_schedule,
    }


@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/team/<int:team_id>")
def dashboard(team_id):
    try:
        data = get_data(team_id)
    except Exception as e:
        return render_template("landing.html", error=str(e)), 400
    return render_template("index.html", data=data, data_json=json.dumps(data, cls=NumpyEncoder), team_id=team_id)


@app.route("/api/data/<int:team_id>")
def api_data(team_id):
    return jsonify(get_data(team_id))


@app.route("/api/simulate-transfer", methods=["POST"])
def simulate_transfer():
    """Simulate transferring players out and find best replacements + new lineup."""
    from fpl.optimizer import SquadOptimizer
    from fpl.transfers import TransferRecommender

    body = request.get_json()
    team_id = body.get("team_id", 9183439)
    sell_ids = body.get("sell_ids", [])
    force_buys = body.get("force_buys", [])
    gw_horizon = body.get("horizon", 5)

    if not sell_ids:
        return jsonify({"error": "No players selected to transfer out"}), 400

    _load_shared()
    pred_features = _cache["pred_features"].copy()
    client = _cache["client"]

    # Use 5GW predictions for ranking if horizon > 1
    if gw_horizon > 1 and "predicted_points_5gw" in pred_features.columns:
        pred_features["predicted_points"] = pred_features["predicted_points_5gw"]

    recommender = TransferRecommender(client, _cache["predictor"], pred_features)
    info = recommender.get_current_squad(team_id)
    squad = info["squad"]

    pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

    # Find the players being sold
    sell_players = squad[squad["id"].isin(sell_ids)]
    remaining = squad[~squad["id"].isin(sell_ids)]

    sell_value = sell_players["selling_price"].sum()
    available_budget = info["bank"] + int(sell_value)

    team_counts = remaining["team"].value_counts().to_dict()
    remaining_ids = set(int(x) for x in remaining["id"])

    # Find best replacement for each sold player
    replacements = []
    budget_left = available_budget
    already_bought_ids = set()

    for sell_idx, (_, sell_row) in enumerate(sell_players.iterrows()):
        pos = int(sell_row["element_type"])
        candidates = pred_features[
            (pred_features["element_type"] == pos)
            & (~pred_features["id"].isin(remaining_ids))
            & (~pred_features["id"].isin(already_bought_ids))
            & (pred_features["predicted_points"] > 0)
            & (pred_features["now_cost"] <= budget_left)
        ].copy()

        # Team constraint
        candidates = candidates[
            candidates["team"].map(lambda t: team_counts.get(int(t), 0) < 3)
        ]

        # Top 5 options for this slot
        top_options = candidates.nlargest(5, "predicted_points")

        options = []
        for _, opt in top_options.iterrows():
            options.append({
                "id": int(opt["id"]),
                "name": opt["web_name"],
                "team": opt.get("team_name", ""),
                "position": pos_map.get(int(opt["element_type"]), "?"),
                "cost": round(opt["now_cost"] / 10, 1),
                "xpts": round(opt["predicted_points"], 1),
                "form": round(float(opt.get("form", opt.get("rolling_3_points", 0))), 1),
                "xgi90": round(opt.get("xgi_per90", 0), 2),
                "penalty": bool(opt.get("is_penalty_taker", 0)),
            })

        # Select: use force_buy if provided, otherwise auto-select best
        forced = None
        if sell_idx < len(force_buys) and force_buys[sell_idx]:
            forced_id = force_buys[sell_idx]
            forced_match = candidates[candidates["id"] == forced_id]
            if not forced_match.empty:
                forced = forced_match.iloc[0]

        pick = forced if forced is not None else (top_options.iloc[0] if not top_options.empty else None)

        if pick is not None:
            selected = {
                "id": int(pick["id"]),
                "name": pick["web_name"],
                "team": pick.get("team_name", ""),
                "position": pos_map.get(int(pick["element_type"]), "?"),
                "cost": round(pick["now_cost"] / 10, 1),
                "xpts": round(pick["predicted_points"], 1),
            }
            budget_left -= int(pick["now_cost"])
            remaining_ids.add(int(pick["id"]))
            already_bought_ids.add(int(pick["id"]))
            t = int(pick["team"])
            team_counts[t] = team_counts.get(t, 0) + 1
        else:
            selected = None

        replacements.append({
            "selling": {
                "id": int(sell_row["id"]),
                "name": sell_row["web_name"],
                "team": sell_row.get("team_name", ""),
                "position": pos_map.get(int(sell_row["element_type"]), "?"),
                "cost": round(sell_row.get("selling_price", sell_row["now_cost"]) / 10, 1),
                "xpts": round(sell_row["predicted_points"], 1),
            },
            "selected": selected,
            "options": options,
        })

    # Build new squad with replacements applied
    new_squad_ids = list(remaining_ids)
    new_squad = pred_features[pred_features["id"].isin(new_squad_ids)].copy()

    if len(new_squad) == 15:
        optimizer = SquadOptimizer(new_squad)
        lineup = optimizer.select_starting_11(new_squad)

        cap_id = lineup["captain"].get("id") if hasattr(lineup["captain"], "get") else None
        vc_id = lineup["vice_captain"].get("id") if hasattr(lineup["vice_captain"], "get") else None

        def player_dict(row, role=""):
            team_id = int(row.get("team", 0))
            tc = _cache["team_code_map"].get(team_id, 0)
            pos = pos_map.get(int(row["element_type"]), "?")
            suffix = "_1" if pos == "GKP" else ""
            kit_url = f"https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_{tc}{suffix}-110.webp"
            return {
                "id": int(row["id"]),
                "name": row["web_name"],
                "team": row.get("team_name", ""),
                "position": pos,
                "cost": round(row["now_cost"] / 10, 1),
                "xpts": round(row["predicted_points"], 1),
                "role": role,
                "kit_url": kit_url,
            }

        starters = []
        for _, p in lineup["starters"].iterrows():
            role = ""
            if p.get("id") == cap_id:
                role = "C"
            elif p.get("id") == vc_id:
                role = "VC"
            starters.append(player_dict(p, role))

        bench = [player_dict(p) for _, p in lineup["bench"].iterrows()]
    else:
        starters = []
        bench = []

    result = {
        "replacements": replacements,
        "starters": starters,
        "bench": bench,
        "bank_remaining": round(budget_left / 10, 1),
        "free_transfers": info["free_transfers"],
        "hits": max(0, len(sell_ids) - info["free_transfers"]),
    }
    return app.response_class(
        response=json.dumps(result, cls=NumpyEncoder),
        mimetype="application/json",
    )


# Custom JSON provider to handle numpy types
from flask.json.provider import DefaultJSONProvider

class CustomJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app.json_provider_class = CustomJSONProvider
app.json = CustomJSONProvider(app)


if __name__ == "__main__":
    app.run(debug=True, port=5050)
