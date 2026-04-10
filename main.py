"""
FPL Analysis Tool

Usage:
  python main.py pull                  Pull fresh data from FPL API
  python main.py train                 Train prediction models
  python main.py predict [--pos POS]   Show top players by predicted next-GW points
  python main.py squad [--budget N]    Build optimal squad from scratch
  python main.py transfers TEAM_ID     Recommend transfers for a team
  python main.py team TEAM_ID          Show current team with predictions
"""

import argparse
import sys
from pathlib import Path

# Ensure we run from project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.chdir(PROJECT_ROOT)


def cmd_pull(args):
    from fpl.ingest import DataIngestor
    print("Pulling data from FPL API...")
    ingestor = DataIngestor()
    ingestor.pull_all(verbose=True)


def cmd_train(args):
    from fpl.features import FeatureBuilder
    from fpl.model import PointsPredictor

    print("Building features...")
    fb = FeatureBuilder()
    fb.load_data()
    features = fb.build_training_features()
    feature_cols = fb.get_feature_columns()

    print(f"Training on {len(features)} rows with {len(feature_cols)} features...")
    predictor = PointsPredictor()
    metrics = predictor.train(features, feature_cols)

    print("\nModel evaluation:")
    for pos, m in metrics.items():
        print(f"  {pos}: MAE={m['mae']}, RMSE={m['rmse']}, r={m['corr']}")

    print("\nModels saved to models/")


def cmd_predict(args):
    from fpl.features import FeatureBuilder
    from fpl.model import PointsPredictor

    fb = FeatureBuilder()
    fb.load_data()

    predictor = PointsPredictor()
    predictor.load()

    pred_features = fb.build_prediction_features()
    pred_features["predicted_points"] = predictor.predict(pred_features)

    pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

    # Filter by position if specified
    df = pred_features.copy()
    if args.pos:
        pos_id = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}.get(args.pos.upper())
        if pos_id:
            df = df[df["element_type"] == pos_id]

    # Top N
    top = df.nlargest(args.top, "predicted_points")

    next_gw = fb._get_next_gw()
    print(f"\nTop {args.top} players by predicted points (GW{next_gw}):")
    print(f"{'Pos':<5} {'Player':<22} {'Team':<5} {'Cost':>6} {'xPts':>6} "
          f"{'Form':>5} {'xGI/90':>7} {'Pen':>4}")
    print("-" * 70)

    for _, p in top.iterrows():
        pen = "Y" if p.get("is_penalty_taker", 0) else ""
        form = p.get("form", p.get("rolling_3_points", 0))
        xgi90 = p.get("xgi_per90", 0)
        print(
            f"{pos_map.get(int(p['element_type']), '?'):<5} "
            f"{p['web_name']:<22} "
            f"{p.get('team_name', ''):<5} "
            f"{p['now_cost'] / 10:>5.1f}m "
            f"{p['predicted_points']:>5.1f} "
            f"{float(form):>5.1f} "
            f"{xgi90:>6.2f} "
            f"{pen:>4}"
        )


def cmd_squad(args):
    from fpl.features import FeatureBuilder
    from fpl.model import PointsPredictor
    from fpl.optimizer import SquadOptimizer, format_squad

    fb = FeatureBuilder()
    fb.load_data()

    predictor = PointsPredictor()
    predictor.load()

    pred_features = fb.build_prediction_features()
    pred_features["predicted_points"] = predictor.predict(pred_features)

    budget = int(args.budget * 10)
    print(f"\nBuilding optimal squad (budget: {args.budget}m)...")

    optimizer = SquadOptimizer(pred_features)
    result = optimizer.select_squad(budget=budget)
    print()
    print(format_squad(result))


def cmd_transfers(args):
    from fpl.api import FPLClient
    from fpl.features import FeatureBuilder
    from fpl.model import PointsPredictor
    from fpl.transfers import TransferRecommender

    fb = FeatureBuilder()
    fb.load_data()

    predictor = PointsPredictor()
    predictor.load()

    pred_features = fb.build_prediction_features()
    pred_features["predicted_points"] = predictor.predict(pred_features)

    client = FPLClient()
    recommender = TransferRecommender(client, predictor, pred_features)

    print(f"\nAnalysing team {args.team_id}...")
    info = recommender.get_current_squad(args.team_id)
    recs = recommender.recommend_transfers(
        args.team_id, max_transfers=args.max_transfers
    )

    print()
    print(recommender.format_recommendations(args.team_id, recs, info))


def cmd_team(args):
    from fpl.api import FPLClient
    from fpl.features import FeatureBuilder
    from fpl.model import PointsPredictor
    from fpl.optimizer import SquadOptimizer
    from fpl.transfers import TransferRecommender

    fb = FeatureBuilder()
    fb.load_data()

    predictor = PointsPredictor()
    predictor.load()

    pred_features = fb.build_prediction_features()
    pred_features["predicted_points"] = predictor.predict(pred_features)

    client = FPLClient()
    recommender = TransferRecommender(client, predictor, pred_features)

    info = recommender.get_current_squad(args.team_id)
    squad = info["squad"]
    pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

    print(f"\n{info['team_name']}")
    print(f"Overall rank: {info['overall_rank']:,}")
    print(f"Total points: {info['total_points']}")
    print(f"Bank: {info['bank'] / 10:.1f}m")

    # Optimal starting 11
    optimizer = SquadOptimizer(squad)
    lineup = optimizer.select_starting_11(squad)

    cap = lineup["captain"]
    vc = lineup["vice_captain"]
    cap_id = cap.get("id") if hasattr(cap, "get") else None
    vc_id = vc.get("id") if hasattr(vc, "get") else None

    next_gw = fb._get_next_gw()
    print(f"\nRECOMMENDED STARTING XI (GW{next_gw}):")
    print(f"{'Pos':<5} {'Player':<22} {'Team':<5} {'Cost':>6} {'xPts':>6} {'Role'}")
    print("-" * 58)
    for _, p in lineup["starters"].iterrows():
        role = ""
        if p.get("id") == cap_id:
            role = "(C)"
        elif p.get("id") == vc_id:
            role = "(VC)"
        print(
            f"{pos_map.get(int(p['element_type']), '?'):<5} "
            f"{p['web_name']:<22} "
            f"{p.get('team_name', ''):<5} "
            f"{p['now_cost'] / 10:>5.1f}m "
            f"{p['predicted_points']:>5.1f} "
            f"{role}"
        )

    print(f"\nBENCH:")
    for _, p in lineup["bench"].iterrows():
        print(
            f"{pos_map.get(int(p['element_type']), '?'):<5} "
            f"{p['web_name']:<22} "
            f"{p.get('team_name', ''):<5} "
            f"{p['now_cost'] / 10:>5.1f}m "
            f"{p['predicted_points']:>5.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description="FPL Analysis Tool")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("pull", help="Pull fresh data from FPL API")
    sub.add_parser("train", help="Train prediction models")

    predict_p = sub.add_parser("predict", help="Show predicted points")
    predict_p.add_argument("--pos", choices=["GKP", "DEF", "MID", "FWD"], help="Filter by position")
    predict_p.add_argument("--top", type=int, default=20, help="Number of players to show")

    squad_p = sub.add_parser("squad", help="Build optimal squad")
    squad_p.add_argument("--budget", type=float, default=100.0, help="Budget in millions")

    transfer_p = sub.add_parser("transfers", help="Recommend transfers")
    transfer_p.add_argument("team_id", type=int, help="FPL team ID")
    transfer_p.add_argument("--max-transfers", type=int, default=2, help="Max transfers to consider")

    team_p = sub.add_parser("team", help="Show team with predictions")
    team_p.add_argument("team_id", type=int, help="FPL team ID")

    args = parser.parse_args()

    commands = {
        "pull": cmd_pull,
        "train": cmd_train,
        "predict": cmd_predict,
        "squad": cmd_squad,
        "transfers": cmd_transfers,
        "team": cmd_team,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
