"""Transfer recommendation engine."""

from itertools import combinations

import pandas as pd

from .api import FPLClient
from .model import PointsPredictor
from .optimizer import SquadOptimizer


class TransferRecommender:
    def __init__(self, client: FPLClient, predictor: PointsPredictor,
                 prediction_df: pd.DataFrame):
        self.client = client
        self.predictor = predictor
        self.prediction_df = prediction_df

    def get_current_squad(self, team_id: int) -> dict:
        """Fetch current squad info."""
        entry = self.client.get_entry(team_id)
        history = self.client.get_entry_history(team_id)

        # Find current GW
        current_events = [e for e in history.get("current", []) if e.get("points", 0) > 0]
        if not current_events:
            raise ValueError("No gameweek data found for this team")

        latest_gw = max(e["event"] for e in current_events)
        latest_event = [e for e in current_events if e["event"] == latest_gw][0]

        picks_data = self.client.get_picks(team_id, latest_gw)
        picks = picks_data["picks"]

        # Get player details
        squad_ids = [p["element"] for p in picks]
        squad_df = self.prediction_df[self.prediction_df["id"].isin(squad_ids)].copy()

        # Add pick info
        pick_info = {p["element"]: p for p in picks}
        squad_df["is_captain"] = squad_df["id"].map(lambda x: pick_info.get(x, {}).get("is_captain", False))
        squad_df["multiplier"] = squad_df["id"].map(lambda x: pick_info.get(x, {}).get("multiplier", 1))
        squad_df["selling_price"] = squad_df["id"].map(
            lambda x: pick_info.get(x, {}).get("selling_price", 0)
        )
        # If selling_price not available, use now_cost
        if squad_df["selling_price"].sum() == 0:
            squad_df["selling_price"] = squad_df["now_cost"]

        bank = latest_event.get("bank", 0)
        free_transfers = self._calculate_free_transfers(history)

        return {
            "squad": squad_df,
            "bank": bank,
            "free_transfers": free_transfers,
            "team_name": entry.get("name", "Unknown"),
            "overall_rank": entry.get("summary_overall_rank", 0),
            "total_points": entry.get("summary_overall_points", 0),
            "gameweek": latest_gw,
        }

    def recommend_transfers(self, team_id: int, max_transfers: int = 2,
                            lookahead_gws: int = 5) -> list[dict]:
        """Generate transfer recommendations."""
        info = self.get_current_squad(team_id)
        squad = info["squad"]
        bank = info["bank"]
        free_transfers = info["free_transfers"]

        all_players = self.prediction_df.copy()

        recommendations = []

        for n_transfers in range(1, max_transfers + 1):
            hits = max(0, n_transfers - free_transfers)
            hit_cost = hits * 4

            # Sell candidates: bottom performers by predicted points
            sell_pool_size = min(len(squad), n_transfers * 3)
            sell_candidates = squad.nsmallest(sell_pool_size, "predicted_points")

            best_rec = None
            best_gain = -999

            for sell_combo in combinations(sell_candidates.iterrows(), n_transfers):
                sell_rows = [row for _, row in sell_combo]
                sell_ids = [int(r["id"]) for r in sell_rows]
                sell_value = sum(r["selling_price"] for r in sell_rows)
                sell_points = sum(r["predicted_points"] for r in sell_rows)

                available_budget = bank + sell_value

                # Remaining squad after selling
                remaining_ids = set(int(x) for x in squad["id"]) - set(sell_ids)
                remaining = squad[squad["id"].isin(remaining_ids)]

                # Team counts in remaining squad
                team_counts = remaining["team"].value_counts().to_dict()

                # Find replacements for each sold player
                buy_candidates = []
                for sell_row in sell_rows:
                    pos = int(sell_row["element_type"])

                    # Available players: right position, not in squad, affordable
                    candidates = all_players[
                        (all_players["element_type"] == pos)
                        & (~all_players["id"].isin(remaining_ids))
                        & (~all_players["id"].isin([b["id"] for b in buy_candidates] if buy_candidates else []))
                        & (all_players["predicted_points"] > 0)
                    ].copy()

                    # Filter by team constraint
                    candidates = candidates[
                        candidates["team"].map(lambda t: team_counts.get(t, 0) < 3)
                    ]

                    if candidates.empty:
                        break

                    buy_candidates.append(candidates)

                if len(buy_candidates) != n_transfers:
                    continue

                # Greedy allocation: pick best affordable combo
                bought = []
                budget_left = available_budget
                for i, cands in enumerate(buy_candidates):
                    affordable = cands[cands["now_cost"] <= budget_left]
                    if affordable.empty:
                        break
                    best = affordable.nlargest(1, "predicted_points").iloc[0]
                    bought.append(best)
                    budget_left -= int(best["now_cost"])
                    # Update team counts
                    t = int(best["team"])
                    team_counts[t] = team_counts.get(t, 0) + 1

                if len(bought) != n_transfers:
                    continue

                buy_points = sum(b["predicted_points"] for b in bought)
                buy_cost = sum(int(b["now_cost"]) for b in bought)
                net_gain = buy_points - sell_points - hit_cost

                if net_gain > best_gain:
                    best_gain = net_gain
                    # Build reasoning for each swap
                    reasons = []
                    for sell_row, buy_row in zip(sell_rows, bought):
                        reasons.append(self._build_reason(sell_row, buy_row))

                    best_rec = {
                        "n_transfers": n_transfers,
                        "hits": hits,
                        "hit_cost": hit_cost,
                        "transfers_out": sell_rows,
                        "transfers_in": bought,
                        "points_gain": round(net_gain, 2),
                        "cost_change": buy_cost - int(sell_value),
                        "bank_after": bank + int(sell_value) - buy_cost,
                        "reasons": reasons,
                    }

            if best_rec:
                recommendations.append(best_rec)

        return recommendations

    def _calculate_free_transfers(self, history: dict) -> int:
        """Calculate free transfers available for next GW from transfer history.

        Rules: start with 1, gain 1 per GW (max 5), spend on transfers.
        If you use a wildcard or free hit, you get 1 back next GW.
        """
        events = sorted(history.get("current", []), key=lambda e: e["event"])
        chips = {c["event"]: c["name"] for c in history.get("chips", [])}

        free = 1
        for event in events:
            gw = event["event"]
            transfers_made = event.get("event_transfers", 0)

            # Wildcard/free hit resets to 1 next GW
            if gw in chips and chips[gw] in ("wildcard", "freehit"):
                free = 1
                continue

            # Spend free transfers (any beyond free cost -4 each)
            free_used = min(transfers_made, free)
            free = free - free_used

            # Gain 1 for next GW, cap at 5
            free = min(free + 1, 5)

        return free

    def _build_reason(self, sell: pd.Series, buy: pd.Series) -> str:
        """Build a human-readable reason for why buy > sell."""
        parts = []

        # Points uplift
        diff = buy["predicted_points"] - sell["predicted_points"]
        parts.append(f"+{diff:.1f} xPts uplift")

        # xGI comparison
        sell_xgi = sell.get("xgi_per90", 0)
        buy_xgi = buy.get("xgi_per90", 0)
        if buy_xgi > sell_xgi + 0.05:
            parts.append(f"higher xGI/90 ({buy_xgi:.2f} vs {sell_xgi:.2f})")

        # Form comparison
        sell_form = float(sell.get("form", sell.get("rolling_3_points", 0)))
        buy_form = float(buy.get("form", buy.get("rolling_3_points", 0)))
        if buy_form > sell_form + 0.5:
            parts.append(f"better form ({buy_form:.1f} vs {sell_form:.1f})")
        elif sell_form > buy_form + 0.5:
            parts.append(f"lower form but model expects reversion")

        # Fixture difficulty
        sell_diff = sell.get("opponent_difficulty", 3)
        buy_diff = buy.get("opponent_difficulty", 3)
        if buy_diff < sell_diff:
            parts.append(f"easier fixture (difficulty {buy_diff} vs {sell_diff})")
        elif buy_diff > sell_diff:
            parts.append(f"harder fixture but outweighed by quality")

        # Penalty taker
        if buy.get("is_penalty_taker", 0) and not sell.get("is_penalty_taker", 0):
            parts.append("on penalties")

        # Cost saving
        cost_diff = sell.get("selling_price", sell["now_cost"]) - buy["now_cost"]
        if cost_diff > 5:  # 0.5m+
            parts.append(f"saves {cost_diff / 10:.1f}m")

        # Minutes risk
        sell_mins = sell.get("avg_minutes_5", sell.get("rolling_5_minutes", 0))
        buy_mins = buy.get("avg_minutes_5", buy.get("rolling_5_minutes", 0))
        if sell_mins < 60 and buy_mins > 70:
            parts.append(f"more nailed ({buy_mins:.0f} vs {sell_mins:.0f} avg mins)")

        return "; ".join(parts)

    def format_recommendations(self, team_id: int, recs: list[dict], info: dict) -> str:
        """Pretty-print transfer recommendations."""
        pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
        lines = []

        lines.append(f"Team: {info['team_name']}")
        lines.append(f"Overall rank: {info['overall_rank']:,}")
        lines.append(f"Total points: {info['total_points']}")
        lines.append(f"Bank: {info['bank'] / 10:.1f}m")
        lines.append(f"Free transfers: {info['free_transfers']}")
        lines.append("")

        # Current squad
        lines.append("CURRENT SQUAD:")
        lines.append(f"{'Pos':<5} {'Player':<22} {'Team':<5} {'Cost':>6} {'xPts':>6}")
        lines.append("-" * 50)
        squad_sorted = info["squad"].sort_values(["element_type", "predicted_points"], ascending=[True, False])
        for _, p in squad_sorted.iterrows():
            lines.append(
                f"{pos_map.get(int(p['element_type']), '?'):<5} "
                f"{p['web_name']:<22} "
                f"{p.get('team_name', ''):<5} "
                f"{p['now_cost'] / 10:>5.1f}m "
                f"{p['predicted_points']:>5.1f}"
            )

        lines.append("")
        lines.append("=" * 60)
        lines.append("TRANSFER RECOMMENDATIONS:")
        lines.append("=" * 60)

        if not recs:
            lines.append("No beneficial transfers found.")
            return "\n".join(lines)

        free = info["free_transfers"]

        for rec in recs:
            lines.append("")
            n = rec["n_transfers"]
            hits = rec["hits"]

            if hits > 0:
                lines.append(f"Option: {n} transfer(s) - COSTS {hits} hit(s) = -{rec['hit_cost']} pts")
            else:
                lines.append(f"Option: {n} transfer(s) - FREE ({free - n} free transfer(s) left after)")

            # Show the raw gain before hit, then net
            raw_gain = rec["points_gain"] + rec["hit_cost"]
            lines.append(f"  Raw xPts uplift: +{raw_gain:.1f}")
            if hits > 0:
                lines.append(f"  Hit cost:        -{rec['hit_cost']}")
                lines.append(f"  Net gain:        {rec['points_gain']:+.1f} pts")
                if rec["points_gain"] > 0:
                    lines.append(f"  Verdict:         Worth the hit (+{rec['points_gain']:.1f} net)")
                else:
                    lines.append(f"  Verdict:         NOT worth the hit (save the transfer)")
            else:
                lines.append(f"  Net gain:        +{rec['points_gain']:.1f} pts (no hit)")

            lines.append(f"  Bank after:      {rec['bank_after'] / 10:.1f}m")
            lines.append("")

            lines.append("  OUT:")
            for p in rec["transfers_out"]:
                lines.append(
                    f"    {pos_map.get(int(p['element_type']), '?')} "
                    f"{p['web_name']:<20} "
                    f"({p['selling_price'] / 10:.1f}m, {p['predicted_points']:.1f} xPts)"
                )

            lines.append("  IN:")
            for p in rec["transfers_in"]:
                lines.append(
                    f"    {pos_map.get(int(p['element_type']), '?')} "
                    f"{p['web_name']:<20} "
                    f"({p['now_cost'] / 10:.1f}m, {p['predicted_points']:.1f} xPts)"
                )

            if rec.get("reasons"):
                lines.append("")
                lines.append("  Why:")
                for i, reason in enumerate(rec["reasons"]):
                    out_name = rec["transfers_out"][i]["web_name"]
                    in_name = rec["transfers_in"][i]["web_name"]
                    lines.append(f"    {out_name} -> {in_name}: {reason}")

        return "\n".join(lines)
