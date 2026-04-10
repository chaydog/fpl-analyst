"""Squad selection and starting 11 optimization using integer programming."""

import pandas as pd
import pulp


class SquadOptimizer:
    def __init__(self, players_df: pd.DataFrame):
        """players_df must have: id, web_name, element_type, team, now_cost, predicted_points"""
        self.players = players_df.copy()
        # Ensure index is clean
        self.players = self.players.reset_index(drop=True)

    def select_squad(self, budget: int = 1000, existing_players: list[int] | None = None,
                     excluded_players: list[int] | None = None) -> dict:
        """Select optimal 15-man squad.

        Args:
            budget: total budget in 0.1m units (1000 = 100.0m)
            existing_players: player IDs that must be in the squad
            excluded_players: player IDs that cannot be in the squad
        """
        prob = pulp.LpProblem("FPL_Squad", pulp.LpMaximize)

        # Filter to available players
        df = self.players[self.players["predicted_points"] > 0].copy()
        if excluded_players:
            df = df[~df["id"].isin(excluded_players)]
        df = df.reset_index(drop=True)

        # Decision variables
        x = pulp.LpVariable.dicts("p", df.index, cat="Binary")

        # Objective: maximize predicted points
        prob += pulp.lpSum(x[i] * df.loc[i, "predicted_points"] for i in df.index)

        # Budget
        prob += pulp.lpSum(x[i] * df.loc[i, "now_cost"] for i in df.index) <= budget

        # Position constraints: 2 GKP, 5 DEF, 5 MID, 3 FWD
        for pos_id, count in {1: 2, 2: 5, 3: 5, 4: 3}.items():
            pos_idx = df[df["element_type"] == pos_id].index
            prob += pulp.lpSum(x[i] for i in pos_idx) == count

        # Max 3 per team
        for team_id in df["team"].unique():
            team_idx = df[df["team"] == team_id].index
            prob += pulp.lpSum(x[i] for i in team_idx) <= 3

        # Force existing players in
        if existing_players:
            for pid in existing_players:
                idx = df[df["id"] == pid].index
                if len(idx) > 0:
                    prob += x[idx[0]] == 1

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if prob.status != 1:
            return {"status": "infeasible", "squad": [], "cost": 0, "predicted_points": 0}

        # Extract squad
        selected = [i for i in df.index if x[i].varValue == 1]
        squad_df = df.loc[selected].copy()
        total_cost = squad_df["now_cost"].sum()
        total_pp = squad_df["predicted_points"].sum()

        return {
            "status": "optimal",
            "squad": squad_df,
            "cost": total_cost,
            "budget_remaining": budget - total_cost,
            "predicted_points": total_pp,
        }

    def select_starting_11(self, squad_df: pd.DataFrame) -> dict:
        """Pick best starting 11 from 15-man squad."""
        prob = pulp.LpProblem("FPL_Starting11", pulp.LpMaximize)
        df = squad_df.reset_index(drop=True)

        x = pulp.LpVariable.dicts("s", df.index, cat="Binary")

        # Maximize points of starters
        prob += pulp.lpSum(x[i] * df.loc[i, "predicted_points"] for i in df.index)

        # Exactly 11 starters
        prob += pulp.lpSum(x[i] for i in df.index) == 11

        # Exactly 1 GKP
        gkp_idx = df[df["element_type"] == 1].index
        prob += pulp.lpSum(x[i] for i in gkp_idx) == 1

        # At least 3 DEF
        def_idx = df[df["element_type"] == 2].index
        prob += pulp.lpSum(x[i] for i in def_idx) >= 3

        # At least 1 FWD
        fwd_idx = df[df["element_type"] == 4].index
        prob += pulp.lpSum(x[i] for i in fwd_idx) >= 1

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        starters_idx = [i for i in df.index if x[i].varValue == 1]
        bench_idx = [i for i in df.index if x[i].varValue == 0]

        starters = df.loc[starters_idx].sort_values("element_type")
        # Bench order: GK always first, then outfield by predicted points
        bench_df = df.loc[bench_idx]
        bench_gk = bench_df[bench_df["element_type"] == 1]
        bench_outfield = bench_df[bench_df["element_type"] != 1].sort_values("predicted_points", ascending=False)
        bench = pd.concat([bench_gk, bench_outfield])

        # Captain and vice captain
        captain = starters.loc[starters["predicted_points"].idxmax()]
        vice_candidates = starters[starters.index != captain.name]
        vice_captain = vice_candidates.loc[vice_candidates["predicted_points"].idxmax()]

        return {
            "starters": starters,
            "bench": bench,
            "captain": captain,
            "vice_captain": vice_captain,
        }


def format_squad(result: dict) -> str:
    """Pretty-print a squad selection result."""
    if result["status"] != "optimal":
        return "Could not find a valid squad with these constraints."

    squad = result["squad"]
    lineup = SquadOptimizer(squad).select_starting_11(squad)

    lines = []
    lines.append(f"Budget: {result['cost'] / 10:.1f}m / {(result['cost'] + result['budget_remaining']) / 10:.1f}m "
                 f"(remaining: {result['budget_remaining'] / 10:.1f}m)")
    lines.append(f"Total predicted points: {result['predicted_points']:.1f}")
    lines.append("")

    cap_id = lineup["captain"]["id"] if "id" in lineup["captain"].index else None
    vc_id = lineup["vice_captain"]["id"] if "id" in lineup["vice_captain"].index else None

    pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

    lines.append("STARTING XI:")
    lines.append(f"{'Pos':<5} {'Player':<22} {'Team':<5} {'Cost':>6} {'xPts':>6} {'Role'}")
    lines.append("-" * 60)
    for _, p in lineup["starters"].iterrows():
        role = ""
        if p.get("id") == cap_id:
            role = "(C)"
        elif p.get("id") == vc_id:
            role = "(VC)"
        lines.append(
            f"{pos_map.get(p['element_type'], '?'):<5} "
            f"{p['web_name']:<22} "
            f"{p.get('team_name', ''):<5} "
            f"{p['now_cost'] / 10:>5.1f}m "
            f"{p['predicted_points']:>5.1f} "
            f"{role}"
        )

    lines.append("")
    lines.append("BENCH:")
    for _, p in lineup["bench"].iterrows():
        lines.append(
            f"{pos_map.get(p['element_type'], '?'):<5} "
            f"{p['web_name']:<22} "
            f"{p.get('team_name', ''):<5} "
            f"{p['now_cost'] / 10:>5.1f}m "
            f"{p['predicted_points']:>5.1f}"
        )

    return "\n".join(lines)
