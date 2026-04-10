"""Chip strategy advisor for FPL.

Chips:
- Wildcard (2 per season: 1 before ~GW20, 1 after): unlimited free transfers, permanent
- Free Hit (1 per season): unlimited transfers for 1 GW, squad reverts next GW
- Bench Boost (1 per season): bench players score points that GW
- Triple Captain (1 per season): captain scores 3x instead of 2x

Key triggers:
- Bench Boost: best in double GWs when bench all play twice
- Triple Captain: best when premium captain has DGW or elite fixture
- Free Hit: best for blank GWs when many of your players don't play
- Wildcard: best during fixture swings or when squad needs major overhaul
"""

import pandas as pd


class ChipAdvisor:
    def __init__(self, history: dict, fixtures_df: pd.DataFrame,
                 gameweeks_df: pd.DataFrame, squad_df: pd.DataFrame,
                 prediction_df: pd.DataFrame):
        self.history = history
        self.fixtures = fixtures_df
        self.gameweeks = gameweeks_df
        self.squad = squad_df
        self.prediction_df = prediction_df

    def get_chips_available(self) -> dict:
        """Determine which chips are still available.

        2025/26 rules: all chips refresh at the mid-season cutoff (~GW19/20).
        Each chip can be used once per half - so 2 uses total per season.
        We determine the half based on whether the current GW is past the cutoff.
        """
        # Mid-season cutoff: chips refresh at GW20 (first half = GW1-19, second half = GW20+)
        mid_gw = 19

        # Determine which half we're in now
        remaining = self.get_remaining_gameweeks()
        current_gw = int(remaining.iloc[0]["id"]) if len(remaining) > 0 else total_gws
        in_second_half = current_gw > mid_gw

        chip_uses = {}
        for chip in self.history.get("chips", []):
            name = chip["name"]
            gw = chip["event"]
            half = "second" if gw > mid_gw else "first"
            chip_uses.setdefault(name, []).append({"event": gw, "half": half})

        available = {}
        all_chips = ["wildcard", "3xc", "bboost", "freehit"]
        chip_labels = {
            "wildcard": "Wildcard",
            "3xc": "Triple Captain",
            "bboost": "Bench Boost",
            "freehit": "Free Hit",
        }

        current_half = "second" if in_second_half else "first"

        for chip in all_chips:
            uses = chip_uses.get(chip, [])
            used_this_half = [u for u in uses if u["half"] == current_half]
            all_used_gws = [u["event"] for u in uses]

            if used_this_half:
                # Used in current half - not available
                available[chip] = {
                    "name": chip_labels[chip],
                    "used": True,
                    "used_gw": used_this_half[-1]["event"],
                    "used_gws": all_used_gws,
                }
            else:
                available[chip] = {
                    "name": chip_labels[chip],
                    "used": False,
                    "used_gw": None,
                    "used_gws": all_used_gws,
                }

        return available

    def get_remaining_gameweeks(self) -> pd.DataFrame:
        """Get upcoming unfinished gameweeks."""
        return self.gameweeks[self.gameweeks["finished"] == False].copy()

    def detect_dgw_bgw(self) -> dict:
        """Detect double and blank gameweeks from fixtures."""
        remaining = self.get_remaining_gameweeks()
        results = {}

        for _, gw in remaining.iterrows():
            gw_id = int(gw["id"])
            gw_fixtures = self.fixtures[self.fixtures["event"] == gw_id]

            # Count fixtures per team
            team_fixtures = {}
            for _, f in gw_fixtures.iterrows():
                th = int(f["team_h"])
                ta = int(f["team_a"])
                team_fixtures[th] = team_fixtures.get(th, 0) + 1
                team_fixtures[ta] = team_fixtures.get(ta, 0) + 1

            # All 20 PL teams
            all_teams = set(range(1, 21))
            teams_with_fixtures = set(team_fixtures.keys())
            blank_teams = all_teams - teams_with_fixtures
            double_teams = {t for t, c in team_fixtures.items() if c >= 2}

            gw_type = "normal"
            if blank_teams:
                gw_type = "blank"
            if double_teams:
                gw_type = "double" if gw_type == "normal" else "double+blank"

            results[gw_id] = {
                "type": gw_type,
                "blank_teams": blank_teams,
                "double_teams": double_teams,
                "total_fixtures": len(gw_fixtures),
            }

        return results

    def analyse_bench_boost(self, gw_schedule: dict) -> dict:
        """Score Bench Boost viability for upcoming GWs."""
        recommendations = []
        bench = self.squad.nsmallest(4, "predicted_points")
        bench_total = bench["predicted_points"].sum()

        remaining = self.get_remaining_gameweeks()
        for _, gw in remaining.head(8).iterrows():
            gw_id = int(gw["id"])
            gw_info = gw_schedule.get(gw_id, {})

            score = bench_total
            reasoning = []

            # Bonus for DGW (bench players play twice)
            if gw_info.get("type") in ("double", "double+blank"):
                dgw_teams = gw_info.get("double_teams", set())
                bench_dgw = sum(1 for _, p in bench.iterrows() if int(p["team"]) in dgw_teams)
                if bench_dgw > 0:
                    score *= 1.5
                    reasoning.append(f"{bench_dgw} bench player(s) have double fixture")
                else:
                    reasoning.append("DGW but none of your bench benefit")

            # Penalty for BGW
            if gw_info.get("type") in ("blank", "double+blank"):
                bgw_teams = gw_info.get("blank_teams", set())
                bench_blank = sum(1 for _, p in bench.iterrows() if int(p["team"]) in bgw_teams)
                if bench_blank > 0:
                    score *= 0.5
                    reasoning.append(f"{bench_blank} bench player(s) have no fixture")

            # Quality check
            if bench_total < 8:
                reasoning.append("bench is weak - consider strengthening first")
            elif bench_total > 12:
                reasoning.append("strong bench makes BB more valuable")

            recommendations.append({
                "gw": gw_id,
                "score": round(score, 1),
                "reasoning": reasoning,
                "gw_type": gw_info.get("type", "normal"),
            })

        best = max(recommendations, key=lambda x: x["score"])
        return {"recommendations": recommendations, "best_gw": best}

    def analyse_triple_captain(self, gw_schedule: dict) -> dict:
        """Score Triple Captain viability for upcoming GWs."""
        recommendations = []
        # Captain is highest predicted points player
        captain = self.squad.nlargest(1, "predicted_points").iloc[0]

        remaining = self.get_remaining_gameweeks()
        for _, gw in remaining.head(8).iterrows():
            gw_id = int(gw["id"])
            gw_info = gw_schedule.get(gw_id, {})

            cap_pts = captain["predicted_points"]
            score = cap_pts  # TC adds 1x extra captain points
            reasoning = []

            if gw_info.get("type") in ("double", "double+blank"):
                dgw_teams = gw_info.get("double_teams", set())
                if int(captain["team"]) in dgw_teams:
                    score *= 2.0
                    reasoning.append(f"{captain['web_name']} has double fixture")
                else:
                    reasoning.append(f"DGW but {captain['web_name']} only plays once")

            if cap_pts > 5:
                reasoning.append(f"{captain['web_name']} has strong underlying stats")
            else:
                reasoning.append(f"{captain['web_name']} projected modestly - wait for better week")

            recommendations.append({
                "gw": gw_id,
                "score": round(score, 1),
                "captain": captain["web_name"],
                "reasoning": reasoning,
                "gw_type": gw_info.get("type", "normal"),
            })

        best = max(recommendations, key=lambda x: x["score"])
        return {"recommendations": recommendations, "best_gw": best}

    def analyse_free_hit(self, gw_schedule: dict) -> dict:
        """Score Free Hit viability for upcoming GWs."""
        recommendations = []
        squad_teams = set(int(t) for t in self.squad["team"])

        remaining = self.get_remaining_gameweeks()
        for _, gw in remaining.head(8).iterrows():
            gw_id = int(gw["id"])
            gw_info = gw_schedule.get(gw_id, {})

            score = 0
            reasoning = []

            if gw_info.get("type") in ("blank", "double+blank"):
                bgw_teams = gw_info.get("blank_teams", set())
                squad_blanks = len(squad_teams & bgw_teams)
                if squad_blanks >= 3:
                    score = squad_blanks * 3
                    reasoning.append(f"{squad_blanks} of your players have no fixture")
                elif squad_blanks > 0:
                    score = squad_blanks * 2
                    reasoning.append(f"only {squad_blanks} player(s) blanking - manageable without FH")
            else:
                reasoning.append("no blank - save Free Hit for a blank GW")

            recommendations.append({
                "gw": gw_id,
                "score": round(score, 1),
                "reasoning": reasoning,
                "gw_type": gw_info.get("type", "normal"),
            })

        best = max(recommendations, key=lambda x: x["score"])
        return {"recommendations": recommendations, "best_gw": best}

    def analyse_wildcard(self, gw_schedule: dict, free_transfers: int) -> dict:
        """Score Wildcard viability."""
        recommendations = []

        # How many transfers would it take to reach optimal squad?
        optimal_ids = set(
            self.prediction_df.nlargest(15, "predicted_points")["id"].values
        )
        current_ids = set(self.squad["id"].values)
        transfers_needed = len(current_ids - optimal_ids)

        remaining = self.get_remaining_gameweeks()
        gws_left = len(remaining)

        reasoning = []
        score = 0

        if transfers_needed >= 6:
            score = transfers_needed * 1.5
            reasoning.append(f"squad needs {transfers_needed} changes to reach optimal")
        elif transfers_needed >= 4:
            score = transfers_needed
            reasoning.append(f"squad needs {transfers_needed} changes - WC could help")
        else:
            score = transfers_needed * 0.5
            reasoning.append(f"only {transfers_needed} changes needed - probably don't need WC")

        if gws_left <= 5:
            score *= 0.5
            reasoning.append(f"only {gws_left} GWs left - limited value from WC")
        elif gws_left <= 10:
            reasoning.append(f"{gws_left} GWs left - reasonable time to benefit")

        # Look for fixture swing GWs
        for _, gw in remaining.head(6).iterrows():
            gw_id = int(gw["id"])
            gw_info = gw_schedule.get(gw_id, {})
            if gw_info.get("type") in ("double", "double+blank"):
                reasoning.append(f"GW{gw_id} is a {gw_info['type']} GW - good WC timing")
                score += 2
                break

        recommendations.append({
            "score": round(score, 1),
            "transfers_needed": transfers_needed,
            "gws_remaining": gws_left,
            "reasoning": reasoning,
        })

        return {"recommendations": recommendations}

    def full_analysis(self, free_transfers: int) -> dict:
        """Run all chip analyses and return combined results."""
        available = self.get_chips_available()
        gw_schedule = self.detect_dgw_bgw()

        analyses = {}

        if not available.get("bboost", {}).get("used"):
            analyses["bench_boost"] = self.analyse_bench_boost(gw_schedule)
        if not available.get("3xc", {}).get("used"):
            analyses["triple_captain"] = self.analyse_triple_captain(gw_schedule)
        if not available.get("freehit", {}).get("used"):
            analyses["free_hit"] = self.analyse_free_hit(gw_schedule)
        if not available.get("wildcard", {}).get("used"):
            analyses["wildcard"] = self.analyse_wildcard(gw_schedule, free_transfers)

        # Determine this-week recommendation
        next_gw = int(self.get_remaining_gameweeks().iloc[0]["id"]) if len(self.get_remaining_gameweeks()) > 0 else None
        this_week_rec = self._this_week_recommendation(analyses, gw_schedule, next_gw)

        return {
            "chips_available": available,
            "gw_schedule": gw_schedule,
            "analyses": analyses,
            "this_week": this_week_rec,
            "next_gw": next_gw,
        }

    def _this_week_recommendation(self, analyses: dict, gw_schedule: dict,
                                   next_gw: int | None) -> dict:
        """Should you play a chip THIS week?"""
        if next_gw is None:
            return {"play_chip": None, "reasoning": "Season over"}

        gw_info = gw_schedule.get(next_gw, {})
        gw_type = gw_info.get("type", "normal")

        # Check each chip's score for this GW
        best_chip = None
        best_score = 0
        best_reason = "Hold all chips - no strong trigger this week"

        for chip_key, analysis in analyses.items():
            recs = analysis.get("recommendations", [])
            this_gw_rec = next((r for r in recs if r.get("gw") == next_gw), None)

            if this_gw_rec:
                # Is this the best GW for this chip?
                best_gw = analysis.get("best_gw", {})
                if best_gw.get("gw") == next_gw and this_gw_rec["score"] > best_score:
                    best_score = this_gw_rec["score"]
                    best_chip = chip_key
                    best_reason = "; ".join(this_gw_rec.get("reasoning", []))

        chip_labels = {
            "bench_boost": "Bench Boost",
            "triple_captain": "Triple Captain",
            "free_hit": "Free Hit",
            "wildcard": "Wildcard",
        }

        if best_chip and best_score > 5:
            return {
                "play_chip": chip_labels.get(best_chip, best_chip),
                "score": best_score,
                "reasoning": best_reason,
            }

        return {
            "play_chip": None,
            "reasoning": best_reason,
        }
