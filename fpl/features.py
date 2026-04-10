"""Feature engineering pipeline for FPL prediction."""

import pandas as pd
import numpy as np
from pathlib import Path


class FeatureBuilder:
    def __init__(self):
        self.processed_dir = Path("data/processed")

    def load_data(self):
        self.history = pd.read_parquet(self.processed_dir / "history.parquet")
        self.players = pd.read_parquet(self.processed_dir / "players.parquet")
        self.fixtures = pd.read_parquet(self.processed_dir / "fixtures.parquet")
        self.teams = pd.read_parquet(self.processed_dir / "teams.parquet")
        self.gameweeks = pd.read_parquet(self.processed_dir / "gameweeks.parquet")

    def build_training_features(self) -> pd.DataFrame:
        """Build features for every (player, GW) in history. No leakage."""
        df = self.history.copy()
        df = df.sort_values(["player_id", "round"]).reset_index(drop=True)

        # Rolling features (shifted to avoid leakage)
        df = self._add_rolling_features(df)

        # Per-90 stats from rolling windows
        df = self._add_per90_features(df)

        # Form / trend
        df = self._add_trend_features(df)

        # Fixture context
        df = self._add_fixture_features(df)

        # Set piece duties from current player data
        df = self._add_set_piece_features(df)

        # Minutes availability
        df = self._add_availability_features(df)

        # Target
        df["target"] = df["total_points"]

        # Drop rows where we don't have enough history for features
        df = df.dropna(subset=["rolling_3_xgi"]).reset_index(drop=True)

        return df

    def build_prediction_features(self, next_gw: int | None = None) -> pd.DataFrame:
        """Build features for predicting the next gameweek."""
        if next_gw is None:
            next_gw = self._get_next_gw()

        # Get latest rolling stats per player from history
        df = self.history.copy()
        df = df.sort_values(["player_id", "round"]).reset_index(drop=True)
        df = self._add_rolling_features(df, shift=False)
        df = self._add_per90_features(df)
        df = self._add_trend_features(df, shift=False)

        # Take last row per player (most recent stats)
        latest = df.groupby("player_id").tail(1).copy()

        # Merge player info
        player_cols = [
            "id", "web_name", "element_type", "team", "team_name",
            "now_cost", "chance_of_playing_next_round",
            "penalties_order", "direct_freekicks_order",
            "corners_and_indirect_freekicks_order",
            "status", "selected_by_percent", "form",
        ]
        available_cols = [c for c in player_cols if c in self.players.columns]
        player_info = self.players[available_cols].copy()
        latest = latest.merge(
            player_info, left_on="player_id", right_on="id", how="left",
            suffixes=("", "_current"),
        )

        # Add fixture features for next GW
        latest["next_gw"] = next_gw
        latest = self._add_prediction_fixture_features(latest, next_gw)

        # Set piece features
        latest = self._add_set_piece_features(latest)

        # Availability features
        latest = self._add_availability_features(latest)

        if "chance_of_playing_next_round" in latest.columns:
            latest["chance_of_playing"] = pd.to_numeric(
                latest["chance_of_playing_next_round"], errors="coerce"
            ).fillna(100) / 100
        else:
            latest["chance_of_playing"] = 1.0

        return latest

    def build_prediction_features_multi_gw(self, n_gws: int = 5) -> dict:
        """Build prediction features for the next N gameweeks.

        Returns dict: {gw_number: features_df, ...}
        Player stats (rolling, form, per90) are shared. Only fixture
        context (opponent, home/away, difficulty, strength) changes per GW.
        """
        start_gw = self._get_next_gw()
        gw_ids = list(range(start_gw, start_gw + n_gws))

        # Build base stats once (expensive part)
        df = self.history.copy()
        df = df.sort_values(["player_id", "round"]).reset_index(drop=True)
        df = self._add_rolling_features(df, shift=False)
        df = self._add_per90_features(df)
        df = self._add_trend_features(df, shift=False)

        latest = df.groupby("player_id").tail(1).copy()

        player_cols = [
            "id", "web_name", "element_type", "team", "team_name",
            "now_cost", "chance_of_playing_next_round",
            "penalties_order", "direct_freekicks_order",
            "corners_and_indirect_freekicks_order",
            "status", "selected_by_percent", "form",
        ]
        available_cols = [c for c in player_cols if c in self.players.columns]
        player_info = self.players[available_cols].copy()
        base = latest.merge(
            player_info, left_on="player_id", right_on="id", how="left",
            suffixes=("", "_current"),
        )
        base = self._add_set_piece_features(base)
        base = self._add_availability_features(base)

        if "chance_of_playing_next_round" in base.columns:
            base["chance_of_playing"] = pd.to_numeric(
                base["chance_of_playing_next_round"], errors="coerce"
            ).fillna(100) / 100
        else:
            base["chance_of_playing"] = 1.0

        # Drop fixture columns that will be re-added per GW
        fixture_cols_to_drop = [
            "is_home", "opponent_difficulty", "team_attack_strength",
            "opponent_defence_strength", "relative_strength",
            "has_fixture", "opponent_team_next", "next_gw",
        ]
        base_clean = base.drop(
            columns=[c for c in fixture_cols_to_drop if c in base.columns],
            errors="ignore",
        )

        result = {}
        for gw in gw_ids:
            gw_features = base_clean.copy()
            gw_features["next_gw"] = gw
            gw_features = self._add_prediction_fixture_features(gw_features, gw)
            result[gw] = gw_features

        return result

    def get_feature_columns(self) -> list[str]:
        return [
            # Rolling means
            "rolling_3_points", "rolling_5_points", "rolling_10_points",
            "rolling_3_xg", "rolling_5_xg",
            "rolling_3_xa", "rolling_5_xa",
            "rolling_3_xgi", "rolling_5_xgi", "rolling_10_xgi",
            "rolling_3_bps", "rolling_5_bps",
            "rolling_3_ict", "rolling_5_ict",
            "rolling_3_influence", "rolling_5_influence",
            "rolling_3_threat", "rolling_5_threat",
            "rolling_3_creativity", "rolling_5_creativity",
            "rolling_3_minutes", "rolling_5_minutes",
            "rolling_3_bonus", "rolling_5_bonus",
            "rolling_3_cs", "rolling_5_cs",
            # Per 90
            "xg_per90", "xa_per90", "xgi_per90", "bps_per90",
            # Trends
            "points_trend", "xgi_trend",
            # Fixture
            "is_home", "opponent_difficulty",
            "team_attack_strength", "opponent_defence_strength",
            "relative_strength",
            # Set pieces
            "is_penalty_taker", "is_set_piece_taker",
            # Availability
            "avg_minutes_5", "start_rate_5",
        ]

    def _add_rolling_features(self, df: pd.DataFrame, shift: bool = True) -> pd.DataFrame:
        grouped = df.groupby("player_id")

        stat_map = {
            "points": "total_points",
            "xg": "expected_goals",
            "xa": "expected_assists",
            "xgi": "expected_goal_involvements",
            "bps": "bps",
            "ict": "ict_index",
            "influence": "influence",
            "threat": "threat",
            "creativity": "creativity",
            "minutes": "minutes",
            "bonus": "bonus",
            "cs": "clean_sheets",
        }

        for label, col in stat_map.items():
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            if shift:
                series = grouped[col].transform(lambda x: pd.to_numeric(x, errors="coerce").shift(1))

            for window in [3, 5, 10]:
                col_name = f"rolling_{window}_{label}"
                df[col_name] = series.groupby(df["player_id"]).transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )

        return df

    def _add_per90_features(self, df: pd.DataFrame) -> pd.DataFrame:
        mins = df.get("rolling_5_minutes", pd.Series(dtype=float))
        safe_mins = mins.clip(lower=30)  # avoid div by zero for low-minute players

        for stat, col in [("xg", "rolling_5_xg"), ("xa", "rolling_5_xa"),
                          ("xgi", "rolling_5_xgi"), ("bps", "rolling_5_bps")]:
            if col in df.columns:
                df[f"{stat}_per90"] = df[col] / safe_mins * 90
            else:
                df[f"{stat}_per90"] = 0.0

        return df

    def _add_trend_features(self, df: pd.DataFrame, shift: bool = True) -> pd.DataFrame:
        for label, col in [("points_trend", "total_points"), ("xgi_trend", "expected_goal_involvements")]:
            if col not in df.columns:
                df[label] = 0.0
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            if shift:
                series = df.groupby("player_id")[col].transform(
                    lambda x: pd.to_numeric(x, errors="coerce").shift(1)
                )

            def slope(x):
                x = x.dropna()
                if len(x) < 3:
                    return 0.0
                return np.polyfit(range(len(x)), x.values, 1)[0]

            df[label] = series.groupby(df["player_id"]).transform(
                lambda x: x.rolling(5, min_periods=3).apply(slope, raw=False)
            )

        return df

    def _add_fixture_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["is_home"] = df["was_home"].astype(int) if "was_home" in df.columns else 0

        # Build team strength lookup
        team_strength = {}
        for _, t in self.teams.iterrows():
            team_strength[t["id"]] = {
                "attack_home": t.get("strength_attack_home", 1000),
                "attack_away": t.get("strength_attack_away", 1000),
                "defence_home": t.get("strength_defence_home", 1000),
                "defence_away": t.get("strength_defence_away", 1000),
            }

        # Opponent difficulty from fixture data
        if "opponent_team" in df.columns:
            # Map opponent difficulty
            fixture_diff = {}
            for _, f in self.fixtures.iterrows():
                gw = f.get("event")
                if pd.isna(gw):
                    continue
                fixture_diff[(int(f["team_h"]), int(gw))] = int(f.get("team_h_difficulty", 3))
                fixture_diff[(int(f["team_a"]), int(gw))] = int(f.get("team_a_difficulty", 3))

            df["opponent_difficulty"] = df.apply(
                lambda r: fixture_diff.get((int(r.get("opponent_team", 0)), int(r.get("round", 0))), 3),
                axis=1,
            )

            # Team attack vs opponent defence strength
            def get_strength(row):
                tid = int(row.get("team", 0))
                oid = int(row.get("opponent_team", 0))
                home = bool(row.get("was_home", False))

                ts = team_strength.get(tid, {})
                os_ = team_strength.get(oid, {})

                if home:
                    ta = ts.get("attack_home", 1000)
                    od = os_.get("defence_away", 1000)
                else:
                    ta = ts.get("attack_away", 1000)
                    od = os_.get("defence_home", 1000)

                return pd.Series({"team_attack_strength": ta, "opponent_defence_strength": od})

            strength = df.apply(get_strength, axis=1)
            df["team_attack_strength"] = strength["team_attack_strength"]
            df["opponent_defence_strength"] = strength["opponent_defence_strength"]
            df["relative_strength"] = df["team_attack_strength"] - df["opponent_defence_strength"]
        else:
            df["opponent_difficulty"] = 3
            df["team_attack_strength"] = 1000
            df["opponent_defence_strength"] = 1000
            df["relative_strength"] = 0

        return df

    def _add_prediction_fixture_features(self, df: pd.DataFrame, next_gw: int) -> pd.DataFrame:
        # Find fixtures for next GW
        gw_fixtures = self.fixtures[self.fixtures["event"] == next_gw]

        team_fixture = {}
        for _, f in gw_fixtures.iterrows():
            th = int(f["team_h"])
            ta = int(f["team_a"])
            team_fixture[th] = {"opponent": ta, "is_home": 1, "difficulty": int(f.get("team_h_difficulty", 3))}
            team_fixture[ta] = {"opponent": th, "is_home": 0, "difficulty": int(f.get("team_a_difficulty", 3))}

        df["is_home"] = df["team"].map(lambda t: team_fixture.get(int(t), {}).get("is_home", 0))
        df["opponent_difficulty"] = df["team"].map(lambda t: team_fixture.get(int(t), {}).get("difficulty", 3))
        df["opponent_team_next"] = df["team"].map(lambda t: team_fixture.get(int(t), {}).get("opponent", 0))

        # Team strength
        team_strength = {}
        for _, t in self.teams.iterrows():
            team_strength[t["id"]] = {
                "attack_home": t.get("strength_attack_home", 1000),
                "attack_away": t.get("strength_attack_away", 1000),
                "defence_home": t.get("strength_defence_home", 1000),
                "defence_away": t.get("strength_defence_away", 1000),
            }

        def compute_strength(row):
            tid = int(row.get("team", 0))
            oid = int(row.get("opponent_team_next", 0))
            home = bool(row.get("is_home", 0))
            ts = team_strength.get(tid, {})
            os_ = team_strength.get(oid, {})
            if home:
                return ts.get("attack_home", 1000), os_.get("defence_away", 1000)
            return ts.get("attack_away", 1000), os_.get("defence_home", 1000)

        strengths = df.apply(compute_strength, axis=1, result_type="expand")
        df["team_attack_strength"] = strengths[0]
        df["opponent_defence_strength"] = strengths[1]
        df["relative_strength"] = df["team_attack_strength"] - df["opponent_defence_strength"]

        # Has fixture flag (blank GW check)
        df["has_fixture"] = df["team"].map(lambda t: int(t) in team_fixture).astype(int)

        return df

    def _add_set_piece_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "player_id" in df.columns:
            pid_col = "player_id"
        else:
            pid_col = "id"

        sp_data = self.players[["id", "penalties_order", "direct_freekicks_order",
                                 "corners_and_indirect_freekicks_order"]].copy()
        for col in ["penalties_order", "direct_freekicks_order", "corners_and_indirect_freekicks_order"]:
            sp_data[col] = pd.to_numeric(sp_data[col], errors="coerce")

        if "is_penalty_taker" not in df.columns:
            sp_map = sp_data.set_index("id")
            df["is_penalty_taker"] = df[pid_col].map(
                lambda x: int(sp_map.loc[x, "penalties_order"] == 1) if x in sp_map.index and pd.notna(sp_map.loc[x, "penalties_order"]) else 0
            )
            df["is_set_piece_taker"] = df[pid_col].map(
                lambda x: int(
                    (sp_map.loc[x, "direct_freekicks_order"] == 1 if pd.notna(sp_map.loc[x, "direct_freekicks_order"]) else False)
                    or (sp_map.loc[x, "corners_and_indirect_freekicks_order"] == 1 if pd.notna(sp_map.loc[x, "corners_and_indirect_freekicks_order"]) else False)
                ) if x in sp_map.index else 0
            )

        return df

    def _add_availability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        grouped = df.groupby("player_id")
        df["avg_minutes_5"] = grouped["minutes"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
        df["start_rate_5"] = grouped["minutes"].transform(
            lambda x: (x.shift(1) > 0).rolling(5, min_periods=1).mean()
        )
        return df

    def _get_next_gw(self) -> int:
        gws = self.gameweeks
        upcoming = gws[gws["finished"] == False]
        if len(upcoming) > 0:
            return int(upcoming.iloc[0]["id"])
        return int(gws.iloc[-1]["id"])
