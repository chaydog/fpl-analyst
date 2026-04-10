"""Data ingestion - pulls FPL API data and builds DataFrames."""

import sys
from pathlib import Path

import pandas as pd

from .api import FPLClient


class DataIngestor:
    def __init__(self, client: FPLClient | None = None):
        self.client = client or FPLClient()
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def pull_all(self, verbose: bool = True) -> None:
        bootstrap = self.client.get_bootstrap()

        players_df = self._build_players_df(bootstrap)
        teams_df = self._build_teams_df(bootstrap)
        gameweeks_df = self._build_gameweeks_df(bootstrap)
        fixtures_df = self._build_fixtures_df()

        players_df.to_parquet(self.processed_dir / "players.parquet")
        teams_df.to_parquet(self.processed_dir / "teams.parquet")
        gameweeks_df.to_parquet(self.processed_dir / "gameweeks.parquet")
        fixtures_df.to_parquet(self.processed_dir / "fixtures.parquet")

        if verbose:
            print(f"Players: {len(players_df)}")
            print(f"Teams: {len(teams_df)}")

        # Fetch per-GW history for players with minutes
        active = players_df[players_df["minutes"] > 0]
        if verbose:
            print(f"Fetching history for {len(active)} active players...")

        history_rows = []
        for i, (_, player) in enumerate(active.iterrows()):
            pid = int(player["id"])
            try:
                summary = self.client.get_player_summary(pid)
                for row in summary.get("history", []):
                    row["player_id"] = pid
                    row["element_type"] = int(player["element_type"])
                    row["team"] = int(player["team"])
                    history_rows.append(row)
            except Exception as e:
                if verbose:
                    print(f"  Failed for player {pid}: {e}")

            if verbose and (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(active)} fetched")
                sys.stdout.flush()

        history_df = pd.DataFrame(history_rows)
        if not history_df.empty:
            # Convert expected stats to float
            for col in [
                "expected_goals", "expected_assists",
                "expected_goal_involvements", "expected_goals_conceded",
            ]:
                if col in history_df.columns:
                    history_df[col] = pd.to_numeric(history_df[col], errors="coerce")

            history_df.to_parquet(self.processed_dir / "history.parquet")
            if verbose:
                print(f"History rows: {len(history_df)}")

        if verbose:
            print("Data pull complete.")

    def _build_players_df(self, bootstrap: dict) -> pd.DataFrame:
        df = pd.DataFrame(bootstrap["elements"])
        # Join team name
        teams = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
        df["team_name"] = df["team"].map(teams)
        pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
        df["position"] = df["element_type"].map(pos_map)
        return df

    def _build_teams_df(self, bootstrap: dict) -> pd.DataFrame:
        return pd.DataFrame(bootstrap["teams"])

    def _build_gameweeks_df(self, bootstrap: dict) -> pd.DataFrame:
        df = pd.DataFrame(bootstrap["events"])
        # Drop complex nested columns that can't be serialized to parquet
        for col in df.columns:
            if df[col].dtype == "object":
                sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                if isinstance(sample, (dict, list)):
                    df = df.drop(columns=[col])
        return df

    def _build_fixtures_df(self) -> pd.DataFrame:
        fixtures = self.client.get_fixtures()
        return pd.DataFrame(fixtures)
