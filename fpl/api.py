"""FPL API client with disk caching and rate limiting."""

import json
import time
from pathlib import Path

import requests

BASE_URL = "https://fantasy.premierleague.com/api"


class FPLClient:
    def __init__(self, cache_dir: Path = Path("data/raw")):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "FPL-Model/0.1"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)
        self._last_request = time.time()

    def _get_cached(self, url: str, cache_key: str, max_age_hours: int = 6) -> dict:
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
            if age_hours < max_age_hours:
                return json.loads(cache_path.read_text())

        self._rate_limit()
        resp = self.session.get(url)
        resp.raise_for_status()
        data = resp.json()
        cache_path.write_text(json.dumps(data))
        return data

    def get_bootstrap(self) -> dict:
        return self._get_cached(f"{BASE_URL}/bootstrap-static/", "bootstrap")

    def get_player_summary(self, player_id: int) -> dict:
        return self._get_cached(
            f"{BASE_URL}/element-summary/{player_id}/",
            f"player_{player_id}",
            max_age_hours=12,
        )

    def get_fixtures(self) -> list[dict]:
        return self._get_cached(f"{BASE_URL}/fixtures/", "fixtures")

    def get_entry(self, team_id: int) -> dict:
        return self._get_cached(
            f"{BASE_URL}/entry/{team_id}/", f"entry_{team_id}", max_age_hours=1
        )

    def get_picks(self, team_id: int, gw: int) -> dict:
        return self._get_cached(
            f"{BASE_URL}/entry/{team_id}/event/{gw}/picks/",
            f"picks_{team_id}_{gw}",
            max_age_hours=1,
        )

    def get_entry_history(self, team_id: int) -> dict:
        return self._get_cached(
            f"{BASE_URL}/entry/{team_id}/history/",
            f"entry_history_{team_id}",
            max_age_hours=1,
        )
