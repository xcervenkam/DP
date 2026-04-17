from typing import Any, Dict, Optional
import time
import requests


class APIFootballClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        request_sleep_seconds: float = 3.0,
        max_retries: int = 5,
        retry_backoff_seconds: float = 8.0,
    ) -> None:
        if not api_key:
            raise ValueError("Missing API key.")

        self.base_url = base_url.rstrip("/")
        self.request_sleep_seconds = request_sleep_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

        self.session = requests.Session()
        self.session.headers.update({
            "x-apisports-key": api_key
        })

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        for attempt in range(self.max_retries + 1):
            response = self.session.get(url, params=params, timeout=60)

            if response.status_code == 200:
                time.sleep(self.request_sleep_seconds)
                return response.json()

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after is not None:
                    wait_seconds = float(retry_after)
                else:
                    wait_seconds = self.retry_backoff_seconds * (attempt + 1)

                print(f"429 Too Many Requests -> waiting {wait_seconds:.1f}s before retry")
                time.sleep(wait_seconds)
                continue

            print("STATUS:", response.status_code)
            print("URL:", response.url)
            print("TEXT:", response.text[:1000])
            response.raise_for_status()

        raise RuntimeError(f"Request failed after retries: {url}")

    def test_connection(self) -> Dict[str, Any]:
        return self._get("countries")

    def get_leagues(self, country: Optional[str] = None, season: Optional[int] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if country is not None:
            params["country"] = country
        if season is not None:
            params["season"] = season
        return self._get("leagues", params=params)

    def get_teams(self, league_id: int, season: int) -> Dict[str, Any]:
        return self._get("teams", params={"league": league_id, "season": season})

    def get_fixtures(self, league_id: int, season: int) -> Dict[str, Any]:
        return self._get("fixtures", params={"league": league_id, "season": season})

    def get_standings(self, league_id: int, season: int) -> Dict[str, Any]:
        return self._get("standings", params={"league": league_id, "season": season})

    def get_fixture_statistics(self, fixture_id: int) -> Dict[str, Any]:
        return self._get("fixtures/statistics", params={"fixture": fixture_id})

    def get_team_statistics(self, league_id: int, season: int, team_id: int, date: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "league": league_id,
            "season": season,
            "team": team_id,
        }
        if date is not None:
            params["date"] = date
        return self._get("teams/statistics", params=params)


def resolve_league_id(leagues_response: Dict[str, Any], league_name_hint: str) -> int:
    candidates = []

    for item in leagues_response.get("response", []):
        league_name = item.get("league", {}).get("name", "")
        if league_name_hint.lower() in league_name.lower():
            candidates.append(item)

    if not candidates:
        available = [
            item.get("league", {}).get("name", "")
            for item in leagues_response.get("response", [])
        ]
        raise ValueError(
            f"League matching '{league_name_hint}' was not found. Available leagues: {available}"
        )

    return int(candidates[0]["league"]["id"])