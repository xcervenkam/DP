import requests
from typing import Any, Dict, Optional


class APIFootballClient:
    def __init__(self, api_key: str, api_host: str, base_url: str) -> None:
        if not api_key:
            raise ValueError("Chybí API klíč. Zkontroluj .env soubor.")
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "x-apisports-key": api_key,
                "x-rapidapi-host": api_host,
            }
        )

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.get(url, params=params, timeout=60)
        response.raise_for_status()
        return response.json()

    def get_leagues(self, country: Optional[str] = None, season: Optional[int] = None) -> Dict[str, Any]:
        params = {}
        if country is not None:
            params["country"] = country
        if season is not None:
            params["season"] = season
        return self._get("leagues", params=params)

    def get_teams(self, league_id: int, season: int) -> Dict[str, Any]:
        return self._get("teams", params={"league": league_id, "season": season})

    def get_fixtures(self, league_id: int, season: int) -> Dict[str, Any]:
        return self._get("fixtures", params={"league": league_id, "season": season})

    def get_fixture_statistics(self, fixture_id: int) -> Dict[str, Any]:
        return self._get("fixtures/statistics", params={"fixture": fixture_id})

    def get_standings(self, league_id: int, season: int) -> Dict[str, Any]:
        return self._get("standings", params={"league": league_id, "season": season})

    def get_team_statistics(self, league_id: int, season: int, team_id: int, date: Optional[str] = None) -> Dict[str, Any]:
        params = {"league": league_id, "season": season, "team": team_id}
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
        raise ValueError(f"Nenalezena liga podle hintu: {league_name_hint}")

    return int(candidates[0]["league"]["id"])