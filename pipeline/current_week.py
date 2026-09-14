"""Print the upcoming NFL week, for the scheduled build to consume."""

from __future__ import annotations

from datetime import date

from .config import CURRENT_SEASON

# The Thursday of week one. Update once a year, or read it from the schedule
# if the nflverse data is already cached locally.
SEASON_OPENER = {2025: date(2025, 9, 4), 2026: date(2026, 9, 10)}


def current_week(today: date | None = None, season: int = CURRENT_SEASON) -> int:
    today = today or date.today()
    opener = SEASON_OPENER.get(season)
    if opener is None:
        return 1
    weeks = (today - opener).days // 7 + 1
    return max(1, min(18, weeks))


if __name__ == "__main__":
    print(current_week())
