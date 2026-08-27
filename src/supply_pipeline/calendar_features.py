"""Calendar effects for Mexican retail demand.

Known-in-advance features only, so they are safe to use for future weeks:
weekday, quincena paydays, federal holidays, Semana Santa, El Buen Fin, and the
December peak. Everything is derived from the date alone.
"""

from __future__ import annotations

from datetime import date, timedelta

import holidays
import pandas as pd
from dateutil.easter import easter

DAILY_FEATURE_COLS = (
    "is_payday",
    "is_payday_window",
    "is_holiday",
    "is_semana_santa",
    "is_buen_fin",
    "is_december_peak",
)


def _third_monday_of_november(year: int) -> date:
    first = date(year, 11, 1)
    offset = (7 - first.weekday()) % 7  # days until first Monday
    return first + timedelta(days=offset + 14)


def buen_fin_dates(year: int) -> set[date]:
    """El Buen Fin: the Friday-to-Monday weekend ending on Revolution Day (3rd Monday of November)."""
    monday = _third_monday_of_november(year)
    return {monday - timedelta(days=k) for k in range(0, 4)}


def semana_santa_dates(year: int) -> set[date]:
    """Holy Week: Palm Sunday through Easter Sunday."""
    easter_sunday = easter(year)
    return {easter_sunday - timedelta(days=k) for k in range(0, 8)}


def build_calendar(start: date, end: date) -> pd.DataFrame:
    """One row per day in [start, end] with boolean calendar features."""
    idx = pd.date_range(start, end, freq="D")
    years = range(start.year, end.year + 1)
    mx_holidays = holidays.country_holidays("MX", years=list(years))
    buen_fin = set().union(*(buen_fin_dates(y) for y in years))
    semana_santa = set().union(*(semana_santa_dates(y) for y in years))

    df = pd.DataFrame({"date": idx})
    d = df["date"].dt
    month_end = d.is_month_end
    df["dow"] = d.dayofweek.astype("int8")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    df["is_payday"] = ((d.day == 15) | month_end).astype("int8")
    # Two days after payday still carry the spending bump.
    payday_series = pd.Series(df["is_payday"].to_numpy(), index=idx)
    window = payday_series.rolling(3, min_periods=1).max()
    df["is_payday_window"] = window.to_numpy().astype("int8")
    dates = [ts.date() for ts in idx]
    df["is_holiday"] = pd.Series([int(x in mx_holidays) for x in dates], dtype="int8").to_numpy()
    df["is_semana_santa"] = pd.Series([int(x in semana_santa) for x in dates], dtype="int8").to_numpy()
    df["is_buen_fin"] = pd.Series([int(x in buen_fin) for x in dates], dtype="int8").to_numpy()
    df["is_december_peak"] = ((d.month == 12) & (d.day >= 15)).astype("int8")
    df["week_start"] = (df["date"] - pd.to_timedelta(df["dow"], unit="D")).dt.normalize()
    return df


def weekly_calendar(daily_calendar: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily calendar features into counts per ISO week (Monday start)."""
    agg = {c: "sum" for c in DAILY_FEATURE_COLS}
    weekly = daily_calendar.groupby("week_start", as_index=False).agg(agg)
    weekly = weekly.rename(columns={c: c.replace("is_", "") + "_days" for c in DAILY_FEATURE_COLS})
    iso = weekly["week_start"].dt.isocalendar()
    weekly["week_of_year"] = iso["week"].astype("int16").to_numpy()
    weekly["month"] = weekly["week_start"].dt.month.astype("int8")
    return weekly
