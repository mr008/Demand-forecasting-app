"""Column specifications for the raw CSVs and lightweight validation.

We deliberately avoid a heavy schema library: the four files are small in
schema terms, and a dict of expected dtypes plus a few invariants is enough to
fail loudly when an input changes shape.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd


class SchemaError(ValueError):
    """Raised when a raw file does not match its expected schema."""


SELL_OUT_DTYPES: Mapping[str, str] = {
    "date": "datetime64[ns]",
    "sell_out_pzs": "float64",
    "upc": "int64",
    "cedis": "string",
    "final_price": "float64",
    "promo_flag": "int8",
}

INVENTORY_DTYPES: Mapping[str, str] = {
    "date": "datetime64[ns]",
    "prime_item_nbr": "int64",
    "upc": "int64",
    "store_nbr": "int64",
    "on_hand_qty": "float64",
}

STORE_CATALOG_DTYPES: Mapping[str, str] = {
    "store_nbr": "int64",
    "cedis": "string",
}

UPC_CATALOG_DTYPES: Mapping[str, str] = {
    "prime_item_nbr": "int64",
    "upc": "int64",
    "avg_daily_sales": "float64",
    "median_daily_sales": "float64",
    "std_daily_sales": "float64",
    "cv_demand": "float64",
    "sales_share": "float64",
    "abc_class": "string",
    "xyz_class": "string",
    "lead_time_days": "int64",
    "moq": "int64",
    "safety_stock_days": "int64",
}


def validate(
    df: pd.DataFrame,
    dtypes: Mapping[str, str],
    name: str,
    *,
    key: Iterable[str] | None = None,
    non_null: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Check required columns, cast dtypes, and enforce key/non-null invariants.

    Returns a new frame restricted to the specified columns, in spec order.
    """
    missing = [c for c in dtypes if c not in df.columns]
    if missing:
        raise SchemaError(f"{name}: missing columns {missing}")

    out = df[list(dtypes)].copy()
    for col, dtype in dtypes.items():
        try:
            if dtype.startswith("datetime64"):
                out[col] = pd.to_datetime(out[col], errors="raise")
            else:
                out[col] = out[col].astype(dtype)
        except (ValueError, TypeError) as exc:
            raise SchemaError(f"{name}: column {col!r} is not castable to {dtype}: {exc}") from exc

    for col in non_null or ():
        n_null = int(out[col].isna().sum())
        if n_null:
            raise SchemaError(f"{name}: column {col!r} has {n_null} nulls")

    if key is not None:
        key = list(key)
        n_dup = int(out.duplicated(key).sum())
        if n_dup:
            raise SchemaError(f"{name}: {n_dup} duplicate rows on key {key}")

    return out
