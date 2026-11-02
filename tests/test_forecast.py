#### Preamble ####
# Purpose: Test the forecast outputs of scripts/04-forecast.py: no missing values,
#          shares and probabilities between 0 and 1, the Republican and opposition
#          shares sum to 1, every race and simulation present, seat counts that
#          agree with the state draws, and summaries that agree with the draws.
# Author: Yi Zhi Zhang
# Date: 28 October 2026
# Contact: yizhiharry.zhang@mail.utoronto.ca
# License: MIT
# Pre-requisites: Run scripts/04-forecast.py (tests skip if the outputs are absent)
#
# Run with: uv run pytest tests/test_forecast.py

#### Workspace setup ####
from pathlib import Path

import numpy as np
import pandera.polars as pa
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "02-analysis_data"

BACKTESTED = ["openai/gpt-4o-mini", "meta-llama/llama-3.3-70b-instruct"]
ALL_MODELS = [*BACKTESTED, "deepseek/deepseek-v4-flash"]
N_DRAWS = 10_000
N_RACES_2026 = 35
N_RACES_2024 = 33
SEATS_NOT_UP_REP = 31  # Republican seats not up in 2026 (scripts/04-forecast.py)


def load(name: str) -> pl.DataFrame:
    path = DATA / f"{name}.parquet"
    if not path.exists():
        pytest.skip(f"{path.name} absent")
    return pl.read_parquet(path)


share = pa.Check.in_range(0, 1)

DRAWS_SCHEMA = pa.DataFrameSchema(
    {
        "draw": pa.Column(pl.Int64, pa.Check.in_range(0, N_DRAWS - 1), nullable=False),
        "state": pa.Column(
            pl.String, pa.Check.str_matches(r"^[A-Z]{2}$"), nullable=False
        ),
        "model": pa.Column(pl.String, pa.Check.isin(BACKTESTED), nullable=False),
        "rep_share": pa.Column(pl.Float64, share, nullable=False),
        "opp_share": pa.Column(pl.Float64, share, nullable=False),
    }
)
SUMMARY_SCHEMA = pa.DataFrameSchema(
    {
        "state": pa.Column(
            pl.String, pa.Check.str_matches(r"^[A-Z]{2}$"), unique=True, nullable=False
        ),
        "p_rep_win": pa.Column(pl.Float64, share, nullable=False),
        "median": pa.Column(pl.Float64, share, nullable=False),
        "lower_80": pa.Column(pl.Float64, share, nullable=False),
        "upper_80": pa.Column(pl.Float64, share, nullable=False),
        **{m: pa.Column(pl.Float64, share, nullable=False) for m in ALL_MODELS},
    }
)
ESTIMATES_SCHEMA = pa.DataFrameSchema(
    {
        "state": pa.Column(pl.String, nullable=False),
        "model": pa.Column(pl.String, pa.Check.isin(ALL_MODELS), nullable=False),
        "rep_share": pa.Column(pl.Float64, share, nullable=False),
        "se": pa.Column(
            pl.Float64, pa.Check.in_range(0, 0.2, include_min=False), nullable=False
        ),
        "n_personas": pa.Column(pl.Int64, pa.Check.ge(50), nullable=False),
    }
)


#### State draws ####
def test_draws_schema_and_size():
    d = load("forecast_2026_draws")
    DRAWS_SCHEMA.validate(d)
    assert d.height == N_DRAWS * N_RACES_2026
    assert d["state"].n_unique() == N_RACES_2026 and d["draw"].n_unique() == N_DRAWS
    assert not d.select("draw", "state").is_duplicated().any()


def test_republican_and_opposition_shares_sum_to_one():
    d = load("forecast_2026_draws")
    assert np.allclose(d["rep_share"] + d["opp_share"], 1.0)


def test_each_simulation_uses_one_model_for_every_state():
    d = load("forecast_2026_draws")
    per_draw = d.group_by("draw").agg(pl.col("model").n_unique())
    assert (per_draw["model"] == 1).all()
    # Both backtested models are used, in roughly equal numbers
    counts = d.filter(pl.col("state") == d["state"][0])["model"].value_counts()
    assert counts.height == 2 and counts["count"].min() > 0.45 * N_DRAWS


def test_draws_are_spread_out_not_degenerate():
    d = load("forecast_2026_draws")
    sd = d.group_by("state").agg(pl.col("rep_share").std())
    assert sd["rep_share"].is_between(0.02, 0.15).all()


#### Seats ####
def test_seat_counts_match_the_state_draws():
    d, seats = load("forecast_2026_draws"), load("forecast_2026_seats")
    recount = d.group_by("draw").agg(won=(pl.col("rep_share") > 0.5).sum())
    joined = seats.join(recount, on="draw")
    assert joined.height == N_DRAWS
    assert (
        joined["rep_seats"].cast(pl.Int64)
        == SEATS_NOT_UP_REP + joined["won"].cast(pl.Int64)
    ).all()


def test_seat_counts_are_possible():
    seats = load("forecast_2026_seats")
    assert seats["rep_seats"].null_count() == 0
    assert (
        seats["rep_seats"]
        .is_between(SEATS_NOT_UP_REP, SEATS_NOT_UP_REP + N_RACES_2026)
        .all()
    )


def test_control_is_uncertain_not_certain():
    seats = load("forecast_2026_seats")
    p_control = float((seats["rep_seats"] >= 50).mean())
    assert 0.01 < p_control < 0.99


#### Race summaries ####
def test_summary_schema_and_order():
    s = load("forecast_2026_summary")
    SUMMARY_SCHEMA.validate(s)
    assert s.height == N_RACES_2026
    assert ((s["lower_80"] <= s["median"]) & (s["median"] <= s["upper_80"])).all()


def test_summary_agrees_with_the_draws():
    d, s = load("forecast_2026_draws"), load("forecast_2026_summary")
    recomputed = d.group_by("state").agg(
        p=(pl.col("rep_share") > 0.5).mean(), med=pl.col("rep_share").median()
    )
    j = s.join(recomputed, on="state")
    assert j.height == N_RACES_2026
    assert np.allclose(j["p_rep_win"], j["p"], atol=1e-9)
    assert np.allclose(j["median"], j["med"], atol=1e-9)


def test_summary_states_match_the_draws():
    d, s = load("forecast_2026_draws"), load("forecast_2026_summary")
    assert set(s["state"]) == set(d["state"].unique())


#### Inputs to the simulation ####
def test_state_estimates_cover_every_race_and_model():
    e = load("estimates_2026_by_model")
    ESTIMATES_SCHEMA.validate(e)
    assert e.height == N_RACES_2026 * len(ALL_MODELS)
    assert not e.select("state", "model").is_duplicated().any()


def test_calibration_lines_shrink_toward_the_middle():
    c = load("calibration_2024")
    assert set(c["model"]) == set(BACKTESTED)
    assert c["slope"].is_between(0.3, 1.0).all()
    assert c["intercept"].is_between(0, 0.5).all()


def test_backtest_errors_are_actual_minus_estimate():
    b = load("backtest_2024_states")
    assert b.height == N_RACES_2024 * len(BACKTESTED)
    assert b.null_count().sum_horizontal().item() == 0
    assert np.allclose(b["error"], b["actual"] - b["rep_share"])
    assert (b["winner_right"] == ((b["actual"] > 0.5) == (b["rep_share"] > 0.5))).all()
