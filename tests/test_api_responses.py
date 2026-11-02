#### Preamble ####
# Purpose: Test the stored model responses from OpenRouter: every persona, model and
#          repeat is present exactly once, replies parse to a valid answer, forced-choice
#          answers name only the two allowed candidates, and settings and costs are as
#          recorded. An unanswered call would otherwise flow into the forecast as NA.
# Author: Yi Zhi Zhang
# Date: 28 October 2026
# Contact: yizhiharry.zhang@mail.utoronto.ca
# License: MIT
# Pre-requisites: Run scripts/03-query_models.py (tests skip if the responses are absent)
#
# Run with: uv run pytest tests/test_api_responses.py

#### Workspace setup ####
import importlib.util
from pathlib import Path

import pandera.polars as pa
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "02-analysis_data"
RACES = ROOT / "data" / "01-raw_data" / "senate_races.csv"

spec = importlib.util.spec_from_file_location(
    "query", ROOT / "scripts" / "03-query_models.py"
)
query = importlib.util.module_from_spec(spec)
spec.loader.exec_module(query)

BACKTESTED = ["openai/gpt-4o-mini", "meta-llama/llama-3.3-70b-instruct"]
DEEPSEEK = "deepseek/deepseek-v4-flash"
N_REPEATS = 5

# Each run: (file, year, personas file, models, forced choice, pure independents only)
RUNS = {
    "2024": (2024, "personas_2024", BACKTESTED, False, False),
    "2024_forced": (2024, "personas_2024", BACKTESTED, True, True),
    "2026": (2026, "personas_2026", [*BACKTESTED, DEEPSEEK], False, False),
    "2026_deepseek": (2026, "personas_2026", [DEEPSEEK], False, False),
    "2026_forced": (2026, "personas_2026", [*BACKTESTED, DEEPSEEK], True, True),
}
# Usable runs: the first 2026 DeepSeek answers were replaced by the rerun (its
# reasoning used up the reply limit), so they are checked separately.
MAX_UNPARSED = 0.002  # at most 0.2% of replies per model may be unusable


def load(run: str) -> pl.DataFrame:
    path = DATA / f"responses_{run}.parquet"
    if not path.exists():
        pytest.skip(f"{path.name} absent")
    return pl.read_parquet(path)


def personas(name: str) -> pl.DataFrame:
    return pl.read_parquet(DATA / f"{name}.parquet")


RESPONSE_SCHEMA = pa.DataFrameSchema(
    {
        "persona_id": pa.Column(pl.Int64, nullable=False),
        "state": pa.Column(
            pl.String, pa.Check.str_matches(r"^[A-Z]{2}$"), nullable=False
        ),
        "model": pa.Column(
            pl.String, pa.Check.isin([*BACKTESTED, DEEPSEEK]), nullable=False
        ),
        "repeat": pa.Column(
            pl.Int64, pa.Check.in_range(0, N_REPEATS - 1), nullable=False
        ),
        "year": pa.Column(pl.Int64, pa.Check.isin([2024, 2026]), nullable=False),
        "temperature": pa.Column(pl.Float64, pa.Check.equal_to(1.0), nullable=False),
        "names_in_prompt": pa.Column(
            pl.Boolean, pa.Check.equal_to(False), nullable=False
        ),
        "raw_reply": pa.Column(pl.String, nullable=True),
        "model_returned": pa.Column(pl.String, nullable=True),
        "cost_usd": pa.Column(pl.Float64, pa.Check.in_range(0, 0.01), nullable=False),
        "answer": pa.Column(
            pl.String, pa.Check.isin(query.VALID_ANSWERS), nullable=True
        ),
    }
)


#### Tests ####
@pytest.mark.parametrize("run", RUNS)
def test_schema_and_settings(run):
    r = load(run)
    RESPONSE_SCHEMA.validate(r)
    year, _, models, forced, _ = RUNS[run]
    assert (r["year"] == year).all()
    assert set(r["model"].unique()) == set(models)
    if "forced_choice" in r.columns:
        assert (r["forced_choice"] == forced).all()
    assert r["error"].null_count() == r.height, "an API call returned an error"


@pytest.mark.parametrize("run", RUNS)
def test_every_persona_model_and_repeat_present_exactly_once(run):
    r = load(run)
    _, persona_file, models, _, only_ind = RUNS[run]
    p = personas(persona_file)
    if only_ind:
        p = p.filter(pl.col("pid7") == "Independent")
    expected = p.height * len(models) * N_REPEATS
    assert r.height == expected
    assert not r.select("persona_id", "model", "repeat").is_duplicated().any()
    assert set(r["persona_id"].unique()) == set(p["persona_id"])


@pytest.mark.parametrize("run", RUNS)
def test_state_matches_the_persona(run):
    r = load(run)
    p = personas(RUNS[run][1]).select("persona_id", persona_state="state")
    joined = r.join(p, on="persona_id", how="left")
    assert joined["persona_state"].null_count() == 0
    assert (joined["state"] == joined["persona_state"]).all()


@pytest.mark.parametrize("run", RUNS)
def test_model_returned_is_the_model_requested(run):
    r = load(run).filter(pl.col("model_returned").is_not_null())
    assert r.select(
        pl.col("model_returned").str.starts_with(pl.col("model")).all()
    ).item()


@pytest.mark.parametrize("run", RUNS)
def test_stored_answer_matches_the_parser(run):
    r = load(run)
    reparsed = r["raw_reply"].map_elements(
        query.parse_answer, return_dtype=pl.String, skip_nulls=False
    )
    assert (reparsed.fill_null("NA") == r["answer"].fill_null("NA")).all()


@pytest.mark.parametrize("run", [run for run in RUNS if run != "2026"])
def test_almost_every_reply_is_usable(run):
    r = load(run)
    rates = r.group_by("model").agg(unparsed=pl.col("answer").is_null().mean())
    assert (rates["unparsed"] <= MAX_UNPARSED).all(), rates


def test_first_2026_run_is_usable_for_the_forecast_models():
    # DeepSeek's answers in this run were unusable and replaced by the rerun.
    r = load("2026")
    rates = r.group_by("model").agg(unparsed=pl.col("answer").is_null().mean())
    used = rates.filter(pl.col("model").is_in(BACKTESTED))
    assert (used["unparsed"] <= MAX_UNPARSED).all(), rates
    assert rates.filter(pl.col("model") == DEEPSEEK)["unparsed"].item() > 0.5


MAX_OFF_MENU = 0.06  # forced-choice replies that name neither offered candidate


@pytest.mark.parametrize("run", ["2024_forced", "2026_forced"])
def test_forced_choice_answers_stay_close_to_the_two_candidates(run):
    # The follow-up offers only the Republican and the main opponent. The models
    # sometimes still answer "Independent" where the main opponent is a Democrat
    # (about 4% of replies). These are off the menu; the forecast counts an
    # "Independent" answer only where an independent is on the ballot, so they
    # drop out of the persona's share rather than counting for either side.
    r = load(run).filter(pl.col("answer").is_not_null())
    races = (
        pl.read_csv(RACES)
        .filter(pl.col("year") == RUNS[run][0])
        .select("state", "main_opponent")
    )
    opp = {"Democratic": "Democrat", "Independent": "Independent"}
    j = r.join(races, on="state").with_columns(
        allowed_opp=pl.col("main_opponent").replace_strict(opp)
    )
    assert j.height == r.height, "a response is in a state with no race"
    off_menu = j.filter(
        (pl.col("answer") != "Republican") & (pl.col("answer") != pl.col("allowed_opp"))
    )
    assert set(off_menu["answer"].unique()) <= {"Independent"}, (
        "no 'Other' or 'Would not vote' when forced"
    )
    assert off_menu.height / j.height <= MAX_OFF_MENU


@pytest.mark.parametrize("run", RUNS)
def test_every_state_in_the_run_has_a_race(run):
    r = load(run)
    races = pl.read_csv(RACES).filter(pl.col("year") == RUNS[run][0])
    assert set(r["state"].unique()) <= set(races["state"])


def test_total_cost_is_small_and_positive():
    total = sum(load(run)["cost_usd"].sum() for run in RUNS)
    assert 0 < total < 10
