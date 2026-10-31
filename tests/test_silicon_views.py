#### Preamble ####
# Purpose: Test the figures and tables on the website: the stored results are read
#          correctly, every race is covered, the numbers shown match the data, and
#          the page follows the design grammar (one page, party colours only).
# Author: AUTHOR
# Date: 31 October 2026
# Contact: EMAIL
# License: MIT
# Pre-requisites: the parquet files in data/02-analysis_data (tests skip without them)
#
# Run with: uv run pytest tests/test_silicon_views.py

#### Workspace setup ####
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "02-analysis_data"
NEEDED = ["forecast_2026_summary", "forecast_2026_draws", "forecast_2026_seats", "backtest_2024_states",
          "backtest_2024_by_party", "calibration_2024", "personas_2024", "responses_2024",
          "responses_2024_forced", "responses_2026", "responses_2026_deepseek", "responses_2026_forced",
          "model_selection", "regression_forecast_2026"]  # fmt: skip
pytestmark = pytest.mark.skipif(not all((DATA / f"{n}.parquet").exists() for n in NEEDED), reason="result files absent")

spec = importlib.util.spec_from_file_location("si", ROOT / "scripts" / "08-silicon_views.py")
si = importlib.util.module_from_spec(spec)
spec.loader.exec_module(si)


@pytest.fixture(scope="module")
def sd():
    return si.load(ROOT)


#### Data behind the figures ####
def test_every_2026_race_has_a_forecast(sd):
    s = sd["summary"]
    assert s.height == 35 and s["state"].n_unique() == 35
    assert s["p_rep_win"].is_between(0, 1).all()
    assert ((s["lower_80"] <= s["median"]) & (s["median"] <= s["upper_80"])).all()
    assert set(si.MODEL_ORDER) <= set(s.columns)


def test_control_probability_matches_the_seat_draws(sd):
    c = si.control_stats(sd["seats"])
    assert c["p_rep_control"] == pytest.approx(float((sd["seats"]["rep_seats"] >= si.MAJORITY_REP).mean()))
    assert c["lo"] <= c["median"] <= c["hi"]


def test_quantile_dots_give_twenty_ordered_outcomes_per_state(sd):
    d = si.quantile_dots(sd["draws"])
    counts = d.groupby("state").size()
    assert len(counts) == 35 and (counts == 20).all()
    for _, g in d.groupby("state"):
        assert g.sort_values("q")["share"].is_monotonic_increasing


def test_independent_shares_sum_to_one_for_each_source(sd):
    sh = si.independents_shares(sd)
    assert set(sh["source"]) == {"Free answer", "Forced answer", "Real vote"}
    assert sh.groupby("source")["share"].sum().round(9).eq(1).all()
    assert set(sh[sh["source"] == "Real vote"]["side"]) <= {"Republican", "Opposition"}
    assert sh.attrs["n_personas"] > 0 and sh.attrs["n_real"] <= sh.attrs["n_personas"]


def test_regression_comparison_covers_every_race(sd):
    c = si.compare_with_regression(sd["summary"], ROOT)
    assert len(c) == 35 and c["regression"].between(0, 1).all()
    assert set(c["call"]) <= {"Both: Republican", "Both: opposition", "Methods disagree"}


#### Tables ####
def test_cost_table_total_matches_the_responses(sd):
    t = si.cost_table(sd["resp"])
    total = sum(df["cost_usd"].sum() for df in sd["resp"].values())
    assert t.attrs["total_cost"] == pytest.approx(total)
    assert t["Run"].iloc[-1] == "Total" and len(t) == len(sd["resp"]) + 1


def test_forecast_table_never_claims_certainty(sd):
    t = si.forecast_table(sd["summary"], sd["races_2026"])
    assert len(t) == 35
    assert not t["P(Republican wins)"].isin(["0%", "100%"]).any()
    assert si.share_text(0.999) == ">99%" and si.share_text(0.001) == "<1%" and si.share_text(0.5) == "50%"


def test_accuracy_table_has_both_methods(sd):
    t = si.accuracy_table(sd["bt"], ROOT)
    assert len(t) == 3 and t["Method"].str.contains("Regression").sum() == 1


def test_booktabs_has_no_vertical_rules():
    import pandas as pd

    html = si.booktabs(pd.DataFrame({"A": ["x"], "B": ["1"]}), ["B"])
    assert html.startswith('<table class="booktabs">') and "border" not in html


#### Figures and design grammar ####
def test_every_figure_builds(sd, tmp_path):
    figs = [si.fig_seats(sd["seats"]), si.fig_state_dots(sd["draws"], sd["summary"]),
            si.fig_backtest(sd["bt"], sd["calib"]), si.fig_by_party(sd["by_party"]),
            si.fig_independents(si.independents_shares(sd)),
            si.fig_vs_regression(si.compare_with_regression(sd["summary"], ROOT)), si.fig_models(sd["summary"])]  # fmt: skip
    for i, f in enumerate(figs):
        f.save(tmp_path / f"f{i}.png", dpi=40, verbose=False)
        assert (tmp_path / f"f{i}.png").stat().st_size > 3000


def test_only_the_agreed_colours_are_used():
    text = (ROOT / "scripts" / "08-silicon_views.py").read_text()
    import re

    colours = {c.upper() for c in re.findall(r'"(#[0-9A-Fa-f]{6})"', text)}
    assert colours <= {"#B4232C", "#1F4E8C", "#8A8A8A", "#222222", "#E6E6E6", "#F2F2F2"}


def test_page_is_one_document_with_all_sections():
    text = (ROOT / "paper" / "index.qmd").read_text()
    for anchor in ["{#forecast}", "{#method}", "{#backtest}", "{#independents}", "{#second-opinions}", "{#limits}", "{#appendix}"]:
        assert anchor in text
    cfg = (ROOT / "paper" / "_quarto.yml").read_text()
    assert "navbar" not in cfg and "github.com/harryyz/US-midterms-2026-silicon-forecast" in cfg
