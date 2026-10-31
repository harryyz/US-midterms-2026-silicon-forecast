#### Preamble ####
# Purpose: Figures and tables for the website, built only from stored results in
#          data/02-analysis_data. Nothing here calls an LLM; the numbers come from
#          scripts/03-query_models.py (responses), scripts/04-forecast.py (backtest,
#          calibration, forecast) and scripts/05-regression.py (regression forecast).
# Author: AUTHOR
# Date: 31 October 2026
# Contact: EMAIL
# License: MIT
# Pre-requisites: the parquet files from scripts/03, 04 and 05; senate_races.csv
#
# Design grammar (one style for every figure and table on the page):
#   - Typeface: Newsreader (bundled in paper/fonts, SIL Open Font License)
#   - Colour only where it means a party: Republican #B4232C, opposition #1F4E8C.
#     Everything else is black #222222 or grey #8A8A8A, told apart by shape.
#   - Newspaper charts: white background, faint horizontal grid only, no box,
#     one dashed reference line per chart, direct labels where possible.
#   - Tables: booktabs (three rules, no vertical lines, right-aligned numbers).
#
# Run with: imported by paper/index.qmd

#### Workspace setup ####
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from matplotlib import font_manager
from plotnine import (
    aes,
    coord_flip,
    element_blank,
    element_line,
    element_text,
    facet_wrap,
    geom_abline,
    geom_col,
    geom_hline,
    geom_point,
    geom_segment,
    geom_text,
    geom_vline,
    ggplot,
    labs,
    position_dodge,
    scale_color_manual,
    scale_fill_manual,
    scale_shape_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_minimal,
)  # fmt: skip

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

REP, OPP, GREY, INK = "#B4232C", "#1F4E8C", "#8A8A8A", "#222222"
GRID = "#E6E6E6"
FONT = "Newsreader"
for _f in (ROOT / "paper" / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(_f))

MODEL_NAME = {
    "openai/gpt-4o-mini": "GPT-4o mini",
    "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
    "deepseek/deepseek-v4-flash": "DeepSeek V4 Flash",
}
MODEL_ORDER = ["GPT-4o mini", "Llama 3.3 70B", "DeepSeek V4 Flash"]
BACKTESTED = ["GPT-4o mini", "Llama 3.3 70B"]  # only these have a 2024 backtest
PID7_ORDER = ["Strong Democrat", "Not Very Strong Democrat", "Lean Democrat", "Independent",
              "Lean Republican", "Not Very Strong Republican", "Strong Republican"]  # fmt: skip
PID7_LABEL = {"Not Very Strong Democrat": "Weak Democrat", "Not Very Strong Republican": "Weak Republican",
              "Independent": "Pure independent"}  # fmt: skip
SEATS_NOT_UP_REP = 31  # as assumed in scripts/04-forecast.py
MAJORITY_REP = 50  # the Vice President breaks a 50-50 tie
STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
    "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}  # fmt: skip


def pct(values):
    return [f"{v:.0%}" for v in values]


def theme_news(width=7.0, height=4.0, **extra):
    """The one chart style used on the page. Keyword arguments override the defaults."""
    settings = {
        "figure_size": (width, height),
        "panel_grid_major_x": element_blank(),
        "panel_grid_minor": element_blank(),
        "panel_grid_major_y": element_line(color=GRID, size=0.5),
        "axis_line_x": element_line(color=INK, size=0.6),
        "axis_ticks_major_x": element_line(color=INK, size=0.5),
        "axis_text": element_text(color=INK, size=10),
        "axis_title": element_text(color=INK, size=10.5),
        "legend_position": "top",
        "legend_title": element_blank(),
        "legend_text": element_text(size=10),
        "strip_text": element_text(size=11, ha="left"),
        "plot_background": element_blank(),
    }
    settings.update(extra)
    return theme_minimal(base_family=FONT, base_size=11) + theme(**settings)


#### Loading ####
def load(root: Path = ROOT) -> dict:
    d = root / "data" / "02-analysis_data"

    def rename_models(df):
        return df.with_columns(model=pl.col("model").replace(MODEL_NAME))

    summary = pl.read_parquet(d / "forecast_2026_summary.parquet")
    summary = summary.rename({c: MODEL_NAME[c] for c in summary.columns if c in MODEL_NAME})
    resp = {n: pl.read_parquet(d / f"{n}.parquet") for n in
            ["responses_2024", "responses_2024_forced", "responses_2026", "responses_2026_deepseek",
             "responses_2026_forced"]}  # fmt: skip
    races = pl.read_csv(root / "data" / "01-raw_data" / "senate_races.csv")
    return {
        "summary": summary,
        "draws": pl.read_parquet(d / "forecast_2026_draws.parquet"),
        "seats": pl.read_parquet(d / "forecast_2026_seats.parquet"),
        "bt": rename_models(pl.read_parquet(d / "backtest_2024_states.parquet")),
        "calib": rename_models(pl.read_parquet(d / "calibration_2024.parquet")),
        "ind": pl.read_parquet(d / "backtest_2024_individual.parquet"),
        "by_party": pl.read_parquet(d / "backtest_2024_by_party.parquet"),
        "personas_2024": pl.read_parquet(d / "personas_2024.parquet"),
        "races_2024": races.filter(pl.col("year") == 2024),
        "races_2026": races.filter(pl.col("year") == 2026),
        "resp": resp,
        "root": root,
    }


def control_stats(seats: pl.DataFrame) -> dict:
    s = seats["rep_seats"]
    return {"p_rep_control": float((s >= MAJORITY_REP).mean()), "median": float(s.median()),
            "lo": float(s.quantile(0.1)), "hi": float(s.quantile(0.9))}  # fmt: skip


def state_order(summary: pl.DataFrame) -> list[str]:
    """Full state names, most Democratic first (plotted bottom to top)."""
    return [STATE_NAMES[s] for s in summary.sort("median")["state"]]


#### FIG A: Senate seats ####
def fig_seats(seats: pl.DataFrame):
    d = (
        seats.group_by("rep_seats").len()
        .with_columns(share=pl.col("len") / seats.height,
                      side=pl.when(pl.col("rep_seats") >= MAJORITY_REP)
                      .then(pl.lit("Republican control")).otherwise(pl.lit("Opposition control")))
        .to_pandas()
    )  # fmt: skip
    top = d["share"].max()
    return (
        ggplot(d, aes("rep_seats", "share", fill="side"))
        + geom_col(width=0.8)
        + geom_vline(xintercept=MAJORITY_REP - 0.5, color=INK, linetype="dashed", size=0.6)
        + geom_text(aes(x=MAJORITY_REP - 0.3, y=top * 1.04), label="50 seats: control",
                    ha="left", size=9, family=FONT, color=INK, inherit_aes=False)
        + scale_fill_manual({"Republican control": REP, "Opposition control": OPP})
        + scale_x_continuous(breaks=list(range(40, 62, 2)))
        + scale_y_continuous(labels=pct, expand=(0, 0, 0.08, 0))
        + labs(x="Republican seats after the election", y="Share of simulations")
        + theme_news(7, 3.6)
    )


#### FIG B: State forecasts as quantile dotplots ####
def quantile_dots(draws: pl.DataFrame, n: int = 20) -> pd.DataFrame:
    """For each state, n equally likely outcomes: the (i + 0.5) / n quantiles of the draws."""
    qs = [(i + 0.5) / n for i in range(n)]
    rows = []
    for state, g in draws.group_by("state"):
        for q in qs:
            rows.append({"state": state[0], "q": q, "share": float(np.quantile(g["rep_share"], q))})
    return pd.DataFrame(rows)


def fig_state_dots(draws: pl.DataFrame, summary: pl.DataFrame):
    d = quantile_dots(draws)
    d["name"] = pd.Categorical(d["state"].map(STATE_NAMES), categories=state_order(summary), ordered=True)
    d["winner"] = np.where(d["share"] > 0.5, "Republican wins", "Opposition wins")
    return (
        ggplot(d, aes("share", "name", color="winner"))
        + geom_vline(xintercept=0.5, color=INK, linetype="dashed", size=0.5)
        + geom_point(size=1.6, stroke=0)
        + scale_color_manual({"Republican wins": REP, "Opposition wins": OPP})
        + scale_x_continuous(labels=pct, breaks=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        + labs(x="Republican share of the vote against the main opposition", y="")
        + theme_news(7, 9.5, panel_grid_major_y=element_line(color="#F2F2F2", size=0.4))
    )


#### FIG C: pipeline (HTML, styled in styles.scss) ####
PIPELINE = [
    ("CES 2024", "real survey respondents"),
    ("Persona", "first-person backstory plus the 2026 ballot"),
    ("3 LLMs", "5 answers each"),
    ("Weights", "survey weight × midterm turnout (CES 2022)"),
    ("Calibration", "line fitted to the 2024 backtest"),
    ("Simulation", "10,000 elections"),
]


def pipeline_html() -> str:
    steps = "".join(
        f'<div class="step"><div class="step-title">{t}</div><div class="step-text">{s}</div></div>'
        for t, s in PIPELINE
    )
    return f'<div class="pipeline">{steps}</div>'


#### FIG D: 2024 backtest ####
def fig_backtest(bt: pl.DataFrame, calib: pl.DataFrame):
    d = bt.with_columns(name=pl.col("state").replace(STATE_NAMES), miss=pl.col("error").abs()).to_pandas()
    worst = d.sort_values("miss", ascending=False).groupby("model").head(3)
    lines = pd.DataFrame([{"model": r["model"], "x": 0.15, "y": r["intercept"] + r["slope"] * 0.15,
                           "xend": 0.85, "yend": r["intercept"] + r["slope"] * 0.85}
                          for r in calib.iter_rows(named=True)])  # fmt: skip
    return (
        ggplot(d, aes("rep_share", "actual"))
        + geom_abline(slope=1, intercept=0, color=GREY, linetype="dashed", size=0.5)
        + geom_segment(aes(x="x", y="y", xend="xend", yend="yend"), data=lines, color=INK, size=0.8, inherit_aes=False)
        + geom_point(color=GREY, size=2.2, stroke=0)
        + geom_text(aes(label="name"), data=worst, size=8, family=FONT, color=INK, nudge_y=0.025)
        + facet_wrap("model")
        + scale_x_continuous(labels=pct, limits=(0.15, 0.85))
        + scale_y_continuous(labels=pct, limits=(0.15, 0.85))
        + labs(x="Republican share from the synthetic voters (before calibration)", y="Actual Republican share")
        + theme_news(7.5, 4.0, panel_grid_major_x=element_line(color=GRID, size=0.5))
    )


#### FIG E: agreement by party identification ####
def fig_by_party(by_party: pl.DataFrame):
    keep = {"openai/gpt-4o-mini": "GPT-4o mini", "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
            "party-line baseline": "Party-line baseline"}  # fmt: skip
    d = by_party.filter(pl.col("model").is_in(list(keep)), pl.col("pid7").is_in(PID7_ORDER)).to_pandas()
    d["who"] = pd.Categorical(d["model"].map(keep), categories=list(keep.values()), ordered=True)
    d["group"] = pd.Categorical(d["pid7"].map(lambda p: PID7_LABEL.get(p, p)),
                                categories=[PID7_LABEL.get(p, p) for p in PID7_ORDER][::-1], ordered=True)  # fmt: skip
    # Group on the x axis and flip, so the dodge separates the markers vertically
    # without changing their agreement values.
    return (
        ggplot(d, aes("group", "agreement", shape="who", color="who", fill="who"))
        + geom_point(size=3, stroke=0.8, position=position_dodge(width=0.55))
        + coord_flip()
        + scale_shape_manual({"GPT-4o mini": "o", "Llama 3.3 70B": "o", "Party-line baseline": "D"})
        + scale_color_manual({"GPT-4o mini": INK, "Llama 3.3 70B": INK, "Party-line baseline": GREY})
        + scale_fill_manual({"GPT-4o mini": INK, "Llama 3.3 70B": "white", "Party-line baseline": GREY})
        + scale_y_continuous(labels=pct, limits=(0.5, 1.0))
        + labs(x="", y="Share who pick the same side as their real 2024 vote")
        + theme_news(7, 4.2, panel_grid_major_y=element_blank(), panel_grid_major_x=element_line(color=GRID, size=0.5))
    )


#### FIG F: the independents problem ####
def independents_shares(sd: dict) -> pd.DataFrame:
    """2024 pure independents: free answers, forced answers and their real votes.
    Opposition = a Democrat or independent actually on that state's ballot.
    Every row is weighted by the respondent's CES survey weight."""
    races = sd["races_2024"].select(
        "state", has_dem=pl.col("democrat").is_not_null(), has_ind=pl.col("independent").is_not_null()
    )
    p = sd["personas_2024"].filter(pl.col("pid7") == "Independent").select(
        "persona_id", "state", "voted_sen_party", w=pl.col("weight").fill_null(0)
    ).join(races, on="state")

    def side(label: pl.Expr) -> pl.Expr:
        opp = ((label == "Democrat") & pl.col("has_dem")) | ((label == "Independent") & pl.col("has_ind"))
        return pl.when(label == "Republican").then(pl.lit("Republican")).when(opp).then(pl.lit("Opposition")).otherwise(pl.lit("Other"))

    rows = []
    for source, df in [("Free answer", sd["resp"]["responses_2024"]), ("Forced answer", sd["resp"]["responses_2024_forced"])]:
        a = df.join(p, on=["persona_id", "state"]).with_columns(side=side(pl.col("answer")))
        if source == "Forced answer":
            a = a.filter(pl.col("answer").is_not_null())
        g = a.group_by("side").agg(w=pl.col("w").sum())
        rows += [{"source": source, "side": r["side"], "share": r["w"] / g["w"].sum()} for r in g.iter_rows(named=True)]
    real = p.with_columns(side=side(pl.col("voted_sen_party").replace({"Democratic": "Democrat"}))).filter(
        pl.col("voted_sen_party").is_not_null(), pl.col("side") != "Other"
    )
    g = real.group_by("side").agg(w=pl.col("w").sum())
    rows += [{"source": "Real vote", "side": r["side"], "share": r["w"] / g["w"].sum()} for r in g.iter_rows(named=True)]
    out = pd.DataFrame(rows)
    out.attrs["n_real"] = real.height
    out.attrs["n_personas"] = p.height
    return out


def fig_independents(shares: pd.DataFrame):
    d = shares.copy()
    d["source"] = pd.Categorical(d["source"], categories=["Real vote", "Forced answer", "Free answer"], ordered=True)
    d["side"] = pd.Categorical(d["side"], categories=["Other", "Opposition", "Republican"], ordered=True)
    d = d.sort_values(["source", "side"], ascending=[True, False])
    d["end"] = d.groupby("source", observed=True)["share"].cumsum()
    d["mid"] = d["end"] - d["share"] / 2
    d["label"] = np.where(d["share"] >= 0.08, d["side"].astype(str) + " " + (100 * d["share"]).round().astype(int).astype(str) + "%", "")
    return (
        ggplot(d, aes("source", "share", fill="side"))
        + geom_col(width=0.6, position="stack")
        + geom_text(aes(y="mid", label="label"), color="white", size=9, family=FONT)
        + scale_fill_manual({"Republican": REP, "Opposition": OPP, "Other": GREY})
        + scale_y_continuous(labels=pct, expand=(0, 0))
        + labs(x="", y="Share of answers (weighted)")
        + theme_news(7, 2.8, legend_position="none", panel_grid_major_y=element_blank(),
                     panel_grid_major_x=element_line(color=GRID, size=0.5))
        + coord_flip()
    )


#### FIG G: silicon sampling versus regression ####
def compare_with_regression(summary: pl.DataFrame, root: Path = ROOT) -> pd.DataFrame:
    d = root / "data" / "02-analysis_data"
    sel = pl.read_parquet(d / "model_selection.parquet").filter(pl.col("selected"), pl.col("phase") == "validation").row(0, named=True)
    reg = pl.read_parquet(d / "regression_forecast_2026.parquet").filter(pl.col("features") == sel["features"], pl.col("train") == "all")
    reg = reg.select(state="st", regression=pl.col("p_rep") / (pl.col("p_dem") + pl.col("p_rep")))
    j = summary.select("state", silicon="median", p_rep_win="p_rep_win").join(reg, on="state").to_pandas()
    j["name"] = j["state"].map(STATE_NAMES)
    j["call"] = np.select(
        [(j["silicon"] > 0.5) & (j["regression"] > 0.5), (j["silicon"] < 0.5) & (j["regression"] < 0.5)],
        ["Both: Republican", "Both: opposition"], "Methods disagree",
    )
    j.attrs["features"] = sel["features"]
    return j


def fig_vs_regression(cmp: pd.DataFrame):
    dis = cmp[cmp["call"] == "Methods disagree"]
    return (
        ggplot(cmp, aes("silicon", "regression", color="call", fill="call"))
        + geom_vline(xintercept=0.5, color=GREY, linetype="dashed", size=0.5)
        + geom_hline(yintercept=0.5, color=GREY, linetype="dashed", size=0.5)
        + geom_point(size=2.8, stroke=0.9)
        + geom_text(aes(label="name"), data=dis, color=INK, size=8.5, family=FONT, ha="left", nudge_x=0.007)
        + scale_color_manual({"Both: Republican": REP, "Both: opposition": OPP, "Methods disagree": INK})
        + scale_fill_manual({"Both: Republican": REP, "Both: opposition": OPP, "Methods disagree": "white"})
        + scale_x_continuous(labels=pct)
        + scale_y_continuous(labels=pct)
        + labs(x="Republican share, silicon sampling", y="Republican share, regression")
        + theme_news(6.5, 5.0, panel_grid_major_x=element_line(color=GRID, size=0.5))
    )


#### FIG H: do the models agree? ####
def fig_models(summary: pl.DataFrame):
    d = summary.select("state", *MODEL_ORDER).to_pandas().melt(id_vars="state", var_name="model", value_name="share")
    d["name"] = pd.Categorical(d["state"].map(STATE_NAMES), categories=state_order(summary), ordered=True)
    d["model"] = pd.Categorical(d["model"], categories=MODEL_ORDER, ordered=True)
    return (
        ggplot(d, aes("share", "name", shape="model", color="model", fill="model"))
        + geom_vline(xintercept=0.5, color=GREY, linetype="dashed", size=0.5)
        + geom_point(size=2.2, stroke=0.7)
        + scale_shape_manual({"GPT-4o mini": "o", "Llama 3.3 70B": "s", "DeepSeek V4 Flash": "^"})
        + scale_color_manual({"GPT-4o mini": INK, "Llama 3.3 70B": GREY, "DeepSeek V4 Flash": INK})
        + scale_fill_manual({"GPT-4o mini": INK, "Llama 3.3 70B": GREY, "DeepSeek V4 Flash": "white"})
        + scale_x_continuous(labels=pct)
        + labs(x="Republican share from each model's synthetic voters (before calibration)", y="")
        + theme_news(7, 9.5, panel_grid_major_y=element_line(color="#F2F2F2", size=0.4))
    )


#### Tables (booktabs, styled in styles.scss) ####
def booktabs(df: pd.DataFrame, numeric: list[str]) -> str:
    head = "".join(f'<th class="{"num" if c in numeric else ""}">{c}</th>' for c in df.columns)
    body = "".join(
        "<tr>" + "".join(f'<td class="{"num" if c in numeric else ""}">{v}</td>' for c, v in zip(df.columns, row)) + "</tr>"
        for row in df.itertuples(index=False)
    )
    return f'<table class="booktabs"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


RUN_LABEL = {
    "responses_2024": "2024 backtest",
    "responses_2024_forced": "2024 forced choice (pure independents)",
    "responses_2026": "2026 forecast (first run)",
    "responses_2026_deepseek": "2026 DeepSeek rerun, reasoning off",
    "responses_2026_forced": "2026 forced choice (pure independents)",
}


def cost_table(resp: dict) -> pd.DataFrame:
    rows = [{"Run": RUN_LABEL[n], "Models": df["model"].n_unique(), "replies": df.height, "cost": df["cost_usd"].sum()}
            for n, df in resp.items()]  # fmt: skip
    out = pd.DataFrame(rows)
    out.loc[len(out)] = {"Run": "Total", "Models": "", "replies": out["replies"].sum(), "cost": out["cost"].sum()}
    out["Replies"] = out["replies"].map("{:,.0f}".format)
    out["Cost (US$)"] = out["cost"].map("{:,.2f}".format)
    out.attrs["total_cost"] = float(out["cost"].iloc[-1])
    return out[["Run", "Models", "Replies", "Cost (US$)"]]


def share_text(p: float) -> str:
    """Whole-number probability that never claims certainty (Gelman et al. 2020)."""
    return ">99%" if p > 0.995 else "<1%" if p < 0.005 else f"{100 * p:.0f}%"


def forecast_table(summary: pl.DataFrame, races_2026: pl.DataFrame) -> pd.DataFrame:
    r = {row["state"]: row for row in races_2026.iter_rows(named=True)}
    rows = []
    for s in summary.sort("p_rep_win", descending=True).iter_rows(named=True):
        race = r[s["state"]]
        indep = race["main_opponent"] == "Independent"
        opp = f"{race['independent']} (I)" if indep else f"{race['democrat']} (D)"
        rows.append({
            "State": STATE_NAMES[s["state"]],
            "Republican": race["republican"],
            "Main opponent": opp,
            "Median R share": f"{100 * s['median']:.0f}%",
            "80% interval": f"{100 * s['lower_80']:.0f}–{100 * s['upper_80']:.0f}%",
            "P(Republican wins)": share_text(s["p_rep_win"]),
        })  # fmt: skip
    return pd.DataFrame(rows)


#### Accuracy on 2024: silicon sampling versus regression ####
def accuracy_table(bt: pl.DataFrame, root: Path = ROOT) -> pd.DataFrame:
    """Out-of-sample 2024 accuracy. Silicon rows are the raw synthetic estimates (no
    model was fitted to 2024); the regression row is its untouched 2024 holdout."""
    rows = []
    for m, g in bt.group_by("model", maintain_order=True):
        rows.append({"Method": f"Silicon sampling, {m[0]} (raw)", "mae": 100 * g["error"].abs().mean(),
                     "right": int(g["winner_right"].sum()), "n": g.height})  # fmt: skip
    sel = (
        pl.read_parquet(root / "data" / "02-analysis_data" / "model_selection.parquet")
        .filter(pl.col("selected"), pl.col("phase") == "holdout").row(0, named=True)
    )
    rows.append({"Method": "Regression (2024 held out)", "mae": sel["official_mae_dem2p"],
                 "right": round(sel["official_winner_acc"] * sel["n"]), "n": sel["n"]})  # fmt: skip
    out = pd.DataFrame(rows).sort_values("Method")
    out["Average miss (points)"] = out["mae"].map("{:.1f}".format)
    out["Winners called"] = out["right"].astype(str) + " of " + out["n"].astype(str)
    return out[["Method", "Average miss (points)", "Winners called"]]
