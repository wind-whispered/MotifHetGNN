"""
LaTeX export utilities for the manuscript tables.

Each function writes a self-contained ``table`` float (booktabs) with a caption
and a label that matches the \\ref{} keys used in main.tex. The tabular body is
produced with ``escape=False`` so that math in the headers (e.g. ``$k$``,
``$z$``) renders correctly; the data columns contain only numbers and simple
ASCII labels, so no escaping is required.
"""
from typing import List, Optional
import pandas as pd


def _wrap_table(tabular: str, caption: str, label: str,
                star: bool = False, small: bool = True) -> str:
    env = "table*" if star else "table"
    size = "\\footnotesize\n" if small else ""
    return (
        f"\\begin{{{env}}}[!tb]\n\\centering\n"
        f"{size}"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        + tabular.strip() + "\n"
        f"\\end{{{env}}}\n"
    )


def _escape_cell(v):
    """Escape LaTeX-special characters in a string cell, leaving intentional
    math (cells containing '$') untouched. Column headers are not passed here."""
    if not isinstance(v, str):
        return v
    if "$" in v:           # already contains intentional math, leave as-is
        return v
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
                 ("#", r"\#"), ("%", r"\%")):
        v = v.replace(a, b)
    return v


def _tabular(df: pd.DataFrame, column_format: Optional[str] = None,
             float_format: str = "{:.3f}") -> str:
    df = df.copy()
    for c in df.columns:
        # pandas >= 3.0 uses StringDtype (not object) for text columns, so guard
        # on "not numeric" to make sure every text cell is escaped.
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(object).map(_escape_cell)
    n_cols = len(df.columns)
    if column_format is None:
        column_format = "l" + "r" * (n_cols - 1)
    return df.to_latex(
        index=False,
        column_format=column_format,
        float_format=float_format.format,
        escape=False,
        longtable=False,
    )


def df_to_latex(df, caption, label, float_format="{:.3f}",
                column_format=None, output_path=None,
                star=False, small=True) -> str:
    out = _wrap_table(_tabular(df, column_format, float_format), caption, label,
                      star=star, small=small)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(out)
    return out


def export_table1(stats_df: pd.DataFrame, output_path: Optional[str] = None) -> str:
    """Table 1: dataset overview."""
    label_map = {
        "n_matches_total": "Matches",
        "n_pass_events": "Successful passes",
        "n_unique_passers": "Distinct players",
        "pct_under_pressure": "Passes under pressure (%)",
        "pct_counterpress": "Counterpress passes (%)",
        "n_adversarial_total": "Adversarial/turnover events",
        "n_Interception": "Interceptions",
        "n_BallRecovery": "Ball recoveries",
        "n_Miscontrol": "Miscontrols",
        "n_Dispossessed": "Dispossessions",
    }
    rows = []
    for key, name in label_map.items():
        if key in stats_df.columns:
            val = stats_df[key].iloc[0]
            if key.startswith("pct"):
                rows.append((name, f"{val:.2f}"))
            else:
                rows.append((name, f"{int(val):,}"))
    df = pd.DataFrame(rows, columns=["Quantity", "Value"])
    return df_to_latex(
        df, caption="Overview of the StatsBomb dataset used in this study.",
        label="tab:dataset", column_format="lr", output_path=output_path,
    )


def export_table2(network_stats_df: pd.DataFrame, output_path: Optional[str] = None) -> str:
    """Table 2: average passing-network metrics for home/away at each threshold w0."""
    metrics = ["density", "transitivity", "pass_diversity",
               "mean_outdegree", "mean_betweenness", "mean_eigenvector"]
    metrics = [m for m in metrics if m in network_stats_df.columns]
    header = {"density": "$D$", "transitivity": "$T$", "pass_diversity": "$\\phi$",
              "mean_outdegree": "$k_{\\mathrm{out}}$", "mean_betweenness": "$b$",
              "mean_eigenvector": "$e$"}
    rows = []
    for w0 in sorted(network_stats_df["w0"].unique()):
        for side in ["home", "away"]:
            sub = network_stats_df[(network_stats_df["w0"] == w0) &
                                   (network_stats_df["team_side"] == side)]
            if sub.empty:
                continue
            row = {"$w_0$": int(w0), "Side": side}
            for m in metrics:
                row[header[m]] = f"{sub[m].mean():.3f}"
            rows.append(row)
    df = pd.DataFrame(rows)
    return df_to_latex(
        df, caption="Average topological measures of the passing networks for "
                    "home and away teams at link-weight thresholds $w_0$: density "
                    "$D$, transitivity $T$, passing diversity $\\phi$, mean "
                    "out-degree $k_{\\mathrm{out}}$, betweenness $b$ and "
                    "eigenvector centrality $e$.",
        label="tab:netstats", column_format="ll" + "r" * len(metrics),
        output_path=output_path, star=True,
    )


def export_table3(order_summary_df: pd.DataFrame, output_path: Optional[str] = None) -> str:
    """Table 3: observed motif counts per order k."""
    cols = ["motif_order_k", "team_side", "theoretical_types",
            "observed_types", "mean_total_count", "std_total_count"]
    cols = [c for c in cols if c in order_summary_df.columns]
    df = order_summary_df[cols].copy()
    df["mean_total_count"] = df["mean_total_count"].map(lambda v: f"{v:.1f}")
    df["std_total_count"] = df["std_total_count"].map(lambda v: f"{v:.1f}")
    df.columns = ["$k$", "Side", "Theoretical classes", "Observed classes",
                  "Mean census", "Std census"][:len(cols)]
    return df_to_latex(
        df, caption="Number of distinct directed motif classes and the mean motif "
                    "census per network across orders $k$.",
        label="tab:motif_counts", column_format="llrrrr", output_path=output_path,
    )


def export_table4(zscore_df: pd.DataFrame, output_path: Optional[str] = None) -> str:
    """Table 4: triadic-motif z-scores (sorted by home z), reproducing the
    reference-style significance table."""
    z = zscore_df[zscore_df.get("motif_order_k", 3) == 3].copy()
    # one row per motif id with home and away z side-by-side
    piv = z.pivot_table(index="motif_id", columns="team_side",
                        values=["mu", "mu_rnd", "z"], aggfunc="first")
    rows = []
    for mid in piv.index:
        def g(col, side):
            try:
                return piv.loc[mid, (col, side)]
            except Exception:
                return float("nan")
        rows.append({
            "id": int(mid),
            "$\\mu_{\\mathrm{H}}$": f"{g('mu','home'):.2f}",
            "$\\mu^{\\mathrm{rnd}}_{\\mathrm{H}}$": f"{g('mu_rnd','home'):.2f}",
            "$z_{\\mathrm{H}}$": f"{g('z','home'):.2f}",
            "$\\mu_{\\mathrm{A}}$": f"{g('mu','away'):.2f}",
            "$\\mu^{\\mathrm{rnd}}_{\\mathrm{A}}$": f"{g('mu_rnd','away'):.2f}",
            "$z_{\\mathrm{A}}$": f"{g('z','away'):.2f}",
            "_sort": g('z', 'home'),
        })
    df = pd.DataFrame(rows).sort_values("_sort").drop(columns="_sort")
    return df_to_latex(
        df, caption="Significance of the 13 triadic motifs at $w_0=2$, ordered by "
                    "the home-network $z$-score. $\\mu$ and $\\mu^{\\mathrm{rnd}}$ "
                    "are the mean counts in the empirical and degree-preserving "
                    "random networks; all deviations are significant by a paired "
                    "$t$-test ($p<0.001$).",
        label="tab:zscore", column_format="lrrrrrr", output_path=output_path,
        star=True,
    )


def export_table5(regression_df: pd.DataFrame, output_path: Optional[str] = None) -> str:
    """Table 5: significant OLS coefficients (goal difference vs motif counts)."""
    df = regression_df.copy()
    # focus on the main triadic panel if a 'panel' column exists
    if "panel" in df.columns:
        panels = df["panel"].unique()
        main = [p for p in panels if "k3" in str(p) or "A" in str(p)]
        if main:
            df = df[df["panel"] == main[0]]
    keep = df.get("significant", pd.Series(True, index=df.index)).fillna(False)
    df = df[keep]
    cols = [c for c in ["variable", "beta", "std_error", "t_stat", "p_value"] if c in df.columns]
    df = df[cols].copy()
    for c in ["beta", "std_error", "t_stat"]:
        if c in df.columns:
            df[c] = df[c].map(lambda v: f"{v:.3f}")
    if "p_value" in df.columns:
        df["p_value"] = df["p_value"].map(lambda v: "$<$0.001" if v < 1e-3 else f"{v:.3f}")
    df.columns = ["Predictor", "$\\beta$", "Std.\\ error", "$t$", "$p$"][:len(cols)]
    return df_to_latex(
        df, caption="Significant ordinary-least-squares coefficients relating the "
                    "home goal difference to triadic-motif counts (Eq.~\\eqref{eq:ols}, "
                    "$w_0=2$). Predictors are labelled by motif id and team side.",
        label="tab:reg", column_format="lrrrr", output_path=output_path,
    )


def export_table6(semantic_df: pd.DataFrame, output_path: Optional[str] = None) -> str:
    """Table 6: tactical semantics of key heterogeneous motifs."""
    cols = ["tactical_label", "description", "goal_diff_relation"]
    cols = [c for c in cols if c in semantic_df.columns]
    df = semantic_df[cols].copy()
    relation_label = {
        "positive_home": "raises home goal difference",
        "negative_home": "lowers home goal difference",
        "positive_away": "raises away goal difference",
        "negative_opponent": "favours the defending team",
        "positive": "raises goal difference",
        "negative": "lowers goal difference",
    }
    if "goal_diff_relation" in df.columns:
        df["goal_diff_relation"] = df["goal_diff_relation"].map(
            lambda v: relation_label.get(str(v), str(v))
        )
    df.columns = ["Motif", "Tactical interpretation", "Outcome relation"][:len(cols)]
    return df_to_latex(
        df, caption="Tactical interpretation of the interpretable cooperative and "
                    "heterogeneous (cooperative--adversarial) motifs.",
        label="tab:semantics", column_format="lp{10.5cm}l", output_path=output_path,
        star=True,
    )


def export_all_tables(
    stats_df: pd.DataFrame,
    network_stats_df: pd.DataFrame,
    order_summary_df: pd.DataFrame,
    zscore_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    semantic_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Export all manuscript tables to LaTeX files."""
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not stats_df.empty:
        export_table1(stats_df, str(out / "table1_dataset_stats.tex"))
    if not network_stats_df.empty:
        export_table2(network_stats_df, str(out / "table2_netstats.tex"))
    if not order_summary_df.empty:
        export_table3(order_summary_df, str(out / "table3_motif_counts.tex"))
    if not zscore_df.empty:
        export_table4(zscore_df, str(out / "table4_zscore_summary.tex"))
    if not regression_df.empty:
        export_table5(regression_df, str(out / "table5_regression.tex"))
    if semantic_df is not None and not semantic_df.empty:
        export_table6(semantic_df, str(out / "table6_tactical_semantics.tex"))
