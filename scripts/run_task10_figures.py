"""
Task 10: Generate all 17 figures and 6 tables for the paper.
Outputs: outputs/figures/fig*.pdf, outputs/tables/table*.tex
"""
import logging
from pathlib import Path

import pandas as pd

from src.data.loader import load_config
from src.visualization import (
    apply_paper_style,
    plot_passing_networks, plot_heterogeneous_schema, plot_event_flow_diagram,
    plot_triadic_motifs, plot_higher_order_motifs, plot_heterogeneous_motif_examples,
    plot_pitch_zones, plot_motif_vocabulary_overview,
    plot_decay_curve, plot_empirical_distribution, plot_saturation_curve,
    plot_zscore_heatmap, plot_spatial_heatmap,
    plot_tactical_fingerprint_radar,
    plot_temporal_evolution, plot_score_state_shift,
    plot_attribution_scatter,
    plot_hetero_errorbar, plot_motif_frequency_comparison,
    plot_w0_robustness,
    export_all_tables,
)
from src.networks.homogeneous import load_network

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def safe_plot(name: str, func, *args, **kwargs):
    """Wrapper that catches errors and logs them without stopping the pipeline."""
    try:
        fig = func(*args, **kwargs)
        logger.info(f"  ✓ {name}")
        return fig
    except Exception as e:
        logger.warning(f"  ✗ {name}: {e}")
        return None


def main():
    cfg = load_config("config.yaml")
    processed_dir = Path(cfg["data"]["processed_dir"])
    motifs_dir = Path(cfg["data"]["motifs_dir"])
    analysis_dir = Path(cfg["data"]["analysis_dir"])
    gnn_dir = Path(cfg["data"]["gnn_dir"])
    net_dir = Path(cfg["data"]["networks_dir"])
    fig_dir = Path("outputs/figures")
    tab_dir = Path("outputs/tables")
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    apply_paper_style()

    # ---- Load data ----
    logger.info("Loading data for figures...")
    match_meta_df = pd.read_parquet(processed_dir / "matches_meta.parquet")
    pass_df = pd.read_parquet(processed_dir / "events_pass.parquet")

    homo_motif_df = pd.read_parquet(motifs_dir / "homogeneous_motifs.parquet") \
        if (motifs_dir / "homogeneous_motifs.parquet").exists() else pd.DataFrame()

    zscore_df = pd.read_parquet(motifs_dir / "homogeneous_zscore.parquet") \
        if (motifs_dir / "homogeneous_zscore.parquet").exists() else pd.DataFrame()

    hetero_motif_df = pd.read_parquet(motifs_dir / "heterogeneous_motifs.parquet") \
        if (motifs_dir / "heterogeneous_motifs.parquet").exists() else pd.DataFrame()

    hetero_zscore_df = pd.read_parquet(motifs_dir / "heterogeneous_zscore.parquet") \
        if (motifs_dir / "heterogeneous_zscore.parquet").exists() else pd.DataFrame()

    regression_df = pd.read_parquet(analysis_dir / "regression_results.parquet") \
        if (analysis_dir / "regression_results.parquet").exists() else pd.DataFrame()

    spatial_df = pd.read_parquet(analysis_dir / "motif_spatial.parquet") \
        if (analysis_dir / "motif_spatial.parquet").exists() else pd.DataFrame()

    attribution_df = pd.read_parquet(gnn_dir / "motif_attribution.parquet") \
        if (gnn_dir / "motif_attribution.parquet").exists() else pd.DataFrame()

    incr_r2_df = pd.read_parquet(analysis_dir / "incremental_r2.parquet") \
        if (analysis_dir / "incremental_r2.parquet").exists() else pd.DataFrame()

    from src.analysis.saturation import compute_decay_curve, compute_hetero_decay_curve
    homo_decay = compute_decay_curve(homo_motif_df) if not homo_motif_df.empty else pd.DataFrame()
    hetero_decay = compute_hetero_decay_curve(hetero_motif_df) if not hetero_motif_df.empty else pd.DataFrame()

    # ---- Select example match ----
    example_match = match_meta_df.iloc[0]
    example_match_id = int(example_match["match_id"])
    home_team_id = int(example_match["home_team_id"])
    away_team_id = int(example_match["away_team_id"])
    home_name = example_match["home_team_name"]
    away_name = example_match["away_team_name"]

    logger.info(f"Example match: {home_name} vs {away_name} (ID={example_match_id})")

    # Try loading example networks
    G_home, G_away = None, None
    home_net_path = net_dir / "homogeneous" / f"{example_match_id}_home.gpickle"
    away_net_path = net_dir / "homogeneous" / f"{example_match_id}_away.gpickle"
    if home_net_path.exists():
        G_home = load_network(str(home_net_path))
    if away_net_path.exists():
        G_away = load_network(str(away_net_path))

    logger.info("Generating figures...")

    # Fig. 1: Passing networks
    if G_home and G_away:
        safe_plot("Fig.1", plot_passing_networks, G_home, G_away,
                  home_name, away_name,
                  output_path=str(fig_dir / "fig01_passing_networks.pdf"))

    # Fig. 2: Heterogeneous schema
    safe_plot("Fig.2", plot_heterogeneous_schema,
              output_path=str(fig_dir / "fig02_heterogeneous_schema.pdf"))

    # Fig. 3: Pitch zones
    safe_plot("Fig.3", plot_pitch_zones,
              output_path=str(fig_dir / "fig03_pitch_zones.pdf"))

    # Fig. 4: Event flow diagram
    safe_plot("Fig.4", plot_event_flow_diagram,
              output_path=str(fig_dir / "fig04_event_flow.pdf"))

    # Fig. 5: Decay curve
    if not homo_decay.empty:
        safe_plot("Fig.5", plot_decay_curve, homo_decay,
                  hetero_decay if not hetero_decay.empty else None,
                  output_path=str(fig_dir / "fig05_decay_curve.pdf"))

    # Fig. 6: Complete motif vocabulary (all k=2 and k=3 classes) + census
    #         statistics panel for k=2..7, single-column
    import json
    _summary_path = analysis_dir / "allorder_summary.json"
    _sizes_path   = analysis_dir / "census_sizes_allk.parquet"
    _census_stats = None
    if _summary_path.exists() and _sizes_path.exists():
        import pandas as _pd
        _sz = _pd.read_parquet(_sizes_path)
        _sz_mean = _sz.groupby(["k", "team_side"])["n_instances"].mean().unstack()
        _size_d  = {k: {"home": float(_sz_mean.loc[k, "home"]),
                        "away": float(_sz_mean.loc[k, "away"])}
                    for k in _sz_mean.index if k <= 7}
        _summ = json.load(open(_summary_path))
        _census_stats = {"size": _size_d, "summary": _summ["table"]}
    safe_plot("Fig.6", plot_motif_vocabulary_overview,
              census_stats=_census_stats,
              output_path=str(fig_dir / "fig06_triadic_motifs.pdf"))

    # Fig. 7: Higher-order motifs
    if not homo_motif_df.empty:
        k_vals = [k for k in sorted(homo_motif_df["motif_order_k"].unique()) if k >= 4]
        if k_vals:
            safe_plot("Fig.7", plot_higher_order_motifs, homo_motif_df, k_vals[:3],
                      output_path=str(fig_dir / "fig07_higher_order_motifs.pdf"))

    # Fig. 8: Heterogeneous motif examples
    safe_plot("Fig.8", plot_heterogeneous_motif_examples,
              output_path=str(fig_dir / "fig08_heterogeneous_motifs.pdf"))

    # Fig. 9: robustness of the network measures to the link-weight threshold w0
    netstats_for_fig = pd.read_parquet(processed_dir / "network_stats.parquet") \
        if (processed_dir / "network_stats.parquet").exists() else pd.DataFrame()
    if not netstats_for_fig.empty:
        safe_plot("Fig.9", plot_w0_robustness, netstats_for_fig,
                  output_path=str(fig_dir / "fig09_w0_robustness.pdf"))

    # Fig. 10: Information saturation (incremental R^2 vs motif order)
    if not incr_r2_df.empty:
        safe_plot("Fig.10", plot_saturation_curve, incr_r2_df,
                  output_path=str(fig_dir / "fig10_empirical_distribution.pdf"))
    elif not homo_decay.empty:
        safe_plot("Fig.10", plot_empirical_distribution, homo_decay,
                  output_path=str(fig_dir / "fig10_empirical_distribution.pdf"))

    # Fig. 11: z-score heatmap
    if not zscore_df.empty:
        safe_plot("Fig.11", plot_zscore_heatmap, zscore_df,
                  output_path=str(fig_dir / "fig11_zscore_heatmap.pdf"))

    # Fig. 12: Heterogeneous motif errorbar
    if not hetero_zscore_df.empty:
        safe_plot("Fig.12", plot_hetero_errorbar, hetero_zscore_df,
                  output_path=str(fig_dir / "fig12_hetero_errorbar.pdf"))

    # Fig. 13: Spatial heatmap
    if not spatial_df.empty:
        safe_plot("Fig.13", plot_spatial_heatmap, spatial_df,
                  output_path=str(fig_dir / "fig13_spatial_heatmap.pdf"))

    # Fig. 14: Score state shift
    score_state_path = analysis_dir / "motif_score_state.parquet"
    if score_state_path.exists():
        score_state_df = pd.read_parquet(score_state_path)
        safe_plot("Fig.14", plot_score_state_shift, score_state_df,
                  output_path=str(fig_dir / "fig14_context_shift.pdf"))

    # Fig. 15: Tactical fingerprint radar
    if not homo_motif_df.empty:
        # Build team-match lookup from match_meta
        teams = match_meta_df["home_team_name"].value_counts().head(5).index.tolist()
        team_match_ids = {}
        for team in teams:
            home_matches = match_meta_df[match_meta_df["home_team_name"] == team]["match_id"].tolist()
            away_matches = match_meta_df[match_meta_df["away_team_name"] == team]["match_id"].tolist()
            team_match_ids[team] = home_matches + away_matches

        safe_plot("Fig.15", plot_tactical_fingerprint_radar,
                  homo_motif_df, teams, team_match_ids,
                  output_path=str(fig_dir / "fig15_tactical_radar.pdf"))

    # Fig. 16: Temporal evolution
    if not homo_motif_df.empty and not pass_df.empty:
        match_motifs = homo_motif_df[homo_motif_df["match_id"] == example_match_id]
        if not match_motifs.empty:
            safe_plot("Fig.16", plot_temporal_evolution,
                      example_match_id, homo_motif_df, pass_df,
                      home_team_id=home_team_id, away_team_id=away_team_id,
                      output_path=str(fig_dir / "fig16_temporal_evolution.pdf"))

    # Fig. 17: Attribution scatter
    if not attribution_df.empty:
        safe_plot("Fig.17", plot_attribution_scatter, attribution_df,
                  regression_df if not regression_df.empty else None,
                  output_path=str(fig_dir / "fig17_attribution_scatter.pdf"))

    # ---- Export tables ----
    logger.info("Exporting LaTeX tables...")
    quality_path = processed_dir / "quality_report.parquet"
    stats_df = pd.read_parquet(quality_path) if quality_path.exists() else pd.DataFrame()

    network_stats_path = processed_dir / "network_stats.parquet"
    network_stats_df = pd.read_parquet(network_stats_path) if network_stats_path.exists() else pd.DataFrame()

    order_summary_path = motifs_dir / "motif_order_summary.parquet"
    order_summary_df = pd.read_parquet(order_summary_path) if order_summary_path.exists() else pd.DataFrame()

    from src.motifs.taxonomy import build_table6_semantic_df
    semantic_df = build_table6_semantic_df(zscore_df, regression_df)

    try:
        export_all_tables(
            stats_df=stats_df,
            network_stats_df=network_stats_df,
            order_summary_df=order_summary_df,
            zscore_df=zscore_df,
            regression_df=regression_df,
            semantic_df=semantic_df,
            output_dir=str(tab_dir),
        )
        logger.info("Tables exported.")
    except Exception as e:
        logger.warning(f"Table export failed: {e}")

    logger.info(f"Task 10 complete. Figures in {fig_dir}, tables in {tab_dir}")


if __name__ == "__main__":
    main()
