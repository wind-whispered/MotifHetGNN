"""Supplementary reviewer evidence from frozen results; no fitting or selection.

All input hashes are checked after plotting. The finite-graph examples are
analytic illustrations, explicitly separate from the empirical null results.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from math import comb
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from src.paths import figure_directory

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/information_sciences"
FIG = figure_directory()
AUDIT = ROOT / "outputs/audits"
FIG.mkdir(parents=True, exist_ok=True)
AUDIT.mkdir(parents=True, exist_ok=True)
HASHES = {}
GENERATED = []
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#70569B"]
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False})


def read(name):
    path = DATA / name
    HASHES[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)


def save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(FIG / f"{name}.{ext}", dpi=400, bbox_inches="tight", pad_inches=.035)
    GENERATED.append(name)
    plt.close(fig)


def finite_graph_examples():
    n = 14
    k = np.arange(2, n+1)
    capacity = np.array([comb(n, int(j)) for j in k])
    series = {"Clique": capacity,
              "Star": np.array([comb(n-1, int(j)-1) for j in k]),
              "Path": n-k+1}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.35), layout="constrained")
    for (label, counts), color, marker in zip(series.items(), PALETTE, ["o", "s", "^"]):
        axes[0].plot(k, counts, marker+"-", lw=1.2, ms=3, color=color, label=label)
        axes[1].plot(k, counts/capacity, marker+"-", lw=1.2, ms=3, color=color, label=label)
    axes[0].set(yscale="log", ylabel=r"Connected count $C_k$", title="(a) Raw finite-size profile")
    axes[1].set(ylabel=r"Occupancy $Q_k=C_k/\binom{n}{k}$", ylim=(-.04,1.07),
                title="(b) Capacity-normalised profile")
    for ax in axes:
        ax.set_xlabel(r"Subgraph order $k$"); ax.set_xticks(range(2,15,2))
        ax.grid(axis="y", color=".91", lw=.5)
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, "Figure_14_Finite_Graph_Examples")


def null_constraints():
    weak = read("finite_size_rewiring_diagnostics.parquet")
    directed = read("finite_size_rewiring_diagnostics_directed_degree_diagnostic.parquet")
    controlled = read("finite_size_controlled.parquet")
    assert len(weak) == len(directed) == 20000
    assert (weak.rewired_weak_edges == weak.original_weak_edges).all()
    assert (weak.successful_swaps == weak.target_swaps).all()
    assert (directed.successful_swaps == directed.target_swaps).all()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), layout="constrained")
    for d, label, color in [(weak,"Weak-degree null",PALETTE[0]),
                            (directed,"Directed-degree diagnostic",PALETTE[1])]:
        delta = d.rewired_weak_edges - d.original_weak_edges
        freq = delta.value_counts(normalize=True).sort_index()
        axes[0].plot(freq.index, freq.values, "o-", ms=3, color=color, label=label)
        ratio = np.sort((d.successful_swaps / d.swap_attempts).to_numpy())
        axes[1].plot(ratio, np.arange(1,len(ratio)+1)/len(ratio), color=color, lw=1.2)
    axes[0].set(xlabel="Change in weak edge count", ylabel="Fraction of null graphs",
                title="(a) Density preservation")
    axes[1].set(xlabel="Accepted / attempted swaps", ylabel="Cumulative fraction",
                title="(b) Rewiring acceptance")
    q = controlled[controlled.k.between(3,7)].groupby("k").q_excess.agg(["mean","std"])
    axes[2].errorbar(q.index, q["mean"], yerr=q["std"], color=PALETTE[0],
                     fmt="o-", ms=3, capsize=2.5, lw=1.1)
    axes[2].axhline(0,color=".5",ls="--",lw=.7)
    axes[2].set(xlabel=r"Subgraph order $k$", ylabel=r"$Q_k-\mathbb{E}_{null}[Q_k]$",
                title="(c) Matched-null residual")
    axes[2].set_xticks(range(3,8))
    for ax in axes: ax.grid(axis="y",color=".92",lw=.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False,fontsize=7)
    save(fig,"Figure_15_Null_Model_Constraint_Checks")


def perturbation_controls():
    d = read("relation_control_paired_effects.csv")
    names = {"pass_availability":"Delete passing", "adversarial_availability":"Delete adversarial",
             "turnover_availability":"Delete turnover", "pass_topology_degree_preserving":"Rewire passing",
             "adversarial_anchor_identity":"Shift adversarial anchors",
             "turnover_anchor_identity":"Shift turnover anchors"}
    fig, axes = plt.subplots(1, 2, figsize=(7.1,2.75), sharey=True, layout="constrained")
    for ax, metric, title in zip(axes,["macro_f1","r2"],[r"(a) Macro-$F_1$",r"(b) $R^2$"]):
        for y, key in enumerate(names):
            row = d[(d.contrast==key)&(d.metric==metric)].iloc[0]
            points = np.array([float(p.split(":")[1]) for p in row.seedwise_improvements.split(";")])
            ax.scatter(points,y+np.linspace(-.11,.11,len(points)),s=12,
                       facecolors="white",edgecolors=PALETTE[0],linewidths=.7,zorder=3)
            ax.plot([row.ci_low,row.ci_high],[y,y],color="#34434D",lw=1.4)
            ax.plot(row.mean_improvement,y,"D",color="#34434D",ms=3.6,zorder=4)
        ax.axvline(0,color=".55",ls="--",lw=.8)
        ax.set_xlabel("Intact model minus perturbed model")
        ax.set_title(title); ax.grid(axis="x",color=".91",lw=.5)
    axes[0].set_yticks(range(len(names)), list(names.values()),fontsize=7.3)
    axes[0].invert_yaxis()
    save(fig,"Figure_16_Relation_Perturbation_Controls")


def confusion_panels():
    d = read("predictions.parquet")
    models = ["TeamPool-MLP", "HeteroSAGE-sum", "SPARC-GNN"]
    fig, axes = plt.subplots(1, 3, figsize=(7.1,2.45), layout="constrained")
    for index,(ax,model) in enumerate(zip(axes,models)):
        rows = d[d.model==model]
        matrices=[]
        for seed, group in rows.groupby("seed"):
            assert len(group)==692 and group.match_id.nunique()==692
            matrices.append(confusion_matrix(group.true_class,group.pred_class,
                                             labels=[0,1,2],normalize="true"))
        assert len(matrices)==5
        matrix=np.mean(matrices,axis=0)
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap="Blues",aspect="equal")
        ax.set_box_aspect(1)
        for y in range(3):
            for x in range(3):
                ax.text(x,y,f"{matrix[y,x]:.2f}",ha="center",va="center",fontsize=8,
                        color="white" if matrix[y,x]>.55 else "#202020")
        ax.set_xticks(range(3),["H","D","A"]); ax.set_yticks(range(3),["H","D","A"])
        ax.set_xlabel("Predicted outcome")
        ax.set_title(f"({chr(97+index)}) {model}")
    axes[0].set_ylabel("Observed outcome")
    fig.colorbar(im,ax=axes,shrink=.73,pad=.025,label="Row-normalised frequency")
    save(fig,"Figure_17_Confusion_Matrices")


def model_cost():
    d = read("model_complexity.parquet")
    d=d[d.model!="CurrentScore"].reset_index(drop=True)
    fig, axes=plt.subplots(1,2,figsize=(7.1,2.7),sharey=True,layout="constrained")
    y=np.arange(len(d))
    for ax,values,title,label in [(axes[0],d.parameters/1000,"(a) Model size","Trainable parameters (thousands)"),
                                 (axes[1],d.inference_ms_per_graph,"(b) Batched inference","Time per graph (ms)")]:
        for i,val in enumerate(values):
            color=PALETTE[1] if d.iloc[i].model=="SPARC-GNN" else PALETTE[0]
            ax.hlines(i,0,val,color=".83",lw=1)
            ax.plot(val,i,"o",color=color,ms=4)
            ax.annotate(f"{val:.1f}" if ax==axes[0] else f"{val:.3f}",
                        (val,i),xytext=(4,0),textcoords="offset points",va="center",fontsize=6.8)
        ax.set(xlim=(0,max(values)*1.25),xlabel=label,title=title)
        ax.grid(axis="x",color=".92",lw=.5)
    axes[0].set_yticks(y,[m.replace("MotifGate-ObservedOnly","MotifGate (observed)") for m in d.model],fontsize=7)
    axes[0].invert_yaxis()
    save(fig,"Figure_18_Computational_Cost")


def temporal_pathways():
    base=read("temporal_results.parquet")
    score=read("temporal_validation_blends.parquet")
    context=read("temporal_context_graph_blends.parquet")
    paths=[(base,"SPARC-GNN+Score","SPARC + score",PALETTE[2],"o"),
           (score,"SPARC-GNN+Score+ValBlend","+ score calibration",PALETTE[1],"s"),
           (context,"SPARC-GNN+Score+ContextBlend","+ causal-context calibration",PALETTE[0],"^"),
           (base,"CurrentScore","Current score","#787878","D")]
    fig,axes=plt.subplots(2,2,figsize=(7.1,4.2),layout="constrained")
    for data,model,label,color,marker in paths:
        data=data.assign(model=data.model.str.replace("MCAR-Decomp","SPARC-GNN",regex=False))
        sub=data[data.model==model]
        assert len(sub)>0, model
        means=sub.groupby("cutoff")[["macro_f1","draw_f1","mae","r2"]].mean()
        std=sub.groupby("cutoff")[["macro_f1","draw_f1","mae","r2"]].std().fillna(0)
        for ax,metric in zip(axes.ravel(),means.columns):
            ax.plot(means.index,means[metric],marker+"-",ms=3,lw=1.2,color=color,label=label)
            ax.fill_between(means.index,means[metric]-std[metric],means[metric]+std[metric],color=color,alpha=.10,lw=0)
    for ax,title in zip(axes.ravel(),[r"(a) Macro-$F_1$",r"(b) Draw-$F_1$","(c) MAE",r"(d) $R^2$"]):
        ax.set_title(title);ax.set_xlabel("Strict event cutoff (min)")
        ax.set_xticks([15,30,45,60,75]);ax.grid(axis="y",color=".91",lw=.5)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2,loc="outside lower center",frameon=False,fontsize=7)
    save(fig,"Figure_19_Temporal_Prediction_Pathways")


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    finite_graph_examples();null_constraints();perturbation_controls()
    confusion_panels();model_cost();temporal_pathways()
    assert all(hashlib.sha256((DATA/name).read_bytes()).hexdigest()==digest
               for name,digest in HASHES.items()), "An experimental input changed"
    (AUDIT / "supplementary_figure_provenance.json").write_text(json.dumps(
        {"input_sha256":HASHES,"generated_figures":GENERATED,"input_files_unchanged":True},indent=2),encoding="utf8")
    print("Generated",len(GENERATED),"figures;",len(HASHES),"input hashes unchanged.")


if __name__=="__main__":
    main()
