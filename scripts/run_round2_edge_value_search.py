"""Second-stage, validation-only encoder candidates with matched controls.

Run only after stage one's selection file is frozen. The base optimisation
setting is inherited from SPARC's mean validation-loss winner, not test data.
All six candidates share that setting. This is not a SOTA evaluation.
"""
from pathlib import Path
import hashlib
import json
import sys
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_round2_validation_search as runner
from src.gnn.mcar_model import MCARFootballGNN
from src.gnn.edge_value_candidate import add_edge_values
from src.gnn.anchor_readout_candidate import AnchorReadoutCandidate


def main():
    stage1=runner.OUT
    decision=stage1/"frozen_validation_selection.json"
    if not decision.exists():
        raise RuntimeError("Finish and freeze stage-one validation selection first")
    selected=json.loads(decision.read_text())
    row=next(r for r in selected["selections"] if r["model"]=="SPARC-GNN")
    cfg=dict(next(c for c in runner.GRID if c["name"]==row["config"]))
    cfg["parent_setting"]=cfg["name"]
    cfg["name"]="frozen_stage1"
    cfg["stage1_selection_sha256"]=hashlib.sha256(decision.read_bytes()).hexdigest()
    cfg["candidate_source_sha256"]={
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),ROOT/"src/gnn/edge_value_candidate.py",
         ROOT/"src/gnn/anchor_readout_candidate.py"]}

    def factory(name, conf):
        decomp=not name.startswith("FixedFusion")
        factory_class=AnchorReadoutCandidate if "AnchorReadout" in name else MCARFootballGNN
        model=factory_class(hidden_dim=conf["hidden"],num_layers=conf["layers"],
            dropout=conf["dropout"],use_info_decomposition=decomp,
            adaptive_gate=decomp, use_motif_context=name!="SPARC-EdgeValue-NoMotif" and decomp,
            use_assist_flags=False,mask_unavailable_relations=True)
        if conf["norm"]=="batch":
            for layer in model.norms:
                for nt in model.NODE_TYPES:
                    layer[nt]=nn.BatchNorm1d(conf["hidden"])
        return add_edge_values(model) if "EdgeValue" in name else model

    runner.OUT=ROOT/"data/information_sciences/round2_edge_value_search"
    runner.MODELS=["SPARC-Control","SPARC-EdgeValue","FixedFusion-EdgeValue",
                   "SPARC-EdgeValue-NoMotif","SPARC-AnchorReadout","FixedFusion-AnchorReadout"]
    runner.GRID=[cfg]
    runner.make_model=factory
    runner.SPARC_REVISION="experimental_v3_plus_edge_values_v1"
    runner.main()


if __name__=="__main__":
    main()
