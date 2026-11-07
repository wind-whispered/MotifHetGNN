"""Editable scientific schematic; vectors and glyphs are illustrative.

A three-stage overview is paired with an expanded local fusion computation.
The pictured model is the frozen manuscript model, not the separate round-two
normalisation/optimisation candidates. No measured quantities are fabricated.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
from src.paths import figure_directory

FIG = figure_directory()
C = dict(pass_="#237FA1", adv="#C37748", turn="#288E79", shared="#75609D",
         ink="#203945", quiet="#60747E", rule="#CDD9DF", pale="#F8FAFB",
         bluepale="#EDF6FA", orangepale="#FCF2E8", greenpale="#EDF7F3",
         purplepale="#F1EDF7")


def draw():
    with plt.rc_context({"font.family": "Arial", "font.size": 7.3,
                         "mathtext.fontset": "dejavusans", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.grid": False}):
        fig = plt.figure(figsize=(180/25.4, 137/25.4), facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 180), ylim=(0, 137))
        ax.set_aspect("equal"); ax.axis("off")

        def txt(x, y, s, size=7.3, color=C["ink"], ha="left", weight="normal", **kw):
            return ax.text(x, y, s, fontsize=size, color=color, ha=ha,
                           va="center", fontweight=weight, **kw)

        def line(points, color=C["rule"], lw=.8, **kw):
            x, y = zip(*points); ax.plot(x, y, color=color, lw=lw, **kw)

        def arrow(points, color=C["ink"], lw=.85, dashed=False, scale=7):
            if len(points) > 2:
                line(points[:-1], color, lw, ls="--" if dashed else "-")
            ax.add_patch(FancyArrowPatch(points[-2], points[-1],
                arrowstyle="-|>", mutation_scale=scale, linewidth=lw,
                color=color, linestyle="--" if dashed else "-",
                shrinkA=0, shrinkB=0))

        def box(x, y, w, h, label="", color=C["rule"], fill="white", size=7.1,
                radius=1.6, lw=.75, weight="normal"):
            ax.add_patch(FancyBboxPatch((x, y), w, h,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=fill, edgecolor=color, linewidth=lw))
            if label:
                txt(x+w/2, y+h/2, label, size, ha="center", weight=weight)

        def vector(x, y, color, n=6, w=1.65, h=3.0, absent=False):
            for i in range(n):
                ax.add_patch(Rectangle((x+i*w,y),w-.25,h,
                    facecolor="white" if absent else color,
                    edgecolor=color, linewidth=.35,
                    alpha=1 if absent else [.35,.8,.5,1,.6,.4][i%6],
                    linestyle="--" if absent else "-"))

        def node(x,y,home=True,r=1.15,anchor=False,color=C["ink"]):
            shape = Circle((x,y),r) if home else Rectangle((x-r,y-r),2*r,2*r)
            shape.set(facecolor="white",edgecolor=color,linewidth=.85,zorder=5)
            ax.add_patch(shape)
            if anchor:
                ax.add_patch(Circle((x,y),.4,fc=color,ec="none",zorder=6))

        def graph(points, edges, color, mixed=False, radius=1.0):
            for u,v in edges:
                a,b = np.array(points[u]),np.array(points[v])
                d=b-a; unit=d/np.linalg.norm(d)
                arrow([a+unit*(radius+.2),b-unit*(radius+.55)],color,.7,scale=5)
            for i,(x,y) in enumerate(points):
                node(x,y,(not mixed) or i != 1,radius,color=color)

        def heading(x,y,letter,label,color=C["ink"]):
            ax.add_patch(Circle((x+2.3,y),2.3,fc=color,ec="none"))
            txt(x+2.3,y+.1,letter,8,"white",ha="center",weight="bold")
            txt(x+6,y,label,8.6,weight="bold")

        def card(x, y, w, h, fill, accent):
            # All depth cues remain vector geometry and survive greyscale print.
            box(x+.3,y-.45,w,h,fill="#E8EEF1",color="#E8EEF1",lw=0)
            box(x,y,w,h,fill=fill,lw=.7)
            line([(x+2,y+h-10),(x+w-2,y+h-10)],accent,.7,alpha=.45)

        def event_clock(x, y):
            ax.add_patch(Circle((x,y),1.45,fc="white",ec=C["quiet"],lw=.65))
            line([(x,y+1),(x,y),(x+.65,y-.35)],C["quiet"],.65)

        def opposition_glyph(x, y, color):
            # Two actors address one opposing-team anchor, not identified duel opponents.
            for yy in [y+2.1,y-2.1]:
                arrow([(x+.9,yy),(x+5.3,y)],color,.7,scale=4.7)
                node(x,yy,True,.75,color=color)
            node(x+6.5,y,False,.95,anchor=True,color=color)

        # Overview: three cards with one unambiguous left-to-right data path.
        card(3,77,42,57,C["bluepale"],C["pass_"])
        card(49,77,78,57,C["pale"],C["shared"])
        card(131,77,46,57,C["greenpale"],C["turn"])
        heading(5,129.5,"a","Observed events")
        heading(51,129.5,"b","Encode and fuse")
        heading(133,129.5,"c","Predict by team")
        event_clock(8,121)
        txt(11,121,"Event records",6.2)
        line([(7,117.5),(40,117.5)],C["rule"],1.2,zorder=0)
        for x,col,letter in [(8,C["pass_"],"P"),(13,C["adv"],"A"),
                             (19,C["pass_"],"P"),(24,C["turn"],"T")]:
            box(x-1.5,115.8,3,3.4,fill=col,color=col,radius=.45,lw=.4)
            txt(x,117.5,letter,5.4,"white",ha="center",weight="bold")
        for x in [33,37,40]:
            line([(x,116.3),(x,118.7)],"#CDD7DD",1)
        line([(28.5,115),(28.5,120)],C["ink"],.8,ls="--")
        txt(29.2,121,r"$\tau$",7.5,ha="center")
        txt(35.8,112.9,"future",5.7,C["quiet"],ha="center")
        arrow([(19,115),(19,111)],C["quiet"])

        # Pitch is a context glyph; connections encode actual graph semantics.
        box(7,88,34,23,fill="#F4F8F2",color="#98B6AA",radius=.6,lw=.65)
        for x in [7.3,15.5,23.7,31.9]:
            ax.add_patch(Rectangle((x,88.3),4.1,22.4,fc="#EAF2E9",ec="none",zorder=1))
        pitch_rule="#B1C8B9"
        line([(24,88.3),(24,110.7)],pitch_rule,.55)
        ax.add_patch(Circle((24,99.5),3.4,fc="none",ec=pitch_rule,lw=.55))
        ax.add_patch(Circle((24,99.5),.25,fc=pitch_rule,ec="none"))
        for x in [7.1,35.9]:
            ax.add_patch(Rectangle((x,93.5),5,12,fc="none",ec=pitch_rule,lw=.5))
        for x in [7.1,38.7]:
            ax.add_patch(Rectangle((x,96.2),2.2,6.6,fc="none",ec=pitch_rule,lw=.5))
        for x in [6.1,41]:
            ax.add_patch(Rectangle((x,97.4),.9,4.2,fc="white",ec="#98B6AA",lw=.45))
        pos={0:(11,102),1:(21,107),2:(21,97),3:(12,92),
             4:(32,107),5:(37,98),6:(30,91)}
        for u,v,col in [(0,1,"pass_"),(1,2,"pass_"),(2,0,"pass_"),
                        (2,3,"pass_"),(4,5,"pass_"),(5,6,"pass_"),
                        (4,1,"adv"),(2,5,"adv"),(3,5,"turn")]:
            a,b=np.array(pos[u]),np.array(pos[v]); d=b-a; unit=d/np.linalg.norm(d)
            arrow([a+unit*1.5,b-unit*1.75],C[col],.85,scale=5.5)
        for i,(x,y) in pos.items():
            node(x,y,i<4,1.2,i in [1,5])
        node(8,83.5,True,.85); txt(10,83.5,"Home",6.0)
        node(22,83.5,False,.85); txt(24,83.5,"Away",6.0)
        ax.add_patch(Circle((35,83.5),.4,fc=C["ink"],ec="none"))
        txt(37,83.5,"Anchor",5.6)
        arrow([(45,102),(48.2,102)],C["ink"],1.1)

        # Relation glyph -> relation-local edge-aware message -> fusion.
        for y,col,pale,label,away in [
                (116,C["pass_"],C["bluepale"],"Passing",False),
                (104,C["adv"],C["orangepale"],"Adversarial",True),
                (92,C["turn"],C["greenpale"],"Turnover",True)]:
            box(53,y-4.7,46,9.4,fill=pale,color=col,lw=.5)
            if away:
                opposition_glyph(56,y,col)
            else:
                graph([(56,y),(60,y+2.5),(63,y-2)],[(0,1),(1,2),(2,0)],col,False,.7)
            txt(66,y+1.8,label,6.5,col,weight="bold")
            txt(66,y-1.9,"Edge-aware encoder",5.8,C["quiet"])
            arrow([(99,y),(103,y)],col,.8)
            vector(104,y-1.5,col,n=5,w=1.3)
            arrow([(111,y),(114,y)],col,.8)
        box(114,86,9.5,36,fill=C["purplepale"],color=C["shared"],lw=.9)
        txt(118.7,106,"SPARC",7,C["shared"],ha="center",rotation=90,weight="bold")
        ax.add_patch(Circle((118.7,91),1.8,fc=C["shared"],ec="none"))
        txt(118.7,91,"d",6.6,"white",ha="center",weight="bold")
        txt(55,81.5,"Node + edge attributes",6.2,C["quiet"])
        txt(122,81.5,r"$\times\,3$ layers",6.3,C["quiet"],ha="right")
        arrow([(123.5,102),(130.3,102)],C["ink"],1.1)

        # Readout and the actual heads; no fake probability/performance chart.
        txt(154,122,"Mean pool by team",6.8,ha="center")
        vector(137,114,C["quiet"],w=1.65)
        vector(162,114,C["quiet"],w=1.65)
        txt(142,110.7,"Home",6.0,ha="center")
        txt(166.7,110.7,"Away",6.0,ha="center")
        arrow([(142,109),(142,106),(154,106),(154,103)],C["quiet"],.65)
        arrow([(167,109),(167,106),(154,106)],C["quiet"],.65)
        box(139,97,30,6,"Concatenate",size=6.8,lw=.65)
        arrow([(154,97),(154,94),(143,94),(143,91)],C["quiet"],.7)
        arrow([(154,94),(166,94),(166,91)],C["quiet"],.7)
        box(134,82,19,9,"Outcome\nH / D / A",color="#AEC9D3",size=6.4)
        box(156,82,19,9,"Goal\ndifference",color="#AEC9D3",size=6.4)

        # A dashed callout links the overview module to its detailed computation.
        arrow([(118.7,77),(118.7,74),(91,74),(91,71.5)],C["shared"],.7,True)

        # Core: shared bypass above a private lane and a separate conditioning lane.
        card(3,3,174,68,"#FDFCFF",C["shared"])
        heading(5,66,"d","SPARC fusion: common signal, relation-specific information",C["shared"])
        txt(6,57,"Observed messages",7.1,weight="bold")
        txt(48,57,"Shared / private decomposition",7.1,weight="bold")
        txt(137,57,"Residual update",7.1,weight="bold")

        for y,col,label,missing in [(47,C["pass_"],"P",False),
                                    (39,C["adv"],"A",False),
                                    (31,C["turn"],"T",True)]:
            txt(7,y+1.5,label,7,col,weight="bold")
            vector(12,y,col,absent=missing)
        txt(12,27.5,"absent",5.8,C["quiet"])
        # Incoming P/A meet at an observed-only mean.
        arrow([(22,48.5),(36,48.5)],C["pass_"],.8)
        arrow([(22,40.5),(29,40.5),(29,48.5),(36,48.5)],C["adv"],.8)
        box(36,44,22,9,"Observed mean",color=C["shared"],fill=C["purplepale"],size=6.4)
        txt(47,41,r"$\bar m_v$",7.5,C["shared"],ha="center")

        # Shared projection follows its own visible bypass to the sum.
        arrow([(58,48.5),(64,48.5)],C["shared"],.95)
        box(64,44,30,9,r"$s_v=W_s\bar m_v$",color=C["shared"],
            fill=C["purplepale"],size=7.5)
        arrow([(94,48.5),(143,48.5)],C["shared"],1.15)
        txt(116,51,"Shared path (ungated)",6.1,C["shared"],ha="center")
        # The private operator explicitly receives both m and its observed mean.
        arrow([(47,39),(47,36)],C["shared"],.75)
        line([(25,48.5),(25,28.5)],C["quiet"],.65)
        for y in [48.5,40.5]:
            ax.add_patch(Circle((25,y),.4,fc=C["quiet"],ec="none",zorder=6))
        arrow([(25,28.5),(36,28.5)],C["quiet"],.7)
        box(36,26,58,10,
            r"$p_{r,v}=W_{p,r}(m_{r,v}-\bar m_v)$",
            color="#C6CDD4",fill="white",size=7.5)
        txt(64.5,23,"Only observed relations contribute",6.0,C["quiet"],ha="center")
        arrow([(94,31),(108,31)],C["ink"],.85)
        box(108,25,23,12,r"$\sum_r\alpha_{r,v}p_{r,v}$",
            fill=C["purplepale"],color=C["shared"],size=7.2)
        txt(119.5,40.4,"Private mixture",6.4,ha="center")
        arrow([(131,31),(145.5,31),(145.5,46.2)],C["ink"],.85)
        ax.add_patch(Circle((145.5,48.5),2.3,fc="white",ec=C["shared"],lw=1,zorder=5))
        txt(145.5,48.5,"+",9,ha="center",zorder=6)
        arrow([(147.8,48.5),(161,48.5),(161,42)],C["ink"],.85)
        box(148,28,26,14,"ReLU + dropout\nResidual + norm",size=6.5)
        arrow([(161,28),(161,21)],C["ink"])
        txt(161,17.5,r"$h_v^{(\ell+1)}$",8,ha="center")
        txt(147,24,r"$h_v^{(\ell)}$",7,ha="right")
        arrow([(148,24),(153,24),(153,28)],C["quiet"],.7)

        # Motif glyphs and gate conditioning are a third, non-overlapping lane.
        line([(7,20),(132,20)],C["rule"],.55)
        txt(7,15.5,"Typed triads",6.6,C["shared"],weight="bold")
        for x,colors in [(35,["pass_","pass_"]),(46,["pass_","adv"]),
                          (57,["adv","turn"])]:
            pts=[(x,11.5),(x+4,16),(x+7,11.5)]
            for (u,v),col in zip([(0,1),(1,2)],colors):
                line([pts[u],pts[v]],C[col],.9)
            for xx,yy in pts:
                ax.add_patch(Circle((xx,yy),.65,fc="white",ec=C["ink"],lw=.5))
        txt(7,7.9,"Node state  +  availability  +  mean  +  conflict",6.15,C["quiet"])
        arrow([(69,13.3),(77,13.3)],C["shared"],.85)
        box(77,7.5,31,11.5,"Motif-conditioned\nmasked gate",color=C["shared"],
            fill=C["purplepale"],size=6.6)
        arrow([(108,13.3),(119.5,13.3),(119.5,25)],C["shared"],.9)
        txt(125,18,r"$\alpha_{r,v}$",7.5,C["shared"],ha="center")
        txt(135,7.8,r"$\frac{1}{2}(\mathcal{L}_{\rm cls}+\mathcal{L}_{\rm reg})+\lambda\mathcal{L}_{\perp}$",
            6.7,C["quiet"])
        # No embedding vector, motif glyph, or line thickness is a measured value.
        FIG.mkdir(parents=True,exist_ok=True)
        for ext in ["pdf","svg","png"]:
            fig.savefig(FIG/f"Figure_01_Method_Overview.{ext}",dpi=450,
                        bbox_inches=None,pad_inches=0)
        plt.close(fig)


if __name__ == "__main__":
    draw()
