"""Final scientific figure styles; vector PDF/SVG and 600-dpi PNG."""
import os
os.environ.setdefault("MPLBACKEND","Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .predictions import GROUPS
def add_panel_header_oneline(ax, letter, title, x=0.00, y=1.06, pad=0.012):
    t_letter = ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        fontweight="bold",
        bbox=dict(
            boxstyle="square,pad=0.18",
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
        ),
        clip_on=False,
    )

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    bb = t_letter.get_window_extent(renderer=renderer)

    x0_px = ax.transAxes.transform((0, 0))[0]
    x1_px = ax.transAxes.transform((1, 0))[0]
    axes_w_px = x1_px - x0_px
    box_w_axes = bb.width / axes_w_px

    ax.text(
        x + box_w_axes + pad,
        y,
        title,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        clip_on=False,
    )

def _nice_xticks(max_x, step=2):
    max_xi = int(np.ceil(max_x))
    return np.arange(0, max_xi + 1, step)

def plot_curves_with_styles(
    ax, x, curves_dict, style_map, period,
    ylim=(0, 100),
    show_ylabel=True,
    y_label="Predicted probability\nof patient reply 7d (%)",
):
    for label, y in curves_dict.items():
        st = style_map.get(label, {})
        ax.plot(
            x, y * 100,
            linewidth=st.get("lw", 2.0),
            linestyle=st.get("ls", "-"),
            color=st.get("color", "0.35"),
            marker=st.get("marker", None),
            markersize=st.get("markersize", 4.6),
            markevery=st.get("markevery", 4),
            markerfacecolor="white",
            markeredgewidth=0.9,
            label=str(label),
        )

    ax.set_xlabel(f"Physician/APP message volume\n(past {period}; capped at p99)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(*ylim)
    ax.set_yticks(np.arange(0, 61, 10))
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_ylabel(y_label if show_ylabel else "")
    ax.tick_params(axis="y", which="both", labelleft=True)

def add_panel_box_label(ax, letter, title, x_letter=-0.06, y=1.055, x_title_pad=0.08):
    ax.text(
        x_letter, y, letter,
        transform=ax.transAxes,
        va="bottom", ha="left",
        fontsize=11, fontweight="bold",
        bbox=dict(
            boxstyle="square,pad=0.18",
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
        ),
        clip_on=False,
    )
    ax.text(
        x_letter + x_title_pad, y, title,
        transform=ax.transAxes,
        va="bottom", ha="left",
        fontsize=11,
        clip_on=False,
    )

def save_figure(fig,run,name):
    for extension in ["pdf","svg","png"]:
        kwargs={"dpi":600} if extension=="png" else {}
        fig.savefig(run/f"{name}.{extension}",format=extension,bbox_inches="tight",
            pad_inches=.05,facecolor="white",**kwargs)
    plt.close(fig)

def figure1(curves,models,run):
    plt.rcdefaults()
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9, 'legend.title_fontsize': 9, 'axes.grid': True, 'grid.alpha': 0.25, 'grid.linestyle': ':', 'savefig.dpi': 300})
    style_map = {50: {'color': 'black', 'linestyle': '-', 'marker': 'o', 'label': '50th (median)'}, 90: {'color': 'dimgray', 'linestyle': '--', 'marker': 's', 'label': '90th'}, 99: {'color': 'gray', 'linestyle': '-.', 'marker': '^', 'label': '99th'}}
    labels={'A_title': 'HCW x System\nReply to all HCW threads', 'B_title': 'Physician/APP x Other HCW\nReply to physician/APP-initiated threads', 'C_title': 'Physician/APP x Other HCW\nReply to other HCW-initiated threads', 'A_xlabel': 'HCW message volume (past 30 days; count; capped at p99)', 'B_xlabel': 'Physician/APP message volume (past 30 days; count; capped at p99)', 'C_xlabel': 'Physician/APP message volume (past 30 days; count; capped at p99)', 'A_legend_title': 'System message percentile (messages)', 'B_legend_title': 'Other HCW message percentile (messages)', 'C_legend_title': 'Other HCW message percentile (messages)'}
    ymax=62.
    height=4.4*ymax/60.
    fig,axes=plt.subplots(1,3,figsize=(17.2,height),sharey=True)
    direction=[]
    display_limits={}
    # Figure 2 displays count 1 with limits 0.35–13.65. Match its fractional
    # left inset in A/B while preserving their existing right limits and C.
    figure2_left_inset_fraction=(1.-.35)/(13.65-.35)
    for ax,letter in zip(axes,"ABC"):
        m=models[letter];part=curves.loc[curves.panel.eq(letter)]
        assert part.message_count_30d.min()==m["focal_minimum"]
        assert part.message_count_30d.max()==m["focal_p99"]
        assert (m["focal_minimum"]==1 if letter in "AB" else m["focal_minimum"]==0)
        assert len(part.groupby("moderator_percentile"))==3
        for pct in [50,90,99]:
            st=style_map[pct];row=part.loc[part.moderator_percentile.eq(pct)].sort_values("message_count_30d")
            moderator=float(m["moderator_percentiles"][str(pct)])
            ax.fill_between(row.message_count_30d.to_numpy(),row.ci95_lower_pct.to_numpy(),row.ci95_upper_pct.to_numpy(),
                            color=st["color"],alpha=.10,linewidth=0,edgecolor="none",zorder=1,rasterized=False)
            ax.plot(row.message_count_30d,row.predicted_probability_pct,lw=2.1,color=st["color"],
                    linestyle=st["linestyle"],marker=st["marker"],markersize=4.8,markevery=20,
                    label=f"{st['label']} ({moderator:.0f} msgs)",zorder=3,clip_on=False)
            direction.append({"panel":letter,"moderator_percentile":pct,"moderator_count":moderator,
                              "first_probability_pct":float(row.predicted_probability_pct.iloc[0]),
                              "last_probability_pct":float(row.predicted_probability_pct.iloc[-1]),
                              "mean_curve_direction":"decreasing" if row.predicted_probability_pct.iloc[-1]<row.predicted_probability_pct.iloc[0] else "increasing"})
        add_panel_header_oneline(ax,letter,labels[f"{letter}_title"],x=-.03,y=1.15,pad=.030)
        ax.set_xlabel(labels[f"{letter}_xlabel"])
        left=m["focal_minimum"]
        if letter in "AB":
            span=m["focal_p99"]-left
            left-=span*figure2_left_inset_fraction/(1.-figure2_left_inset_fraction)
        ax.set_xlim(left,m["focal_p99"])
        ticks=_nice_xticks(m["focal_p99"],step=2)
        ax.set_xticks(ticks[ticks>=m["focal_minimum"]])
        display_limits[letter]=list(ax.get_xlim())
        if letter in "AB":
            inset=(m["focal_minimum"]-left)/(m["focal_p99"]-left)
            assert np.isclose(inset,figure2_left_inset_fraction,atol=1e-12)
        else:
            assert list(ax.get_xlim())==[0.,9.] and 0 in ax.get_xticks()
        ax.legend(frameon=False,loc="upper center",bbox_to_anchor=(.5,-.22),ncol=3,
                  title=labels[f"{letter}_legend_title"])
        ax.set_ylabel("Predicted probability\nof patient reply 7d (%)")
        ax.set_ylim(0,ymax);ax.set_yticks(np.arange(0,ymax+.001,10))
        ax.tick_params(axis="y",which="both",labelleft=True)
    fig.subplots_adjust(left=.06,right=.99,top=.83,bottom=.28,wspace=.26)
    assert not fig.texts and all(len(ax.collections)==3 and len(ax.lines)==3 for ax in axes)
    fig.canvas.draw()

    save_figure(fig,run,"Figure1")

def figure2(curves,run):
    plt.rcdefaults()
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9, 'legend.title_fontsize': 9, 'savefig.dpi': 300})
    ref = {"color":"0.0","ls":"-","lw":2.2,"marker":"o"}
    main = {"color":"0.30","ls":"--","lw":2.2,"marker":"s"}
    mild = {"color":"0.55","ls":(0,(1,1)),"lw":2.0,"marker":"^"}
    other = {"color":"0.70","ls":(0,(3,1,1,1)),"lw":2.0,"marker":"D"}
    styles = {"Female":ref,"Male":main,"White":ref,"Black/African American":main,"Asian":mild,
              "Other/Multiple/Unknown":other,"0 (None)":ref,"3-4 (Moderate)":main,
              "1-2 (Mild)":mild,"5+ (Severe)":other}
    # Preserve the original 4/8/12-message marker locations; replace the old
    # zero-message marker with the supported minimum of one.
    styles = {k:{**v,"markevery":[0,3,7,11]} for k,v in styles.items()}
    original_height, original_ymax = 4.4, 62.
    ymin, ymax = 20., 70.
    assert curves.ci95_lower_pct.min() > ymin and curves.ci95_upper_pct.max() < ymax
    # Common probability scale across subgroup panels.
    height = original_height
    fig, axes = plt.subplots(1,3,figsize=(13.2,height),dpi=300)
    for idx,(panel,(_,groups)) in enumerate(GROUPS.items()):
        ax = axes[idx]
        line_curves = {}
        for group in groups:
            subset = curves.loc[curves.panel.eq(panel)&curves.group.eq(group)]
            assert len(subset) == 241
            band = ax.fill_between(subset.message_count_30d.to_numpy(),subset.ci95_lower_pct.to_numpy(),
                                   subset.ci95_upper_pct.to_numpy(),color=styles[group]["color"],
                                   alpha=.055,edgecolor="none",linewidth=0,zorder=1,rasterized=False)
            integer = subset.loc[np.isclose(subset.message_count_30d,subset.message_count_30d.round())]
            assert integer.message_count_30d.tolist() == list(range(1,14))
            line_curves[group] = integer.predicted_probability_pct.to_numpy()/100
        plot_curves_with_styles(ax,list(range(1,14)),line_curves,styles,"30d",ylim=(ymin,ymax),show_ylabel=(idx==0))
        # The legacy helper sets ticks at 0–60, which can expand the axis.
        # Replace those ticks and enforce the requested limits afterward.
        ax.set_yticks(np.arange(20,71,10))
        ax.set_ylim(ymin,ymax)
        for line in ax.lines:
            line.set_zorder(3)
        # Display regular even ticks; predictions still begin at one message.
        ax.set_xticks([2,4,6,8,10,12])
        ax.set_xlim(.35,13.65)
        add_panel_box_label(ax,"ABC"[idx],["Sex","Race","Charlson comorbidity index"][idx])
        handles,labels = ax.get_legend_handles_labels()
        labels = ["Other" if label=="Other/Multiple/Unknown" else label for label in labels]
        ax.legend(handles,labels,frameon=False,title={"Sex":"Sex","Race":"Race","CCI":"Charlson group"}[panel],
                  loc="lower right",bbox_to_anchor=(1.12,.10),bbox_transform=ax.transAxes,borderaxespad=.6,
                  handlelength=2.6,handletextpad=.8,labelspacing=.4)
    fig.subplots_adjust(bottom=.14,wspace=.24,top=.88)
    assert not fig.texts, "Do not put explanatory notes in the figure."
    fig.canvas.draw()
    positions = []
    for ax in axes:
        bounds = ax.get_position()
        positions.append({"width_inches":bounds.width*13.2,"height_inches":bounds.height*height,
                          "height_per_percentage_point_inches":bounds.height*height/(ymax-ymin)})
        assert ax.get_ylim() == (20.,70.)
        assert list(ax.get_yticks()) == [20,30,40,50,60,70]
        assert list(ax.get_xticks()) == [2,4,6,8,10,12]
        assert all(min(line.get_xdata()) == 1 for line in ax.lines)
        assert len(ax.collections) == len(ax.lines)

    save_figure(fig,run,"Figure2")
