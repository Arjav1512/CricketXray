"""Matplotlib charts for the Streamlit app, styled with the design-system tokens.

Every function returns a Figure; the app displays it with st.pyplot and closes it.
Charts only present values computed in modeling.py; they never change them.
"""

import matplotlib

matplotlib.use("Agg")  # no GUI backend needed inside Streamlit

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Design-system tokens (same values as .streamlit/config.toml).
BG = "#0A0A0A"
SURFACE = "#141414"
RAISED = "#1E1E1E"
BORDER = "#292929"
FG = "#F5F5F3"
MUTED = "#A3A3A3"
SUBTLE = "#707070"
ACCENT = "#F2693A"   # orange: the selected / highlighted value
GREEN = "#59D08A"
VIOLET = "#A78BFA"

# One fixed colour and marker per cluster number, used by every chart and badge.
# The first item is the Streamlit badge colour name with the closest theme colour.
CLUSTER_PALETTE = [
    ("orange", "#F2693A"), ("green", "#59D08A"), ("violet", "#A78BFA"), ("yellow", "#F8B544"),
    ("blue", "#6CB6F5"), ("red", "#EF5B5B"), ("gray", "#EDEDE8"), ("gray", "#8A8A8A"),
]
CLUSTER_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _pick_font(candidates):
    available = {f.name for f in font_manager.fontManager.ttflist}
    return next((name for name in candidates if name in available), "DejaVu Sans")


matplotlib.rcParams.update({
    "font.family": _pick_font(["Geist", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]),
    "font.size": 9,
    "axes.titleweight": "medium",
    "text.color": FG,
    "axes.labelcolor": MUTED,
    "xtick.color": SUBTLE,
    "ytick.color": SUBTLE,
    "xtick.labelcolor": MUTED,
    "ytick.labelcolor": MUTED,
})


def cluster_color(cluster_id):
    return CLUSTER_PALETTE[(int(cluster_id) - 1) % len(CLUSTER_PALETTE)][1]


def cluster_badge_color(cluster_id):
    return CLUSTER_PALETTE[(int(cluster_id) - 1) % len(CLUSTER_PALETTE)][0]


def cluster_marker(cluster_id):
    return CLUSTER_MARKERS[(int(cluster_id) - 1) % len(CLUSTER_MARKERS)]


def _figure(width, height, grid_axis="y"):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BORDER)
    ax.tick_params(length=0, pad=6, labelsize=9)
    if grid_axis:
        ax.grid(axis=grid_axis, color=RAISED, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _finish(fig, ax, title="", xlabel="", ylabel=""):
    # The app shows chart titles as UI headings; pass a title only for standalone images.
    if title:
        ax.set_title(title, fontsize=12, loc="left", pad=14, color=FG)
    ax.set_xlabel(xlabel, fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=8)
    fig.tight_layout()
    return fig


def _empty(fig, ax, message, title=""):
    """Placeholder when there is nothing valid to plot."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, 0.5, message, ha="center", va="center", color=MUTED, fontsize=10, transform=ax.transAxes)
    return _finish(fig, ax, title)


def _legend(ax, **kwargs):
    legend = ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, handletextpad=0.4, **kwargs)
    return legend


def elbow_chart(evaluation, selected_k, elbow_k=None, title=""):
    """Inertia for each valid K. Orange = selected K; violet ring = sharpest bend."""
    fig, ax = _figure(4.4, 3.1)
    valid = evaluation.dropna(subset=["inertia"])
    if valid.empty:
        return _empty(fig, ax, "No valid K for these settings", title)
    ax.plot(valid.index, valid["inertia"], color=MUTED, linewidth=1.6, zorder=2)
    ax.scatter(valid.index, valid["inertia"], s=28, color=FG, zorder=3)
    if elbow_k in valid.index:
        ax.scatter([elbow_k], [valid.loc[elbow_k, "inertia"]], s=200, facecolors="none",
                   edgecolors=VIOLET, linewidths=1.6, zorder=4, label=f"sharpest bend (K = {elbow_k})")
    if selected_k in valid.index:
        ax.scatter([selected_k], [valid.loc[selected_k, "inertia"]], s=70, color=ACCENT, zorder=5,
                   label=f"selected (K = {selected_k})")
    ax.set_xticks(list(valid.index))
    ax.margins(x=0.08, y=0.15)
    _legend(ax, loc="upper right")
    return _finish(fig, ax, title, "Number of clusters (K)", "Inertia (within-cluster SS)")


def silhouette_chart(evaluation, selected_k, title=""):
    """Silhouette score for each valid K; selected K in orange, highest score marked."""
    fig, ax = _figure(4.4, 3.1)
    valid = evaluation.dropna(subset=["silhouette"])
    if valid.empty:
        return _empty(fig, ax, "No valid silhouette scores for these settings", title)
    best = valid["silhouette"].idxmax()
    colors = [ACCENT if k == selected_k else "#3A3A3A" for k in valid.index]
    bars = ax.bar(valid.index, valid["silhouette"], color=colors, width=0.62)
    for k, bar, value in zip(valid.index, bars, valid["silhouette"]):
        x = bar.get_x() + bar.get_width() / 2
        ax.text(x, bar.get_height() + 0.004, f"{value:.3f}", ha="center", va="bottom", fontsize=8,
                color=FG if k == selected_k else MUTED)
        if k == best:  # labelled in words, not by colour alone
            ax.text(x, bar.get_height() + 0.026, "highest", ha="center", va="bottom", fontsize=8, color=GREEN)
    ax.set_xticks(list(valid.index))
    ax.set_ylim(0, max(valid["silhouette"].max() * 1.3, 0.1))
    return _finish(fig, ax, title, "Number of clusters (K)", "Mean silhouette (higher = better)")


def pca_scatter(coords, labels, cluster_names, explained, width=9.2, height=5.4, title="",
                compact_legend=False):
    """2-D PCA projection. Each cluster has its own colour and marker shape.
    compact_legend shows only cluster numbers (for small previews next to a labelled chart)."""
    fig, ax = _figure(width, height, grid_axis="both")
    if coords.empty:
        return _empty(fig, ax, "No players to plot", title)
    ax.axhline(0, color=BORDER, linewidth=0.8, zorder=1)
    ax.axvline(0, color=BORDER, linewidth=0.8, zorder=1)
    for cluster_id in sorted(labels.unique()):
        members = labels[labels == cluster_id].index
        ax.scatter(coords.loc[members, "PC1"], coords.loc[members, "PC2"], s=16, alpha=0.85,
                   color=cluster_color(cluster_id), marker=cluster_marker(cluster_id), linewidths=0,
                   zorder=2, label=(f"Cluster {cluster_id}" if compact_legend else
                                    f"{cluster_id} · {cluster_names[cluster_id]} ({len(members)})"))
    n = labels.nunique()
    _legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.16 if compact_legend else -0.13),
            ncol=min(n, 4) if compact_legend else 2, markerscale=1.8, columnspacing=1.6)
    return _finish(fig, ax, title,
                   f"PC1 · {explained['PC1']:.1%} of variance", f"PC2 · {explained['PC2']:.1%} of variance")


def cluster_sizes_chart(sizes, cluster_names, width=4.4, title=""):
    """Horizontal bars: players per cluster, in the cluster colours."""
    fig, ax = _figure(width, 0.5 * len(sizes) + 1.2, grid_axis=None)
    if len(sizes) == 0:
        return _empty(fig, ax, "No clusters", title)
    y = np.arange(len(sizes))
    total = sizes.sum()
    ax.barh(y, sizes.values, color=[cluster_color(c) for c in sizes.index], height=0.55)
    for yi, (cluster_id, n) in zip(y, sizes.items()):
        ax.text(n + total * 0.01, yi, f"{n}  ·  {n / total:.0%}", va="center", fontsize=9, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c} · {cluster_names[c]}" for c in sizes.index], fontsize=9, color=FG)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, sizes.max() * 1.3)
    return _finish(fig, ax, title)


def loadings_chart(loadings, feature_label, width=4.4, title=""):
    """Horizontal bars of each feature's weight on PC1 (orange) and PC2 (violet)."""
    fig, ax = _figure(width, 0.42 * len(loadings) + 1.5, grid_axis="x")
    y = np.arange(len(loadings))
    ax.barh(y - 0.2, loadings["PC1"], height=0.38, color=ACCENT, label="PC1")
    ax.barh(y + 0.2, loadings["PC2"], height=0.38, color=VIOLET, label="PC2")
    ax.set_yticks(y)
    ax.set_yticklabels([feature_label(f) for f in loadings.index], fontsize=9, color=FG)
    ax.axvline(0, color=SUBTLE, linewidth=0.8)
    ax.invert_yaxis()
    _legend(ax, loc="lower right")
    return _finish(fig, ax, title, "Weight on component")


# Diverging map: violet (below average) -> surface -> orange (above average).
RELATIVE_CMAP = LinearSegmentedColormap.from_list("relative", [VIOLET, RAISED, ACCENT])


def relative_heatmap(relative, feature_label, cluster_names, title=""):
    """Cluster centres in standard deviations from the average modeled player."""
    data = relative.T
    fig, ax = _figure(9.2, 0.46 * data.shape[0] + 1.8, grid_axis=None)
    if data.empty:
        return _empty(fig, ax, "No clusters", title)
    for spine in ax.spines.values():
        spine.set_visible(False)
    image = ax.imshow(data.to_numpy(), cmap=RELATIVE_CMAP, vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels([f"{c}\n{cluster_names[c]}" for c in data.columns], fontsize=8)
    ax.xaxis.tick_top()
    for tick, cluster_id in zip(ax.xaxis.get_major_ticks(), data.columns):
        tick.label2.set_color(cluster_color(cluster_id))  # top labels are label2
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels([feature_label(f) for f in data.index], fontsize=9, color=FG)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data.iat[i, j]
            text = "0.0" if round(value, 1) == 0 else f"{value:+.1f}"  # never show "-0.0"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color=BG if abs(value) >= 1.1 else FG)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8, length=0)
    bar.set_label("SD from average (relative score)", color=MUTED, fontsize=8)
    if title:
        ax.set_title(title, fontsize=12, loc="left", pad=36, color=FG)
    fig.tight_layout()
    return fig
