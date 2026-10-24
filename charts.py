"""Matplotlib charts for the Streamlit app, styled to match .streamlit/config.toml.

Every function returns a Figure; the app displays it with st.pyplot and closes it.
"""

import matplotlib

matplotlib.use("Agg")  # no GUI backend needed inside Streamlit

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BACKGROUND = "#0B1410"
PANEL = "#14231B"
TEXT = "#E6EEE9"
MUTED = "#94A89C"
GRID = "#24392D"
ACCENT = "#3DDC84"   # pitch green
HIGHLIGHT = "#F2C14E"  # trophy gold
CLUSTER_COLORS = ["#3DDC84", "#F2C14E", "#F06A6A", "#5BB4F0", "#B48CF2", "#F59E5B", "#4FD1C5", "#E6EEE9"]


def cluster_color(cluster_id):
    return CLUSTER_COLORS[(int(cluster_id) - 1) % len(CLUSTER_COLORS)]


def _figure(width=6.4, height=3.8):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _finish(fig, ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=10, color=TEXT)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    fig.tight_layout()
    return fig


def elbow_chart(evaluation, selected_k):
    """Inertia for each valid K, with the selected K highlighted."""
    fig, ax = _figure()
    valid = evaluation.dropna(subset=["inertia"])
    ax.plot(valid.index, valid["inertia"], color=ACCENT, marker="o", linewidth=2, markersize=6)
    if selected_k in valid.index:
        y = valid.loc[selected_k, "inertia"]
        ax.scatter([selected_k], [y], s=160, facecolors="none", edgecolors=HIGHLIGHT, linewidths=2, zorder=3)
        ax.annotate(f"selected K = {selected_k}", (selected_k, y), xytext=(10, 12),
                    textcoords="offset points", color=HIGHLIGHT, fontsize=8)
    ax.set_xticks(list(valid.index))
    return _finish(fig, ax, "Elbow method", "Number of clusters (K)",
                   "Inertia (within-cluster sum of squares)")


def silhouette_chart(evaluation, selected_k):
    """Silhouette score for each valid K; the selected K is drawn in gold."""
    fig, ax = _figure()
    valid = evaluation.dropna(subset=["silhouette"])
    colors = [HIGHLIGHT if k == selected_k else ACCENT for k in valid.index]
    bars = ax.bar(valid.index, valid["silhouette"], color=colors, width=0.6)
    for bar, value in zip(bars, valid["silhouette"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}",
                ha="center", va="bottom", color=TEXT, fontsize=7)
    ax.set_xticks(list(valid.index))
    if len(valid):
        ax.set_ylim(0, max(valid["silhouette"].max() * 1.2, 0.1))
    return _finish(fig, ax, "Silhouette score (higher = better separated)",
                   "Number of clusters (K)", "Mean silhouette score")


def pca_scatter(coords, labels, cluster_names, explained):
    """2-D PCA projection, one colour per cluster."""
    fig, ax = _figure(7.2, 5.0)
    for cluster_id in sorted(labels.unique()):
        members = labels[labels == cluster_id].index
        ax.scatter(coords.loc[members, "PC1"], coords.loc[members, "PC2"], s=14, alpha=0.75,
                   color=cluster_color(cluster_id), edgecolors="none",
                   label=f"{cluster_id} · {cluster_names[cluster_id]} ({len(members)})")
    legend = ax.legend(fontsize=7, loc="best", frameon=True, markerscale=1.6)
    legend.get_frame().set_facecolor(BACKGROUND)
    legend.get_frame().set_edgecolor(GRID)
    for text in legend.get_texts():
        text.set_color(TEXT)
    return _finish(fig, ax, "Players projected onto the first two principal components",
                   f"PC1 ({explained['PC1']:.1%} of variance)",
                   f"PC2 ({explained['PC2']:.1%} of variance)")


def loadings_chart(loadings, feature_label):
    """Horizontal bars of each feature's weight on PC1 and PC2."""
    fig, ax = _figure(6.4, 0.42 * len(loadings) + 1.4)
    y = np.arange(len(loadings))
    ax.barh(y - 0.2, loadings["PC1"], height=0.38, color=ACCENT, label="PC1")
    ax.barh(y + 0.2, loadings["PC2"], height=0.38, color=HIGHLIGHT, label="PC2")
    ax.set_yticks(y)
    ax.set_yticklabels([feature_label(f) for f in loadings.index], color=TEXT, fontsize=8)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.invert_yaxis()
    legend = ax.legend(fontsize=8, frameon=False)
    for text in legend.get_texts():
        text.set_color(TEXT)
    return _finish(fig, ax, "PCA loadings (feature weights)", "Loading", "")


def relative_heatmap(relative, feature_label, cluster_names):
    """Cluster centres in standard deviations from the average modeled player."""
    data = relative.T
    fig, ax = _figure(1.5 * data.shape[1] + 3.2, 0.42 * data.shape[0] + 1.6)
    ax.grid(False)
    image = ax.imshow(data.to_numpy(), cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels([f"{c}\n{cluster_names[c]}" for c in data.columns], fontsize=7, color=TEXT)
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels([feature_label(f) for f in data.index], fontsize=8, color=TEXT)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data.iat[i, j]
            ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=7,
                    color="#0B1410" if abs(value) < 1.2 else "white")
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.ax.tick_params(colors=MUTED, labelsize=7)
    bar.set_label("SD from average (relative, not a real statistic)", color=MUTED, fontsize=7)
    return _finish(fig, ax, "Relative cluster profiles", "", "")
