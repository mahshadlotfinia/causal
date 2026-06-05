"""
figures/supplementary_figure2_reliability_diagrams.py
Created May 27, 2026

Supplementary Figure 2: Reliability diagrams for label-referenced calibration.

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia/
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


class SupplementaryFigure2Reliability:
    ORIGINAL_CSV = "/PATH/mimic_original_cases.csv"
    METRICS_CSV = "/PATH/all_metrics_mimic.csv"
    OUT_PDF = "/PATH/supplementary_figure2_reliability_diagrams.pdf"

    MODELS = [
        "Gemma-4-26B", "GPT-5", "Qwen3-VL-32B", "MedGemma-1.5-4B",
        "RAD-DINO", "LLaVA-Med-7B", "MedGemma-27B-text", "Mistral-Small-4-119B",
    ]

    CAT_COLORS = {
        "Uses image": "#2166AC",
        "Ignores image": "#B2182B",
        "Unstable": "#E08214",
        "Other": "#777777",
    }

    PANEL_LETTERS = list("abcdefgh")

    def __init__(self, original_csv=None, metrics_csv=None, out_pdf=None):
        self.original_csv = Path(original_csv or self.ORIGINAL_CSV)
        self.metrics_csv = Path(metrics_csv or self.METRICS_CSV)
        self.out_pdf = Path(out_pdf or self.OUT_PDF)
        self.log = logging.getLogger(self.__class__.__name__)
        self._set_rcparams()
        self._load_data()
        self.fig = None

    @staticmethod
    def _set_rcparams():
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 15.5,
            "axes.labelsize": 15.5,
            "axes.titlesize": 16.0,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 14.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })

    @staticmethod
    def _require_columns(df, columns, where):
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {where}: {missing}")

    def _load_data(self):
        if not self.original_csv.exists():
            raise FileNotFoundError(self.original_csv)
        if not self.metrics_csv.exists():
            raise FileNotFoundError(self.metrics_csv)
        self.original = pd.read_csv(self.original_csv)
        self.metrics = pd.read_csv(self.metrics_csv)

        self._require_columns(self.original, ["ground_truth"], self.original_csv.name)
        needed = ["model", "CGR", "CGR_ci_lower", "UAR", "irrelevant_stable", "ECE"]
        self._require_columns(self.metrics, needed, self.metrics_csv.name)

        for model in self.MODELS:
            self._require_columns(
                self.original,
                [f"{model}_parsed_answer", f"{model}_confidence"],
                self.original_csv.name,
            )

        self.metrics = self.metrics.set_index("model", drop=False)
        self.metrics["category"] = self.metrics.apply(self._assign_category, axis=1)

    @staticmethod
    def _assign_category(row):
        cgr = row["CGR"]
        cgr_lo = row["CGR_ci_lower"]
        uar = row["UAR"]
        isv = row["irrelevant_stable"]
        if pd.isna(cgr) or pd.isna(cgr_lo) or pd.isna(uar) or pd.isna(isv):
            return "Other"
        if isv < 0.70:
            return "Unstable"
        if np.isclose(cgr, 0.0) and np.isclose(uar, 1.0) and np.isclose(isv, 1.0):
            return "Ignores image"
        if (cgr > 0.0) and (cgr_lo > 0.0) and (isv >= 0.90):
            return "Uses image"
        return "Other"

    @staticmethod
    def _lighten(hex_color, amount=0.72):
        hex_color = hex_color.lstrip("#")
        rgb = np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)], dtype=float) / 255.0
        out = rgb + (1.0 - rgb) * amount
        return tuple(out)

    def _model_color(self, model):
        cat = self.metrics.loc[model, "category"]
        return self.CAT_COLORS.get(cat, self.CAT_COLORS["Other"])

    def _case_data(self, model):
        ans_col = f"{model}_parsed_answer"
        conf_col = f"{model}_confidence"
        df = self.original[["ground_truth", ans_col, conf_col]].copy()
        df = df.rename(columns={ans_col: "parsed_answer", conf_col: "confidence"})
        df = df[df["parsed_answer"].notna() & df["confidence"].notna() & df["ground_truth"].notna()].copy()
        df["confidence"] = df["confidence"].astype(float).clip(0, 1)
        df["ground_truth"] = df["ground_truth"].astype(float)
        return df

    @staticmethod
    def _reliability_bins(df, nbins=10):
        if df.empty:
            return pd.DataFrame(columns=["bin", "mean_conf", "pos_rate", "n"])
        conf = df["confidence"].to_numpy(dtype=float)
        y = df["ground_truth"].to_numpy(dtype=float)
        idx = np.minimum((conf * nbins).astype(int), nbins - 1)
        rows = []
        for b in range(nbins):
            mask = idx == b
            if not np.any(mask):
                continue
            rows.append({
                "bin": b,
                "mean_conf": float(conf[mask].mean()),
                "pos_rate": float(y[mask].mean()),
                "n": int(mask.sum()),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _format_ece(value):
        if pd.isna(value):
            return "ECE = N/A"
        return f"ECE = {value * 100:.1f}"

    @staticmethod
    def _panel_label(ax, letter, title="", x=-0.20, y=1.06, title_offset=0.075):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=22,
                fontweight="bold", ha="left", va="bottom")
        if title:
            ax.text(x + title_offset, y, title, transform=ax.transAxes,
                    fontsize=14.8, fontweight="normal",
                    ha="left", va="bottom", color="#222222")

    def build(self):
        self.fig = plt.figure(figsize=(20, 11), facecolor="white")
        gs_leg = self.fig.add_gridspec(1, 1, left=0.045, right=0.985, top=0.985, bottom=0.945)
        gs = self.fig.add_gridspec(
            2, 4, left=0.060, right=0.980, top=0.885, bottom=0.070,
            wspace=0.28, hspace=0.42,
        )
        self._draw_legend(self.fig.add_subplot(gs_leg[0, 0]))
        for i, model in enumerate(self.MODELS):
            ax = self.fig.add_subplot(gs[i // 4, i % 4])
            self._panel(ax, model, self.PANEL_LETTERS[i], show_ylabel=(i % 4 == 0))
        return self

    def _draw_legend(self, ax):
        ax.axis("off")
        size_small = Line2D([0], [0], marker="o", lw=0, markersize=6,
                            mfc="#555555", mec="white", mew=0.6, label="small bin")
        size_large = Line2D([0], [0], marker="o", lw=0, markersize=13,
                            mfc="#555555", mec="white", mew=0.6, label="large bin")
        diag = Line2D([0], [0], color="#888888", lw=1.2, ls="--", label="dashed: perfect calibration")
        shade = Patch(facecolor="#2166AC", alpha=0.12, edgecolor="none", label="shaded: calibration gap")
        text_handle = Patch(facecolor="white", edgecolor="white", label="panel color = behavioral category")
        handles = [size_small, size_large, diag, shade, text_handle]
        labels = ["marker area ∝ bin count", "", "dashed: perfect calibration", "shaded: calibration gap", "panel color = behavioral category"]
        ax.legend(handles, labels, ncol=5, frameon=False, loc="center", bbox_to_anchor=(0.51, 0.52),
                  columnspacing=1.35, handletextpad=0.45)

    def _panel(self, ax, model, letter, show_ylabel=False):
        col = self._model_color(model)
        light_col = self._lighten(col, 0.45)
        df = self._case_data(model)
        bins = self._reliability_bins(df, nbins=10)

        ax.plot([0, 1], [0, 1], color="#888888", lw=1.2, ls="--", zorder=1)

        if not bins.empty:
            x = bins["mean_conf"].to_numpy()
            y = bins["pos_rate"].to_numpy()
            # Fill by sorting x to avoid self-crossing where bins are sparse.
            order = np.argsort(x)
            ax.fill_between(
                x[order], y[order], x[order], color=col, alpha=0.12, zorder=2,
                interpolate=True,
            )
            ax.plot(x[order], y[order], color=col, lw=1.9, zorder=3)
            sizes = np.clip(20 + 4 * bins["n"].to_numpy(), 45, 520)
            ax.scatter(x, y, s=sizes, color=col, edgecolor="white", linewidth=0.6, zorder=4)

        # Bottom confidence rug as a marginal histogram.
        rug_ax = ax.inset_axes([0.0, 0.010, 1.0, 0.115], transform=ax.transAxes)
        hist, edges = np.histogram(df["confidence"].to_numpy(dtype=float), bins=np.linspace(0, 1, 21))
        if hist.max() > 0:
            hist_scaled = hist / hist.max()
        else:
            hist_scaled = hist
        centers = (edges[:-1] + edges[1:]) / 2
        rug_ax.bar(centers, hist_scaled, width=0.047, color="#BBBBBB", alpha=0.70,
                   edgecolor="none", align="center")
        rug_ax.set_xlim(0, 1)
        rug_ax.set_ylim(0, 1.05)
        rug_ax.axis("off")

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xlabel("mean predicted P(Yes) in bin", fontsize=13.8)
        if show_ylabel:
            ax.set_ylabel("empirical positive rate", fontsize=13.8)
        else:
            ax.set_ylabel("")
        ax.tick_params(labelsize=12.2)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        self._panel_label(ax, letter, model)

        ece_text = self._format_ece(self.metrics.loc[model, "ECE"])
        ax.text(0.965, 0.100, ece_text, transform=ax.transAxes, fontsize=12.8,
                ha="right", va="bottom", color="#333333",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.88))
        ax.text(0.965, 0.045, f"n = {len(df)}", transform=ax.transAxes, fontsize=11.5,
                ha="right", va="bottom", color="#666666",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.82))

    def save(self):
        if self.fig is None:
            raise RuntimeError("Call build() before save().")
        os.makedirs(self.out_pdf.parent, exist_ok=True)
        self.fig.savefig(self.out_pdf, format="pdf", bbox_inches="tight", facecolor="white")
        plt.close(self.fig)
        print(f"Saved: {self.out_pdf}")
        return str(self.out_pdf)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    SupplementaryFigure2Reliability().build().save()
