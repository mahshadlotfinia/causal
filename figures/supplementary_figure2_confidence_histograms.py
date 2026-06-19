"""
figures/supplementary_figure2_confidence_histograms.py
Created May 27, 2026

Supplementary Figure 2: Per-regime confidence distributions on MIMIC.

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


class SupplementaryFigure1ConfidenceHistograms:
    ORIGINAL_CSV = "/PATH/mimic_original_cases.csv"
    TARGET_MASK_CSV = "/PATH/mimic_target_mask_cases.csv"
    METRICS_CSV = "/PATH/all_metrics_mimic.csv"
    OUT_PDF = "/PATH/supplementary_figure2_confidence_histograms.pdf"

    MODEL_ORDER = [
        "Gemma-4-26B",
        "MedGemma-1.5-4B",
        "GPT-5",
        "Qwen3-VL-32B",
        "RAD-DINO",
        "LLaVA-Med-7B",
        "DeepSeek-R1-7B",
        "Mistral-Small-4-119B",
    ]

    CATEGORY_COLORS = {
        "Uses image": "#2166AC",
        "Ignores image": "#B2182B",
        "Unstable": "#E08214",
        "Other": "#777777",
    }
    INCORRECT_COLOR = "#9C9C9C"
    REGIME_ORDER = ["incorrect", "ungrounded-correct", "grounded-correct"]
    LETTERS = list("abcdefgh")

    def __init__(self, original_csv=None, target_mask_csv=None, metrics_csv=None, out_pdf=None):
        self.original_csv = Path(original_csv or self.ORIGINAL_CSV)
        self.target_mask_csv = Path(target_mask_csv or self.TARGET_MASK_CSV)
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
            "axes.titlesize": 16.5,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 14.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.05,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })

    @staticmethod
    def _blend_with_white(hex_color, frac_white=0.55):
        hex_color = hex_color.lstrip("#")
        rgb = np.array([int(hex_color[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0
        out = rgb * (1 - frac_white) + np.ones(3) * frac_white
        return tuple(out)

    @staticmethod
    def _short_model_name(model):
        return {
            "Gemma-4-26B": "Gemma-4-26B",
            "MedGemma-1.5-4B": "MedGemma-1.5-4B",
            "GPT-5": "GPT-5",
            "Qwen3-VL-32B": "Qwen3-VL-32B",
            "RAD-DINO": "RAD-DINO",
            "LLaVA-Med-7B": "LLaVA-Med-7B",
            "DeepSeek-R1-7B": "DeepSeek-R1-7B",
            "Mistral-Small-4-119B": "Mistral-Small-4-119B",
        }[model]

    @staticmethod
    def _fmt_mean(x):
        return "n/a" if pd.isna(x) else f"{x:.2f}"

    @staticmethod
    def _clean_axes(ax):
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    @staticmethod
    def _panel_label(ax, letter, title="", x=-0.10, y=1.05, title_offset=0.075):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=22,
                fontweight="bold", ha="left", va="bottom")
        if title:
            ax.text(x + title_offset, y, title, transform=ax.transAxes,
                    fontsize=16.0, fontweight="normal",
                    ha="left", va="bottom", color="#222222")

    def _require_columns(self, df, columns, name):
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {name}: {missing}")

    def _load_data(self):
        if not self.original_csv.exists():
            raise FileNotFoundError(self.original_csv)
        if not self.target_mask_csv.exists():
            raise FileNotFoundError(self.target_mask_csv)
        if not self.metrics_csv.exists():
            raise FileNotFoundError(self.metrics_csv)

        original = pd.read_csv(self.original_csv)
        target = pd.read_csv(self.target_mask_csv)
        metrics = pd.read_csv(self.metrics_csv)

        self._require_columns(original, ["case_id", "ground_truth"], self.original_csv.name)
        self._require_columns(target, ["case_id", "ground_truth"], self.target_mask_csv.name)
        self._require_columns(
            metrics,
            ["model", "CGR", "CGR_ci_lower", "UAR", "irrelevant_stable", "is_text_only", "is_vision_only"],
            self.metrics_csv.name,
        )

        for model in self.MODEL_ORDER:
            self._require_columns(
                original,
                [f"{model}_parsed_answer", f"{model}_confidence"],
                self.original_csv.name,
            )
            self._require_columns(target, [f"{model}_parsed_answer"], self.target_mask_csv.name)

        metrics = metrics.copy()
        metrics["category"] = metrics.apply(self._assign_category, axis=1)
        self.category = dict(zip(metrics["model"], metrics["category"]))

        self.original = original
        self.target = target
        self.regimes = self._compute_regimes()

    @staticmethod
    def _assign_category(row):
        cgr = float(row["CGR"]) * 100.0
        cgr_lo = float(row["CGR_ci_lower"]) * 100.0
        uar = float(row["UAR"]) * 100.0
        isv = float(row["irrelevant_stable"]) * 100.0
        if isv < 70.0:
            return "Unstable"
        if np.isclose(cgr, 0.0) and np.isclose(uar, 100.0) and np.isclose(isv, 100.0):
            return "Ignores image"
        if (cgr > 0.0) and (cgr_lo > 0.0) and (isv >= 90.0):
            return "Uses image"
        return "Other"

    def _compute_regimes(self):
        target_by_case = self.target.set_index("case_id")
        out = {}
        for model in self.MODEL_ORDER:
            orig_cols = ["case_id", "ground_truth", f"{model}_parsed_answer", f"{model}_confidence"]
            data = self.original[orig_cols].rename(
                columns={
                    f"{model}_parsed_answer": "orig_answer",
                    f"{model}_confidence": "confidence",
                }
            ).copy()
            targ_ans = target_by_case[f"{model}_parsed_answer"].rename("target_answer")
            data = data.merge(targ_ans, how="left", left_on="case_id", right_index=True)
            parsed = data["orig_answer"].notna() & data["confidence"].notna()
            data = data.loc[parsed].copy()
            data["confidence"] = pd.to_numeric(data["confidence"], errors="coerce").clip(0, 1)
            data = data.loc[data["confidence"].notna()].copy()

            correct = data["orig_answer"].eq(data["ground_truth"])
            grounded = correct & data["target_answer"].notna() & data["target_answer"].ne(data["orig_answer"])
            ungrounded = correct & ~grounded
            incorrect = ~correct
            out[model] = {
                "grounded-correct": data.loc[grounded, "confidence"].astype(float).to_numpy(),
                "ungrounded-correct": data.loc[ungrounded, "confidence"].astype(float).to_numpy(),
                "incorrect": data.loc[incorrect, "confidence"].astype(float).to_numpy(),
            }
        return out

    def _regime_colors(self, model):
        hue = self.CATEGORY_COLORS.get(self.category.get(model, "Other"), self.CATEGORY_COLORS["Other"])
        return {
            "grounded-correct": hue,
            "ungrounded-correct": self._blend_with_white(hue, 0.55),
            "incorrect": self.INCORRECT_COLOR,
        }

    def build(self):
        self.fig = plt.figure(figsize=(20, 11), facecolor="white")
        gs_leg = self.fig.add_gridspec(1, 1, left=0.055, right=0.985, top=0.985, bottom=0.940)
        gs = self.fig.add_gridspec(
            2, 4, left=0.060, right=0.980, top=0.885, bottom=0.070,
            wspace=0.26, hspace=0.42,
        )
        ax_leg = self.fig.add_subplot(gs_leg[0, 0])
        self._draw_legend(ax_leg)
        for idx, model in enumerate(self.MODEL_ORDER):
            ax = self.fig.add_subplot(gs[idx // 4, idx % 4])
            self._draw_panel(ax, model, self.LETTERS[idx], show_ylabel=(idx % 4 == 0))
        return self

    def _draw_legend(self, ax):
        ax.axis("off")
        handles = [
            patches.Patch(facecolor=self.CATEGORY_COLORS["Uses image"], alpha=0.55,
                          edgecolor="none", label="grounded-correct"),
            patches.Patch(facecolor=self._blend_with_white(self.CATEGORY_COLORS["Uses image"], 0.55),
                          alpha=0.70, edgecolor="none", label="ungrounded-correct"),
            patches.Patch(facecolor=self.INCORRECT_COLOR, alpha=0.55,
                          edgecolor="none", label="incorrect"),
            Line2D([0], [0], color="#333333", lw=1.8, ls="--",
                   label="dashed lines: regime mean confidence"),
            Line2D([0], [0], marker="s", lw=0, ms=12, color="white",
                   mfc=self.CATEGORY_COLORS["Uses image"], mec="none", label="panel hue: behavioral category"),
        ]
        ax.legend(handles, [h.get_label() for h in handles], ncol=5, frameon=False,
                  loc="center", bbox_to_anchor=(0.52, 0.52), columnspacing=1.6,
                  handletextpad=0.55)

    def _draw_hist(self, ax, values, bins, color, alpha, label, zorder):
        if len(values) == 0:
            return None
        ax.hist(values, bins=bins, density=True, color=color, alpha=alpha,
                edgecolor="white", linewidth=0.6, label=label, zorder=zorder)
        return float(np.nanmean(values))

    def _draw_panel(self, ax, model, letter, show_ylabel=False):
        self._clean_axes(ax)
        bins = np.linspace(0.0, 1.0, 21)
        colors = self._regime_colors(model)
        data = self.regimes[model]
        cat = self.category.get(model, "Other")
        hue = self.CATEGORY_COLORS.get(cat, self.CATEGORY_COLORS["Other"])

        y_max = 0
        for regime in ["incorrect", "ungrounded-correct", "grounded-correct"]:
            vals = data[regime]
            if len(vals) > 0:
                counts, _ = np.histogram(vals, bins=bins, density=True)
                y_max = max(y_max, float(np.nanmax(counts)))
        y_top = max(1.0, y_max * 1.30)

        # Draw incorrect first, then ungrounded-correct, then grounded-correct in front.
        means = {}
        means["incorrect"] = self._draw_hist(ax, data["incorrect"], bins, colors["incorrect"], 0.45,
                                             "incorrect", 1)
        means["ungrounded-correct"] = self._draw_hist(ax, data["ungrounded-correct"], bins,
                                                       colors["ungrounded-correct"], 0.58,
                                                       "ungrounded-correct", 2)
        means["grounded-correct"] = self._draw_hist(ax, data["grounded-correct"], bins,
                                                     colors["grounded-correct"], 0.62,
                                                     "grounded-correct", 3)

        for regime in ["incorrect", "ungrounded-correct", "grounded-correct"]:
            mean = means.get(regime)
            if mean is not None and not pd.isna(mean):
                ax.axvline(mean, color=colors[regime], lw=1.8, ls="--", zorder=4)

        n_g = len(data["grounded-correct"])
        n_u = len(data["ungrounded-correct"])
        n_i = len(data["incorrect"])
        ax.text(0.985, 0.925, f"g: {n_g}\nu: {n_u}\ni: {n_i}", transform=ax.transAxes,
                fontsize=12.2, ha="right", va="top", linespacing=1.15,
                color="#333333", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.78))

        if n_g == 0:
            ax.text(0.045, 0.745, "no grounded-correct pool", transform=ax.transAxes,
                    fontsize=12.1, ha="left", va="center", color=hue,
                    bbox=dict(boxstyle="round,pad=0.13", fc="white", ec="none", alpha=0.82))
        if model == "DeepSeek-R1-7B":
            zero_frac = np.mean(np.concatenate([data["ungrounded-correct"], data["incorrect"]]) == 0.0)
            ax.text(0.045, 0.630, f"near-degenerate:\n{zero_frac * 100:.1f}% at 0", transform=ax.transAxes,
                    fontsize=12.0, ha="left", va="center", color="#333333",
                    bbox=dict(boxstyle="round,pad=0.13", fc="white", ec="none", alpha=0.82))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, y_top)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1.0"])
        if show_ylabel:
            ax.set_ylabel("Density")
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        if letter in ["e", "f", "g", "h"]:
            ax.set_xlabel("Confidence")
        self._panel_label(ax, letter, self._short_model_name(model))

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
    SupplementaryFigure1ConfidenceHistograms().build().save()
