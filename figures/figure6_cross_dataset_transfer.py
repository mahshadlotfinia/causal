"""
figures/figure6_cross_dataset_transfer.py
Created May 27, 2026

Figure 6: Cross-dataset transfer of UAR, accuracy, baseline gaps, and testability.

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
from scipy.stats import spearmanr


class Figure6CrossDatasetTransfer:
    MIMIC_CSV = "/PATH/all_metrics_mimic.csv"
    CHEXPERT_CSV = "/PATH/all_metrics_chexpert.csv"
    PAIRWISE_CSV = "/PATH/paired_comparisons.csv"
    OUT_PDF = "/PATH/figure6_cross_dataset_transfer.pdf"

    CAT_COLORS = {
        "Uses image": "#2166AC",
        "Ignores image": "#B2182B",
        "Unstable": "#E08214",
        "Other": "#777777",
    }
    SHAPES = {
        "Multimodal": "o",
        "Text-only": "s",
        "Vision-only": "D",
    }
    PREFERRED_MODEL_ORDER = [
        "Gemma-4-26B",
        "GPT-5",
        "Qwen3-VL-32B",
        "MedGemma-27B-text",
        "RAD-DINO",
        "MedGemma-1.5-4B",
        "LLaVA-Med-7B",
        "DeepSeek-R1-7B",
        "Mistral-Small-4-119B",
    ]
    STRONG_BASELINE = "MedGemma-27B-text"

    def __init__(self, mimic_csv=None, chexpert_csv=None, pairwise_csv=None, out_pdf=None):
        self.mimic_csv = Path(mimic_csv or self.MIMIC_CSV)
        self.chexpert_csv = Path(chexpert_csv or self.CHEXPERT_CSV)
        self.pairwise_csv = Path(pairwise_csv or self.PAIRWISE_CSV)
        self.out_pdf = Path(out_pdf or self.OUT_PDF)
        self.log = logging.getLogger(self.__class__.__name__)
        self._set_rcparams()
        self._load_data()
        self.fig = None

    @staticmethod
    def _set_rcparams():
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 19.5,
            "axes.labelsize": 18.5,
            "axes.titlesize": 20.5,
            "xtick.labelsize": 15.5,
            "ytick.labelsize": 15.5,
            "legend.fontsize": 16.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.15,
            "xtick.major.width": 1.05,
            "ytick.major.width": 1.05,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })

    @staticmethod
    def _clean_axes(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    @staticmethod
    def _panel_label(ax, letter, title, x=-0.10, y=1.030, title_offset=0.045):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=28, fontweight="bold",
                ha="left", va="bottom")
        ax.text(x + title_offset, y, title, transform=ax.transAxes, fontsize=20.5,
                fontweight="normal", ha="left", va="bottom")

    @staticmethod
    def _fmt_model(model):
        return model

    @staticmethod
    def _fmt_rho(rho):
        return f"{rho:.3f}"

    @staticmethod
    def _fmt_p(p):
        if pd.isna(p):
            return "p = n/a"
        if p < 0.001:
            return "p < 0.001"
        return f"p = {p:.3f}"

    @staticmethod
    def _fmt_diff(x):
        return f"{x:+.1f}"

    def _require_columns(self, df, cols, name):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {name}: {missing}")

    def _load_data(self):
        mimic = pd.read_csv(self.mimic_csv)
        chex = pd.read_csv(self.chexpert_csv)
        pw = pd.read_csv(self.pairwise_csv)

        base_cols = ["model", "accuracy", "accuracy_ci_lower", "accuracy_ci_upper",
                     "UAR", "UAR_ci_lower", "UAR_ci_upper", "is_text_only", "is_vision_only"]
        self._require_columns(mimic, base_cols + ["CGR", "CGR_ci_lower", "irrelevant_stable"], self.mimic_csv.name)
        self._require_columns(chex, base_cols, self.chexpert_csv.name)
        self._require_columns(pw, ["comparison_type", "dataset", "model_a", "model_b", "metric",
                                  "diff", "diff_ci_lower", "diff_ci_upper", "p_value_fdr"],
                              self.pairwise_csv.name)

        for df in [mimic, chex]:
            for col in ["accuracy", "accuracy_ci_lower", "accuracy_ci_upper",
                        "UAR", "UAR_ci_lower", "UAR_ci_upper"]:
                df[col] = df[col].astype(float) * 100.0

        for col in ["CGR", "CGR_ci_lower", "irrelevant_stable"]:
            mimic[col] = mimic[col].astype(float) * 100.0
        for col in ["diff", "diff_ci_lower", "diff_ci_upper"]:
            pw[col] = pw[col].astype(float) * 100.0

        # Defragment after repeated column-wise updates.
        mimic = mimic.copy()
        chex = chex.copy()

        mimic_modality = np.select(
            [mimic["is_text_only"].astype(bool), mimic["is_vision_only"].astype(bool)],
            ["Text-only", "Vision-only"],
            default="Multimodal",
        )
        chex_modality = np.select(
            [chex["is_text_only"].astype(bool), chex["is_vision_only"].astype(bool)],
            ["Text-only", "Vision-only"],
            default="Multimodal",
        )

        mimic_category = mimic.apply(self._assign_mimic_category, axis=1)
        order_map = {m: i for i, m in enumerate(self.PREFERRED_MODEL_ORDER)}

        mimic = mimic.assign(
            modality=mimic_modality,
            category=mimic_category,
            order_ix=mimic["model"].map(order_map),
        )

        chex = chex.assign(
            modality=chex_modality,
            order_ix=chex["model"].map(order_map),
        )

        # Keep stored dataframes defragmented.
        mimic = mimic.copy()
        chex = chex.copy()

        merged = mimic[["model", "accuracy", "accuracy_ci_lower", "accuracy_ci_upper",
                        "UAR", "UAR_ci_lower", "UAR_ci_upper", "category", "modality", "order_ix"]].merge(
            chex[["model", "accuracy", "accuracy_ci_lower", "accuracy_ci_upper",
                  "UAR", "UAR_ci_lower", "UAR_ci_upper", "modality"]],
            on="model", suffixes=("_mimic", "_chexpert"), validate="one_to_one")
        merged = merged.sort_values("order_ix").reset_index(drop=True)

        self.mimic = mimic
        self.chex = chex
        self.pw = pw
        self.df = merged

    def _assign_mimic_category(self, row):
        cgr = row["CGR"]
        cgr_lo = row["CGR_ci_lower"]
        uar = row["UAR"]
        isv = row["irrelevant_stable"]
        if pd.isna(cgr) or pd.isna(cgr_lo) or pd.isna(uar) or pd.isna(isv):
            return "Other"
        if isv < 70.0:
            return "Unstable"
        if np.isclose(cgr, 0.0) and np.isclose(uar, 100.0) and np.isclose(isv, 100.0):
            return "Ignores image"
        if (cgr > 0.0) and (cgr_lo > 0.0) and (isv >= 90.0):
            return "Uses image"
        return "Other"

    def _color(self, row_or_category):
        cat = row_or_category if isinstance(row_or_category, str) else row_or_category["category"]
        return self.CAT_COLORS.get(cat, self.CAT_COLORS["Other"])

    def _shape(self, row):
        return self.SHAPES.get(row["modality_mimic"], "o")

    def _row(self, model):
        rows = self.df.loc[self.df["model"] == model]
        if rows.empty:
            raise KeyError(model)
        return rows.iloc[0]

    def _spearman(self, metric, exclude_rad=False):
        data = self.df.copy()
        if exclude_rad:
            data = data.loc[data["model"] != "RAD-DINO"]
        rho, p = spearmanr(data[f"{metric}_mimic"], data[f"{metric}_chexpert"])
        return float(rho), float(p)

    def build(self):
        self.fig = plt.figure(figsize=(18.0, 18.0), facecolor="white")
        gs_leg = self.fig.add_gridspec(1, 1, left=0.045, right=0.990, top=0.992, bottom=0.925)

        # Panels moved upward; legend unchanged.
        gs_r1 = self.fig.add_gridspec(1, 2, left=0.085, right=0.975, top=0.895, bottom=0.585,
                                      wspace=0.30)
        gs_r2 = self.fig.add_gridspec(1, 1, left=0.115, right=0.975, top=0.515, bottom=0.315)
        gs_r3 = self.fig.add_gridspec(1, 1, left=0.060, right=0.985, top=0.255, bottom=0.080)

        ax_leg = self.fig.add_subplot(gs_leg[0, 0])
        ax_a = self.fig.add_subplot(gs_r1[0, 0])
        ax_b = self.fig.add_subplot(gs_r1[0, 1])
        ax_c = self.fig.add_subplot(gs_r2[0, 0])
        ax_d = self.fig.add_subplot(gs_r3[0, 0])

        self._draw_legend(ax_leg)
        self._panel_a(ax_a)
        self._panel_b(ax_b)
        self._panel_c(ax_c)
        self._panel_d(ax_d)
        return self

    def _draw_legend(self, ax):
        ax.axis("off")
        cat_handles = [
            Line2D([0], [0], marker="o", lw=0, markersize=16,
                   mfc=self.CAT_COLORS[c], mec="white", mew=0.8, label=c)
            for c in ["Uses image", "Ignores image", "Unstable"]
        ]
        mod_handles = [
            Line2D([0], [0], marker=m, lw=0, markersize=16,
                   mfc="#777777", mec="white", mew=0.8, label=label)
            for label, m in self.SHAPES.items()
        ]
        endpoint_handles = [
            Line2D([0], [0], marker="o", lw=1.3, markersize=12,
                   mfc="white", mec="#444444", color="#AAAAAA", label="MIMIC endpoint"),
            Line2D([0], [0], marker="o", lw=1.3, markersize=12,
                   mfc="#444444", mec="white", color="#AAAAAA", label="CheXpert endpoint"),
        ]
        leg1 = ax.legend(cat_handles, [h.get_label() for h in cat_handles],
                         ncol=3, frameon=False, loc="center", bbox_to_anchor=(0.15, 0.70),
                         title="Behavioral category", title_fontsize=16.5,
                         columnspacing=1.4, handletextpad=0.45)
        ax.add_artist(leg1)
        leg2 = ax.legend(mod_handles, [h.get_label() for h in mod_handles],
                         ncol=3, frameon=False, loc="center", bbox_to_anchor=(0.83, 0.70),
                         title="Modality", title_fontsize=16.5,
                         columnspacing=1.4, handletextpad=0.45)
        ax.add_artist(leg2)

        leg3 = ax.legend(endpoint_handles, [h.get_label() for h in endpoint_handles],
                         ncol=2, frameon=False, loc="center", bbox_to_anchor=(0.50, 0.70),
                         title="Panel c endpoints", title_fontsize=16.0,
                         columnspacing=1.2, handletextpad=0.45)
        ax.add_artist(leg3)

    def _scatter_with_ci(self, ax, row, xmetric, ymetric, special_rad_open=False):
        x = row[f"{xmetric}_mimic"]
        y = row[f"{ymetric}_chexpert"]
        col = self._color(row)
        marker = self._shape(row)
        xlo = row[f"{xmetric}_ci_lower_mimic"]
        xhi = row[f"{xmetric}_ci_upper_mimic"]
        ylo = row[f"{ymetric}_ci_lower_chexpert"]
        yhi = row[f"{ymetric}_ci_upper_chexpert"]
        ax.errorbar(x, y, xerr=[[x - xlo], [xhi - x]], yerr=[[y - ylo], [yhi - y]],
                    fmt="none", ecolor=col, elinewidth=1.25, capsize=3, alpha=0.95, zorder=2)
        if special_rad_open and row["model"] == "RAD-DINO":
            ax.scatter(x, y, s=165, marker=marker, facecolor="white", edgecolor=col,
                       linewidth=2.0, zorder=4)
        else:
            ax.scatter(x, y, s=145, marker=marker, color=col, edgecolor="white",
                       linewidth=0.8, zorder=3)

    def _panel_a(self, ax):
        self._clean_axes(ax)
        ax.set_xlim(70, 103)
        ax.set_ylim(70, 103)
        ax.set_aspect("equal", adjustable="box")
        ax.plot([70, 103], [70, 103], color="#888888", lw=1.1, ls="--", zorder=0)
        ax.add_patch(patches.Rectangle((73, 73), 14, 14, facecolor="#2166AC", alpha=0.05,
                                       edgecolor="none", zorder=0))

        for _, row in self.df.iterrows():
            self._scatter_with_ci(ax, row, "UAR", "UAR")

        mistral = self._row("Mistral-Small-4-119B")
        ax.annotate("Mistral UAR rose\non CheXpert",
                    xy=(mistral["UAR_mimic"], mistral["UAR_chexpert"]),
                    xytext=(83.5, 98.0), textcoords="data", fontsize=13.8,
                    ha="left", va="center", color="#333333",
                    arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8,
                                    shrinkA=0, shrinkB=6))

        ax.text(99.2, 100.8, "three ignores-image\nsystems at (100,100)", fontsize=13.5,
                ha="right", va="bottom", color=self.CAT_COLORS["Ignores image"])
        ax.text(73.6, 86.3, "uses-image\ntransfer band", fontsize=13.5,
                ha="left", va="top", color=self.CAT_COLORS["Uses image"], alpha=0.85)

        rho_all, p_all = self._spearman("UAR", exclude_rad=False)
        rho_ex, p_ex = self._spearman("UAR", exclude_rad=True)
        # ax.text(71.0, 101.9,
        #         f"Spearman rho = {self._fmt_rho(rho_all)} (all 9)\n"
        #         f"rho = {self._fmt_rho(rho_ex)} (ex-RAD-DINO)",
        #         fontsize=13.7, ha="left", va="top", color="#333333",
        #         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

        ax.set_xlabel("MIMIC UAR (%)")
        ax.set_ylabel("CheXpert UAR (%)")
        self._panel_label(ax, "a", "UAR transfer preserves the split", x=-0.13, y=1.035, title_offset=0.060)

    def _panel_b(self, ax):
        self._clean_axes(ax)
        ax.set_xlim(40, 75)
        ax.set_ylim(40, 75)
        ax.set_aspect("equal", adjustable="box")
        ax.plot([40, 75], [40, 75], color="#888888", lw=1.1, ls="--", zorder=0)

        for _, row in self.df.iterrows():
            self._scatter_with_ci(ax, row, "accuracy", "accuracy", special_rad_open=True)

        rad = self._row("RAD-DINO")
        x, y = rad["accuracy_mimic"], rad["accuracy_chexpert"]
        ax.plot([x, x], [x, y], color=self.CAT_COLORS["Uses image"], lw=1.4, alpha=0.75, zorder=1)
        ax.annotate("RAD-DINO\nin-distribution",
                    xy=(x, y), xytext=(62.2, 72.2), textcoords="data",
                    fontsize=13.8, ha="left", va="center", color="#333333",
                    bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="#DDDDDD", lw=0.8),
                    arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8,
                                    shrinkA=0, shrinkB=6))
        ax.text(x + 0.5, (x + y) / 2, "+12.6 pp", fontsize=13.2,
                ha="left", va="center", color=self.CAT_COLORS["Uses image"])

        rho_all, p_all = self._spearman("accuracy", exclude_rad=False)
        rho_ex, p_ex = self._spearman("accuracy", exclude_rad=True)
        # ax.text(41.0, 73.6,
        #         f"rho = {self._fmt_rho(rho_all)}, {self._fmt_p(p_all)} (all 9)\n"
        #         f"rho = {self._fmt_rho(rho_ex)}, {self._fmt_p(p_ex)} (ex-RAD-DINO)",
        #         fontsize=13.5, ha="left", va="top", color="#333333",
        #         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

        ax.set_xlabel("MIMIC accuracy (%)")
        ax.set_ylabel("CheXpert accuracy (%)")
        self._panel_label(ax, "b", "Accuracy transfer with one explained outlier", x=-0.13, y=1.035, title_offset=0.060)

    def _gap_row(self, dataset, model):
        rows = self.pw[
            (self.pw["comparison_type"] == "vs_text_baseline") &
            (self.pw["dataset"] == dataset) &
            (self.pw["metric"] == "accuracy") &
            (self.pw["model_a"] == model) &
            (self.pw["model_b"] == self.STRONG_BASELINE)
        ]
        if rows.empty:
            return None
        return rows.iloc[0]

    def _panel_c(self, ax):
        self._clean_axes(ax)
        models = ["RAD-DINO", "Gemma-4-26B", "GPT-5", "MedGemma-1.5-4B",
                  "Qwen3-VL-32B", "Mistral-Small-4-119B"]
        rows = []
        for model in models:
            mim = self._gap_row("mimic", model)
            chx = self._gap_row("chexpert", model)
            if mim is None or chx is None:
                self.log.warning("Skipping gap row for %s", model)
                continue
            rows.append((model, mim, chx))

        y = np.arange(len(rows))
        ax.axvline(0, color="#333333", lw=1.15, zorder=0)
        ax.text(0.35, -0.62, "strong text-only baseline", fontsize=13.5,
                ha="left", va="bottom", color="#333333")
        ax.text(-22.2, 0.95, "hollow: MIMIC   filled: CheXpert\nblack ring: p < 0.05",
                fontsize=12.6, ha="left", va="bottom", color="#333333",
                bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.88))

        for i, (model, mim, chx) in enumerate(rows):
            cat = self._row(model)["category"]
            col = self._color(cat)
            x1 = float(mim["diff"])
            x2 = float(chx["diff"])
            if model == "MedGemma-1.5-4B":
                ax.axhspan(i - 0.42, i + 0.42, color=col, alpha=0.08, zorder=0)
            ax.plot([x1, x2], [i, i], color="#777777", lw=1.9, zorder=1)

            # Significance rings, avoiding stars or p-value clutter.
            for x, row_stat, is_chex in [(x1, mim, False), (x2, chx, True)]:
                ring_col = "#111111" if float(row_stat["p_value_fdr"]) < 0.05 else "#BBBBBB"
                ax.scatter(x, i, s=210, facecolor="none", edgecolor=ring_col,
                           linewidth=1.4, zorder=2.7)
            ax.scatter(x1, i, s=125, facecolor="white", edgecolor=col, linewidth=2.0, zorder=3)
            ax.scatter(x2, i, s=135, facecolor=col, edgecolor="white", linewidth=0.8, zorder=4)
            label_dy = -0.20  # good default for most labels

            dy1 = label_dy
            dy2 = label_dy

            # Special case: both dots are on the right side of the zero line.
            # Keep one label as-is, move the other with the opposite y-offset.
            if x1 > 0 and x2 > 0:
                dy1 = label_dy
                dy2 = -label_dy +0.35

            ax.text(x1, i + dy1, self._fmt_diff(x1), fontsize=12.8,
                    ha="center", va="bottom", color="#333333")
            ax.text(x2, i + dy2, self._fmt_diff(x2), fontsize=12.8,
                    ha="center", va="bottom", color="#333333")

        med_i = [i for i, (m, _, _) in enumerate(rows) if m == "MedGemma-1.5-4B"][0]
        ax.annotate("MedGemma-1.5-4B\nreverses sign",
                    xy=(float(self._gap_row("chexpert", "MedGemma-1.5-4B")["diff"]), med_i),
                    xytext=(13.2, med_i - 0.85), textcoords="data", fontsize=13.5,
                    ha="left", va="center", color="#333333",
                    arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8,
                                    shrinkA=0, shrinkB=6))

        ax.set_xlim(-23, 20)
        ax.set_ylim(-0.75, len(rows) - 0.25)
        ax.set_yticks(y)
        ax.set_yticklabels([m for m, _, _ in rows], fontsize=13.7)
        ax.invert_yaxis()
        ax.set_xlabel("Accuracy difference vs MedGemma-27B-text (pp)")
        self._panel_label(ax, "c", "Baseline gap transfer", x=-0.075, y=1.040, title_offset=0.038)

    def _panel_d(self, ax):
        ax.axis("off")
        self._panel_label(ax, "d", "CheXpert testability ledger", x=0.000, y=1.035, title_offset=0.032)

        order = self.PREFERRED_MODEL_ORDER
        x_glyph = 0.025
        x_model = 0.065
        x_cat = 0.365
        x_uar = 0.545
        x_cons = 0.690
        x_test = 0.845
        y0 = 0.86
        row_h = 0.082

        headers = [
            (x_model, "system"), (x_cat, "MIMIC category"), (x_uar, "CheXpert UAR"),
            (x_cons, "UAR consistency"), (x_test, "CheXpert testability"),
        ]
        for x, txt in headers:
            ax.text(x, 0.965, txt, transform=ax.transAxes, fontsize=14.8,
                    ha="left", va="center", color="#222222")
        ax.plot([0.015, 0.985], [0.925, 0.925], transform=ax.transAxes,
                color="#DDDDDD", lw=0.9)

        for idx, model in enumerate(order):
            row = self._row(model)
            y = y0 - idx * row_h
            cat = row["category"]
            col = self._color(cat)
            uar = row["UAR_chexpert"]
            marker = self.SHAPES[row["modality_mimic"]]
            ax.scatter(x_glyph, y, transform=ax.transAxes, s=86, marker=marker,
                       color=col, edgecolor="white", linewidth=0.7, zorder=3)
            ax.text(x_model, y, model, transform=ax.transAxes, fontsize=13.7,
                    ha="left", va="center", color="#222222")
            # category color block
            ax.add_patch(patches.FancyBboxPatch(
                (x_cat, y - 0.025), 0.135, 0.050, transform=ax.transAxes,
                boxstyle="round,pad=0.006,rounding_size=0.010",
                facecolor=col, edgecolor="none", alpha=0.95))
            ax.text(x_cat + 0.006, y, cat, transform=ax.transAxes, fontsize=12.5,
                    ha="left", va="center", color="white")
            ax.text(x_uar, y, f"{uar:.1f}%", transform=ax.transAxes, fontsize=13.4,
                    ha="left", va="center", color="#222222")
            if cat == "Ignores image":
                consistent = "consistent: UAR = 100"
            elif cat == "Uses image":
                consistent = "consistent: UAR < 100"
            elif cat == "Unstable":
                consistent = "partial: UAR only"
            else:
                consistent = "not classified"
            cons_color = col if cat in ["Uses image", "Ignores image", "Unstable"] else "#777777"
            ax.text(x_cons, y, consistent, transform=ax.transAxes, fontsize=12.8,
                    ha="left", va="center", color=cons_color)
            if cat == "Unstable":
                testability = "UAR yes; CGR/IS no"
            else:
                testability = "UAR yes; CGR/IS no"
            ax.text(x_test, y, testability, transform=ax.transAxes, fontsize=12.8,
                    ha="left", va="center", color="#333333")
            if idx < len(order) - 1:
                ax.plot([0.015, 0.985], [y - row_h / 2, y - row_h / 2], transform=ax.transAxes,
                        color="#EEEEEE", lw=0.7)

        # ax.text(0.845, 0.075, "no CheXpert bounding boxes: CGR/IS cannot be re-derived",
        #         transform=ax.transAxes, fontsize=12.8, ha="left", va="center", color="#333333",
        #         style="italic")

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
    Figure6CrossDatasetTransfer().build().save()
