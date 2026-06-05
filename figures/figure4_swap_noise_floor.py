"""
figures/figure4_swap_noise_floor.py
Created May 27, 2026

Figure 4: Swap-invariant decisions, irrelevant-mask noise floor, and UAR departures.

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


class Figure4SwapNoiseFloor:
    MAIN_CSV = "/PATH/all_metrics_mimic.csv"
    PAIRWISE_CSV = "/PATH/paired_comparisons.csv"
    OUT_PDF = "/PATH/figure4_swap_noise_floor.pdf"

    CAT_COLORS = {
        "Uses image": "#2166AC",
        "Ignores image": "#B2182B",
        "Unstable": "#E08214",
        "Other": "#777777",
    }
    BASELINE_COLORS = {
        "MedGemma-27B-text": "#2166AC",
        "DeepSeek-R1-7B": "#92C5DE",
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
    TEXT_BASELINES = ["MedGemma-27B-text", "DeepSeek-R1-7B"]

    def __init__(self, main_csv=None, pairwise_csv=None, out_pdf=None):
        self.main_csv = Path(main_csv or self.MAIN_CSV)
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
            "ytick.labelsize": 15.0,
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
    def _fmt_p(p):
        if pd.isna(p):
            return "p = n/a"
        if p < 0.001:
            return "p < 0.001"
        return f"p = {p:.3f}"

    @staticmethod
    def _fmt_model(model):
        return model

    @staticmethod
    def _panel_label(ax, letter, title, x=-0.10, y=1.030, title_offset=0.040):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=28, fontweight="bold",
                ha="left", va="bottom")
        ax.text(x + title_offset, y, title, transform=ax.transAxes, fontsize=20.5,
                fontweight="normal", ha="left", va="bottom")

    @staticmethod
    def _clean_axes(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    def _require_columns(self, df, columns, name):
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {name}: {missing}")

    def _load_data(self):
        df = pd.read_csv(self.main_csv)
        pw = pd.read_csv(self.pairwise_csv)
        self._require_columns(
            df,
            ["model", "UAR", "UAR_ci_lower", "UAR_ci_upper",
             "irrelevant_stable", "irrelevant_stable_ci_lower", "irrelevant_stable_ci_upper",
             "CGR", "CGR_ci_lower", "CGR_ci_upper", "is_text_only", "is_vision_only"],
            self.main_csv.name,
        )
        self._require_columns(
            pw,
            ["comparison_type", "dataset", "model_a", "model_b", "metric",
             "diff", "diff_ci_lower", "diff_ci_upper", "p_value_fdr", "n_shared"],
            self.pairwise_csv.name,
        )

        pct_cols = [
            "UAR", "UAR_ci_lower", "UAR_ci_upper",
            "irrelevant_stable", "irrelevant_stable_ci_lower", "irrelevant_stable_ci_upper",
            "CGR", "CGR_ci_lower", "CGR_ci_upper",
        ]
        for col in pct_cols:
            df[col] = df[col].astype(float) * 100.0
        for col in ["diff", "diff_ci_lower", "diff_ci_upper"]:
            pw[col] = pw[col].astype(float) * 100.0

        # Defragment after repeated column-wise updates.
        df = df.copy()

        modality = np.select(
            [df["is_text_only"].astype(bool), df["is_vision_only"].astype(bool)],
            ["Text-only", "Vision-only"],
            default="Multimodal",
        )

        category = df.apply(self._assign_category, axis=1)
        noise_floor = 100.0 - df["irrelevant_stable"]

        df = df.assign(
            modality=modality,
            category=category,
            image_contingent=100.0 - df["UAR"],
            noise_floor=noise_floor,
            GSP=df["CGR"] - noise_floor,
            order_ix=df["model"].map({m: i for i, m in enumerate(self.PREFERRED_MODEL_ORDER)}),
        )

        # Keep stored dataframe defragmented too.
        self.df = df.copy()
        self.pw = pw

    def _assign_category(self, row):
        cgr, cgr_lo, uar, isv = row["CGR"], row["CGR_ci_lower"], row["UAR"], row["irrelevant_stable"]
        if pd.isna(cgr) or pd.isna(cgr_lo) or pd.isna(uar) or pd.isna(isv):
            return "Other"
        if isv < 70.0:
            return "Unstable"
        if np.isclose(cgr, 0.0) and np.isclose(uar, 100.0) and np.isclose(isv, 100.0):
            return "Ignores image"
        if (cgr > 0.0) and (cgr_lo > 0.0) and (isv >= 90.0):
            return "Uses image"
        return "Other"

    def _row(self, model):
        rows = self.df.loc[self.df["model"] == model]
        if rows.empty:
            raise KeyError(model)
        return rows.iloc[0]

    def _color(self, category):
        return self.CAT_COLORS.get(category, self.CAT_COLORS["Other"])

    def _draw_legend(self, ax):
        ax.axis("off")
        cat_handles = [
            Line2D([0], [0], marker="o", lw=0, markersize=16,
                   mfc=self.CAT_COLORS[c], mec="white", mew=0.8, label=c)
            for c in ["Uses image", "Ignores image", "Unstable"]
        ]
        segment_handles = [
            patches.Patch(facecolor="#888888", edgecolor="#777777", alpha=0.30,
                          hatch="///", label="swap-invariant" ),
            patches.Patch(facecolor="#888888", edgecolor="none", alpha=0.95,
                          label="swap-flipped"),
        ]
        baseline_handles = [
            Line2D([0], [0], marker="o", lw=1.4, markersize=10,
                   color=self.BASELINE_COLORS["MedGemma-27B-text"],
                   mfc=self.BASELINE_COLORS["MedGemma-27B-text"], mec="white", mew=0.6,
                   label="vs strong text" ),
            Line2D([0], [0], marker="o", lw=1.4, markersize=10,
                   color=self.BASELINE_COLORS["DeepSeek-R1-7B"],
                   mfc=self.BASELINE_COLORS["DeepSeek-R1-7B"], mec="white", mew=0.6,
                   label="vs weak text" ),
        ]
        ax.text(0.015, 0.66, "Behavioral category", transform=ax.transAxes,
                fontsize=16.5, ha="left", va="center")
        ax.text(0.015, 0.24, "Segment / paired baseline", transform=ax.transAxes,
                fontsize=16.5, ha="left", va="center")
        leg1 = ax.legend(cat_handles, [h.get_label() for h in cat_handles],
                         ncol=3, frameon=False, loc="center", bbox_to_anchor=(0.47, 0.66),
                         columnspacing=1.9, handletextpad=0.50)
        ax.add_artist(leg1)
        ax.legend(segment_handles + baseline_handles,
                  [h.get_label() for h in segment_handles + baseline_handles],
                  ncol=4, frameon=False, loc="center", bbox_to_anchor=(0.62, 0.24),
                  columnspacing=1.15, handletextpad=0.48)

    def build(self):
        self.fig = plt.figure(figsize=(18.0, 19.6), facecolor="white")
        gs_leg = self.fig.add_gridspec(1, 1, left=0.045, right=0.990, top=0.990, bottom=0.936)
        gs_r1 = self.fig.add_gridspec(1, 1, left=0.110, right=0.975, top=0.885, bottom=0.655)
        gs_r2 = self.fig.add_gridspec(1, 2, left=0.105, right=0.975, top=0.575, bottom=0.355,
                                      wspace=0.30)
        gs_r3 = self.fig.add_gridspec(1, 1, left=0.125, right=0.975, top=0.270, bottom=0.060)

        ax_leg = self.fig.add_subplot(gs_leg[0, 0])
        ax_a = self.fig.add_subplot(gs_r1[0, 0])
        ax_b = self.fig.add_subplot(gs_r2[0, 0])
        ax_c = self.fig.add_subplot(gs_r2[0, 1])
        ax_d = self.fig.add_subplot(gs_r3[0, 0])

        self._draw_legend(ax_leg)
        self._panel_a(ax_a)
        self._panel_b(ax_b)
        self._panel_c(ax_c)
        self._panel_d(ax_d)
        return self

    def _panel_a(self, ax):
        self._clean_axes(ax)
        data = self.df.loc[self.df["UAR"].notna()].copy()
        data = data.sort_values(["image_contingent", "order_ix"], ascending=[False, True]).reset_index(drop=True)
        y = np.arange(len(data))
        height = 0.62

        for i, row in data.iterrows():
            col = self._color(row["category"])
            prior = row["UAR"]
            contingent = row["image_contingent"]
            ax.barh(i, prior, height=height, left=0, color=col, alpha=0.25,
                    edgecolor=col, linewidth=0.7, hatch="///", zorder=2)
            ax.barh(i, contingent, height=height, left=prior, color=col, alpha=0.92,
                    edgecolor="none", zorder=3)
            if contingent > 0.3:
                x_text = min(100.6, prior + contingent + 0.7)
                ax.text(x_text, i, f"{contingent:.1f}", fontsize=14.6, color="#333333",
                        ha="left", va="center")

        ax.axvline(100, color="#333333", lw=1.0, ls="--", zorder=1)
        ax.text(100.3, -0.62, "100% ceiling", fontsize=14.2, ha="left", va="bottom",
                color="#333333")
        uses = data.loc[data["category"] == "Uses image", "image_contingent"]
        if not uses.empty:
            ax.text(67.5, 0.35,
                    f"uses-image systems:\nimage-contingent {uses.min():.1f}-{uses.max():.1f}%",
                    fontsize=14.5, ha="left", va="center", color="#333333",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#DDDDDD", lw=0.8))

        ax.set_xlim(0, 106.5)
        ax.set_ylim(-0.75, len(data) - 0.25)
        ax.set_yticks(y)
        ax.set_yticklabels([self._fmt_model(m) for m in data["model"]], fontsize=14.2)
        ax.invert_yaxis()
        ax.set_xlabel("Share of correct-on-original answers (%)")
        self._panel_label(ax, "a", "Correct answers split into prior-reachable and image-contingent fractions",
                          x=-0.085, y=1.025, title_offset=0.038)

    def _panel_b(self, ax):
        self._clean_axes(ax)
        models = ["RAD-DINO", "Gemma-4-26B", "MedGemma-1.5-4B", "GPT-5", "Qwen3-VL-32B", "Mistral-Small-4-119B"]
        data = self.df.set_index("model").loc[models].reset_index()
        y = np.arange(len(data))
        rad_is = float(self._row("RAD-DINO")["irrelevant_stable"])
        rad_lo = float(self._row("RAD-DINO")["irrelevant_stable_ci_lower"])
        rad_hi = float(self._row("RAD-DINO")["irrelevant_stable_ci_upper"])

        ax.axvspan(50, 70, color=self.CAT_COLORS["Unstable"], alpha=0.07, zorder=0)
        ax.axvline(rad_is, color="#333333", lw=1.2, ls="--", zorder=1)
        ax.axvspan(rad_lo, rad_hi, color="#333333", alpha=0.06, zorder=0)

        for i, row in data.iterrows():
            col = self._color(row["category"])
            x = row["irrelevant_stable"]
            lo = row["irrelevant_stable_ci_lower"]
            hi = row["irrelevant_stable_ci_upper"]
            ax.errorbar(x, i, xerr=[[x - lo], [hi - x]], fmt="none",
                        ecolor=col, elinewidth=1.45, capsize=3, zorder=2)
            ax.scatter(x, i, s=130, marker=self.SHAPES[row["modality"]], color=col,
                       edgecolor="white", linewidth=0.8, zorder=3)
            ax.text(min(101.2, x + 1.0), i - 0.17, f"{x:.1f}", fontsize=13.5,
                    ha="left", va="bottom", color="#333333")

        ax.text(rad_is - 3.6, 0.55, "vision-only\nnoise floor", fontsize=13.6,
                ha="right", va="center", color="#333333")
        ax.text(53.0, 3.35, "unstable\nregion", fontsize=13.5,
                ha="left", va="top", color=self.CAT_COLORS["Unstable"])
        ax.text(80.7, 4.45, "multimodal cluster\nleft of noise floor", fontsize=13.4,
                ha="left", va="center", color="#333333")

        ax.set_xlim(50, 102)
        ax.set_ylim(-0.75, len(data) - 0.25)
        ax.set_yticks(y)
        ax.set_yticklabels([self._fmt_model(m) for m in data["model"]], fontsize=13.4)
        ax.invert_yaxis()
        ax.set_xlabel("Irrelevant-mask stability, IS (%)")
        self._panel_label(ax, "b", "Irrelevant-mask stability resolves the noise floor",
                          x=-0.19, y=1.035, title_offset=0.070)

    def _panel_c(self, ax):
        self._clean_axes(ax)
        models = ["Gemma-4-26B", "MedGemma-1.5-4B", "GPT-5", "Qwen3-VL-32B", "RAD-DINO", "Mistral-Small-4-119B"]
        data = self.df.set_index("model").loc[models].reset_index()
        data = data.sort_values(["GSP", "order_ix"], ascending=[False, True]).reset_index(drop=True)
        y = np.arange(len(data))

        for i, row in data.iterrows():
            col = self._color(row["category"])
            nf = row["noise_floor"]
            cgr = row["CGR"]
            ax.plot([nf, cgr], [i, i], color=col, lw=3.0, alpha=0.95, zorder=2)
            ax.scatter(nf, i, s=115, marker="o", facecolor="white", edgecolor=col,
                       linewidth=1.8, zorder=3)
            ax.scatter(cgr, i, s=125, marker=self.SHAPES[row["modality"]], color=col,
                       edgecolor="white", linewidth=0.8, zorder=4)
            label_x = max(nf, cgr) + 1.0
            ax.text(label_x, i, f"GSP {row['GSP']:+.1f}", fontsize=13.2,
                    ha="left", va="center", color="#333333")

        ax.axvline(0, color="#BBBBBB", lw=0.9, zorder=0)
        ax.text(0.8, 0.55, "open marker: 100 - IS\nfilled marker: CGR", fontsize=13.3,
                ha="left", va="center", color="#333333",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.88))
        ax.text(31.8, 4.05, "Mistral: target\nsensitivity below\nnoise floor",
                fontsize=13.0, ha="left", va="center", color=self.CAT_COLORS["Unstable"])
        ax.set_xlim(-1, 47)
        ax.set_ylim(-0.75, len(data) - 0.25)
        ax.set_yticks(y)
        ax.set_yticklabels([self._fmt_model(m) for m in data["model"]], fontsize=13.4)
        ax.invert_yaxis()
        ax.set_xlabel("CGR and irrelevant-occlusion noise floor (%)")
        self._panel_label(ax, "c", "Target sensitivity vs noise floor",
                          x=-0.18, y=1.035, title_offset=0.070)

    def _uar_forest_rows(self):
        # Main-text panel focuses on non-text multimodal systems with negative, significant
        # departures from perfect UAR. RAD-DINO has no paired UAR rows and is omitted.
        rows = self.pw[
            (self.pw["comparison_type"] == "vs_text_baseline") &
            (self.pw["dataset"] == "mimic") &
            (self.pw["metric"] == "UAR") &
            (self.pw["model_b"].isin(self.TEXT_BASELINES))
        ].copy()
        rows = rows[(rows["diff"] < 0) & (rows["p_value_fdr"] < 0.05)].copy()
        rows = rows[rows["model_a"] != "RAD-DINO"].copy()
        avg = rows.groupby("model_a")["diff"].mean().sort_values()
        order = avg.index.tolist()
        rows["model_order"] = rows["model_a"].map({m: i for i, m in enumerate(order)})
        rows["baseline_order"] = rows["model_b"].map({"MedGemma-27B-text": 0, "DeepSeek-R1-7B": 1})
        return rows.sort_values(["model_order", "baseline_order"]).reset_index(drop=True), order

    def _panel_d(self, ax):
        self._clean_axes(ax)
        rows, order = self._uar_forest_rows()
        y_base = {m: i for i, m in enumerate(order)}
        offsets = {"MedGemma-27B-text": -0.16, "DeepSeek-R1-7B": 0.16}

        ax.axvline(0, color="#333333", lw=1.15, zorder=0)
        for _, row in rows.iterrows():
            y = y_base[row["model_a"]] + offsets[row["model_b"]]
            col = self.BASELINE_COLORS[row["model_b"]]
            x = row["diff"]
            lo = row["diff_ci_lower"]
            hi = row["diff_ci_upper"]
            ax.errorbar(x, y, xerr=[[x - lo], [hi - x]], fmt="none",
                        ecolor=col, elinewidth=1.45, capsize=3, zorder=2)
            ax.scatter(x, y, s=115, color=col, edgecolor="white", linewidth=0.8, zorder=3)
            # Default label position
            dx = 0.0
            dy = -0.36

            # Move only the two bottom light-blue labels.
            # Light blue = vs DeepSeek-R1-7B baseline.
            if row["model_b"] == "DeepSeek-R1-7B" and row["model_a"] == order[-1]:
                dx = 0.0      # bottom light-blue: + moves right, - moves left
                dy = 0.2    # more negative moves visually up

            elif row["model_b"] == "DeepSeek-R1-7B" and row["model_a"] == order[-2]:
                dx = 0.0      # light-blue above bottom
                dy = 0.15

            ax.text(
                x + dx,
                y + dy,
                f"{x:.1f}",
                fontsize=12.8,
                ha="center",
                va="top",
                color="#333333",
            )
        max_p = rows["p_value_fdr"].max() if not rows.empty else np.nan
        p_note = "all paired tests: " + self._fmt_p(max_p)
        ax.text(-1.15, -0.82, p_note, fontsize=13.8, ha="right", va="bottom", color="#333333")
        ax.text(-33.0, len(order) - 0.55, "RAD-DINO omitted: no paired UAR row",
                fontsize=13.3, ha="left", va="center", color="#333333")

        ax.set_xlim(-34, 2.4)
        ax.set_ylim(-0.95, len(order) - 0.10)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(order, fontsize=14.0)
        ax.invert_yaxis()
        ax.set_xlabel("UAR difference vs text-only baseline (pp)")
        self._panel_label(ax, "d", "Paired UAR departures from text-only baselines",
                          x=-0.085, y=1.035, title_offset=0.038)

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
    Figure4SwapNoiseFloor().build().save()
