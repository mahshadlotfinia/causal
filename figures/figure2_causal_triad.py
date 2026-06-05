"""
figures/figure2_causal_triad.py
Created May 27, 2026

Figure 2: A causal triad separates image-using, image-ignoring, and unstable systems.

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


class Figure2CausalTriad:
    MAIN_CSV = "/PATH/all_metrics_mimic.csv"
    PAIRWISE_CSV = "/PATH/paired_comparisons.csv"
    OUT_PDF = "/PATH/figure2_causal_triad.pdf"

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

    TAXONOMY_ORDER = ["Uses image", "Ignores image", "Unstable"]

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
            "font.size": 20,
            "axes.labelsize": 18.0,
            "axes.titlesize": 19.0,
            "xtick.labelsize": 15.6,
            "ytick.labelsize": 15.6,
            "legend.fontsize": 17.4,
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
    def _fmt_signed(x, digits=1):
        return "n/a" if pd.isna(x) else f"{x:+.{digits}f}"

    @staticmethod
    def _fmt_p(p):
        if pd.isna(p):
            return "p_FDR = n/a"
        if p < 0.001:
            return r"$p_{FDR}<0.001$"
        return rf"$p_{{FDR}}={p:.3f}$"

    @staticmethod
    def add_panel_label(ax, letter, title="", x=-0.15, y=1.03, title_offset=0.05):
        ax.text(x, y, letter, transform=ax.transAxes,
                fontsize=26, fontweight="bold", va="bottom", ha="left")
        if title:
            ax.text(x + title_offset, y, title, transform=ax.transAxes,
                    fontsize=20, fontweight="normal", va="bottom", ha="left")

    @staticmethod
    def _clean_axes(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    def _load_data(self):
        df = pd.read_csv(self.main_csv)
        pw = pd.read_csv(self.pairwise_csv)

        percent_cols = [
            "accuracy", "accuracy_ci_lower", "accuracy_ci_upper",
            "CGR", "CGR_ci_lower", "CGR_ci_upper",
            "UAR", "UAR_ci_lower", "UAR_ci_upper",
            "irrelevant_stable", "irrelevant_stable_ci_lower", "irrelevant_stable_ci_upper",
            "sensitivity", "specificity",
        ]
        for col in percent_cols:
            df[col] = df[col].astype(float) * 100.0

        # Defragment after repeated column updates.
        df = df.copy()

        modality = np.select(
            [df["is_text_only"].astype(bool), df["is_vision_only"].astype(bool)],
            ["Text-only", "Vision-only"],
            default="Multimodal",
        )

        category = df.apply(self._assign_category, axis=1)

        df = df.assign(
            modality=modality,
            category=category,
            GSP=df["CGR"] - (100.0 - df["irrelevant_stable"]),
            order_ix=df["model"].map({m: i for i, m in enumerate(self.PREFERRED_MODEL_ORDER)}),
        )

        self.df = df.copy()
        self.pw = pw


    def _assign_category(self, row):
        cgr, cgr_lo, uar, isv = row["CGR"], row["CGR_ci_lower"], row["UAR"], row["irrelevant_stable"]
        if isv < 70.0:
            return "Unstable"
        if np.isclose(cgr, 0.0) and np.isclose(uar, 100.0) and np.isclose(isv, 100.0):
            return "Ignores image"
        if (cgr > 0.0) and (cgr_lo > 0.0) and (isv >= 90.0):
            return "Uses image"
        return "Other"

    def _color(self, category):
        return self.CAT_COLORS.get(category, self.CAT_COLORS["Other"])

    def _shape(self, modality):
        return self.SHAPES.get(modality, "o")

    def _data_row(self, model):
        return self.df.loc[self.df["model"] == model].iloc[0]

    def _paired_row(self, model_a, model_b):
        rows = self.pw[
            (self.pw["dataset"] == "mimic") &
            (self.pw["metric"] == "accuracy") &
            (self.pw["comparison_type"] == "pairwise_accuracy") &
            (self.pw["model_a"] == model_a) &
            (self.pw["model_b"] == model_b)
        ]
        if not rows.empty:
            return rows.iloc[0]
        rows = self.pw[
            (self.pw["dataset"] == "mimic") &
            (self.pw["metric"] == "accuracy") &
            (self.pw["comparison_type"] == "pairwise_accuracy") &
            (self.pw["model_a"] == model_b) &
            (self.pw["model_b"] == model_a)
        ]
        row = rows.iloc[0].copy()
        row["diff"] = -float(row["diff"])
        return row

    @staticmethod
    def _compact_y_label(model):
        return {
            "Gemma-4-26B": "Gemma-4-26B",
            "MedGemma-1.5-4B": "MedGemma-1.5-4B",
            "GPT-5": "GPT-5",
            "Qwen3-VL-32B": "Qwen3-VL-32B",
            "RAD-DINO": "RAD-DINO",
            "MedGemma-27B-text": "MedGemma-27B-text",
            "LLaVA-Med-7B": "LLaVA-Med-7B",
            "DeepSeek-R1-7B": "DeepSeek-R1-7B",
            "Mistral-Small-4-119B": "Mistral-Small-4-119B",
        }[model]

    def build(self):
        self.fig = plt.figure(figsize=(19.5, 22.4), facecolor="white")

        # Tighter vertical packing than v2, but preserve clear separation.
        gs_leg = self.fig.add_gridspec(1, 1, left=0.05, right=0.988, top=0.989, bottom=0.963)
        gs_r1 = self.fig.add_gridspec(1, 2, left=0.085, right=0.983, top=0.931, bottom=0.703,
                                      width_ratios=[1.28, 1.12], wspace=0.42)
        gs_r2 = self.fig.add_gridspec(1, 2, left=0.105, right=0.982, top=0.648, bottom=0.452,
                                      width_ratios=[0.92, 1.38], wspace=0.14)
        gs_r3 = self.fig.add_gridspec(1, 2, left=0.085, right=0.980, top=0.402, bottom=0.082,
                                      width_ratios=[1.62, 1.00], wspace=0.18)

        self._draw_legend(self.fig.add_subplot(gs_leg[0, 0]))
        ax_a = self.fig.add_subplot(gs_r1[0, 0])
        ax_b = self.fig.add_subplot(gs_r1[0, 1])
        ax_c = self.fig.add_subplot(gs_r2[0, 0])
        ax_d = self.fig.add_subplot(gs_r2[0, 1])
        ax_e = self.fig.add_subplot(gs_r3[0, 0])
        ax_f = self.fig.add_subplot(gs_r3[0, 1])

        # Move panel c slightly to the right and slightly downward.
        # The small downward shift increases the vertical gap between panels a and c.
        pos_c = ax_c.get_position()
        ax_c.set_position([pos_c.x0 + 0.018, pos_c.y0 - 0.008, pos_c.width * 0.97, pos_c.height])

        # Panel d: keep the adjusted size and move it slightly downward as well,
        # increasing the vertical gap between panels b and d.
        pos_d = ax_d.get_position()
        new_w_d = pos_d.width * 1.02
        new_h_d = pos_d.height * 1.02
        new_y0_d = pos_d.y0 - (new_h_d - pos_d.height) - 0.008
        ax_d.set_position([pos_d.x0 + 0.040, new_y0_d, new_w_d, new_h_d])

        # Make panel e shorter in the vertical direction, and shift it slightly downward
        # so the vertical gap between panels c and e increases slightly.
        pos_e = ax_e.get_position()
        new_h = pos_e.height * 0.80
        new_y0 = pos_e.y0 + (pos_e.height - new_h) - 0.009
        ax_e.set_position([pos_e.x0, new_y0, pos_e.width, new_h])

        # Shift panel f slightly downward.
        pos_f = ax_f.get_position()
        ax_f.set_position([pos_f.x0, pos_f.y0 - 0.012, pos_f.width, pos_f.height])

        self._panel_a(ax_a)
        self._panel_b(ax_b)
        self._panel_c(ax_c)
        self._panel_d(ax_d)
        self._panel_e(ax_e)
        self._panel_f(ax_f)
        return self

    def _draw_legend(self, ax):
        ax.axis("off")
        cat_handles = [
            Line2D([0], [0], marker="o", lw=0, markersize=16,
                   mfc=self.CAT_COLORS[c], mec="white", mew=0.8, label=c)
            for c in self.TAXONOMY_ORDER
        ]
        mod_handles = [
            Line2D([0], [0], marker=m, lw=0, markersize=16,
                   mfc="#777777", mec="white", mew=0.8, label=label)
            for label, m in self.SHAPES.items()
        ]
        leg1 = ax.legend(cat_handles, [h.get_label() for h in cat_handles], ncol=3, frameon=False,
                         loc="center", bbox_to_anchor=(0.305, 0.47), title="Behavioral category",
                         title_fontsize=18.5, columnspacing=2.1, handletextpad=0.52)
        ax.add_artist(leg1)
        ax.legend(mod_handles, [h.get_label() for h in mod_handles], ncol=3, frameon=False,
                  loc="center", bbox_to_anchor=(0.80, 0.47), title="Modality",
                  title_fontsize=18.5, columnspacing=2.1, handletextpad=0.52)

    def _plot_point_with_ci(self, ax, row, y_jit=0.0, size=150):
        x = row["irrelevant_stable"]
        y = row["CGR"] + y_jit
        col = self._color(row["category"])
        ax.errorbar(x, y,
                    xerr=[[x - row["irrelevant_stable_ci_lower"]], [row["irrelevant_stable_ci_upper"] - x]],
                    yerr=[[y - (row["CGR_ci_lower"] + y_jit)], [(row["CGR_ci_upper"] + y_jit) - y]],
                    fmt="none", ecolor=col, elinewidth=1.2, capsize=3, zorder=2)
        ax.scatter(x, y, s=size, marker=self._shape(row["modality"]),
                   color=col, edgecolor="white", linewidth=0.8, zorder=4)

    def _panel_a(self, ax):
        self._clean_axes(ax)
        ax.set_xlim(50, 103)
        ax.set_ylim(-3, 55)
        ax.set_xlabel("Irrelevant-mask stability, IS (%)", fontsize=18.0)
        ax.set_ylabel("Causal grounding rate, CGR (%)", fontsize=18.0)

        ax.add_patch(patches.Rectangle((90, 0), 13, 55, color=self.CAT_COLORS["Uses image"], alpha=0.07, zorder=0, lw=0))
        ax.add_patch(patches.Rectangle((50, -3), 20, 58, color=self.CAT_COLORS["Unstable"], alpha=0.07, zorder=0, lw=0))
        ax.add_patch(patches.Rectangle((97, -3), 6, 7, color=self.CAT_COLORS["Ignores image"], alpha=0.09, zorder=0, lw=0))
        ax.text(101.8, 52.6, "uses image", color=self.CAT_COLORS["Uses image"], fontsize=16.4, alpha=0.45, ha="right", va="top")
        ax.text(58.2, 53.5, "unstable", color=self.CAT_COLORS["Unstable"], fontsize=16.4, alpha=0.45, ha="left", va="top")
        ax.text(96.0, 0.7, "ignores \nimage", color=self.CAT_COLORS["Ignores image"], fontsize=15.8, alpha=0.50, ha="right", va="bottom")

        ignore_models = self.df.loc[self.df["category"] == "Ignores image"].sort_values("order_ix")["model"].tolist()
        jitters = dict(zip(ignore_models, [-1.2, 0.0, 1.2]))
        for _, row in self.df.sort_values("order_ix").iterrows():
            self._plot_point_with_ci(ax, row, y_jit=jitters.get(row["model"], 0.0), size=150)

        offsets = {
            "Gemma-4-26B": (18, 18),
            "MedGemma-1.5-4B": (-114, 18),
            "GPT-5": (16, -6),
            "Qwen3-VL-32B": (16, -9),
            "RAD-DINO": (-88, 14),
        }
        for model, offset in offsets.items():
            row = self._data_row(model)
            ax.annotate(model, xy=(row["irrelevant_stable"], row["CGR"]), xytext=offset,
                        textcoords="offset points", fontsize=15.7, color="#333333",
                        ha="left" if offset[0] >= 0 else "right", va="center",
                        arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8, shrinkA=0, shrinkB=6))

        mistral = self._data_row("Mistral-Small-4-119B")
        ax.annotate(f"Mistral-Small-4-119B\nhigh CGR but low IS\n(n={int(mistral['CGR_n'])})",
                    xy=(mistral["irrelevant_stable"], mistral["CGR"]), xytext=(60.8, 48.5),
                    textcoords="data", fontsize=14.9, color="#333333", ha="left", va="top",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#DDDDDD", lw=0.8),
                    arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8, shrinkA=0, shrinkB=6))


        self.add_panel_label(ax, "a", "Three behavioral categories from one joint criterion", x=-0.13, y=1.03, title_offset=0.048)

    def _panel_b(self, ax):
        self._clean_axes(ax)
        order = self.df.sort_values(["GSP", "order_ix"], ascending=[False, True]).reset_index(drop=True)

        # Shift the blue and red groups downward while leaving the orange unstable row in place.
        y_positions = []
        y_cursor = 0.65
        for _, row in order.iterrows():
            if row["category"] == "Uses image":
                y_positions.append(y_cursor)
                y_cursor += 1.0
            elif row["category"] == "Ignores image":
                if y_cursor < 6.0:
                    y_cursor = 6.0
                y_positions.append(y_cursor)
                y_cursor += 1.0
            else:
                y_positions.append(8.0)
        y_positions = np.array(y_positions)

        ax.set_xlim(-10, 35)
        ax.set_ylim(-0.1, 8.7)
        ax.axvline(0, color="#444444", lw=1.0, zorder=0)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([self._compact_y_label(m) for m in order["model"].tolist()], fontsize=13.0)
        ax.tick_params(axis="y", pad=1)
        ax.invert_yaxis()
        ax.set_xlabel("Grounding-specificity premium, CGR - (100 - IS) (pp)", fontsize=18.0)

        for i, row in order.iterrows():
            ypos = y_positions[i]
            col = self._color(row["category"])
            if row["category"] == "Unstable":
                ax.axhspan(ypos - 0.45, ypos + 0.45, color=col, alpha=0.08, zorder=0)
            ax.plot([0, row["GSP"]], [ypos, ypos], color=col, lw=3.2, zorder=2)
            ax.scatter(row["GSP"], ypos, s=110, color=col, edgecolor="white", linewidth=0.8, zorder=3)
            ax.text(row["GSP"] + (0.8 if row["GSP"] >= 0 else -0.8), ypos, self._fmt_signed(row["GSP"]),
                    fontsize=14.3, color="#333333", va="center",
                    ha="left" if row["GSP"] >= 0 else "right")

        use_ix = order.index[order["category"] == "Uses image"].tolist()
        y0, y1 = y_positions[min(use_ix)] - 0.35, y_positions[max(use_ix)] + 0.35
        x = 35.0
        ax.plot([x, x], [y0, y1], color="#555555", lw=0.9, clip_on=False)
        ax.plot([x - 0.8, x], [y0, y0], color="#555555", lw=0.9, clip_on=False)
        ax.plot([x - 0.8, x], [y1, y1], color="#555555", lw=0.9, clip_on=False)
        ax.text(x - 0.9, (y0 + y1) / 2, "target-region\nocclusion drives\nmore flips",
                fontsize=14.0, color="#333333", ha="right", va="center")

        self.add_panel_label(ax, "b", "Specificity of the grounding signal", x=-0.16, y=1.0, title_offset=0.06)
        self.gsp_order = order["model"].tolist()

    def _panel_c(self, ax):
        self._clean_axes(ax)
        order = self.df.set_index("model").loc[self.gsp_order].reset_index()
        y = np.arange(len(order))
        ax.set_xlim(0, 105)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_yticks(y)
        ax.set_yticklabels([self._compact_y_label(m) for m in order["model"].tolist()], fontsize=12.8)
        ax.tick_params(axis="y", pad=1)
        ax.invert_yaxis()
        ax.set_xlabel("Unrelated-image answer rate, UAR (%)", fontsize=18.0)
        ax.axvline(100, color=self.CAT_COLORS["Ignores image"], lw=1.0, ls="--", zorder=0)
        ax.text(110.0, -0.8, "perfect UAR", color=self.CAT_COLORS["Ignores image"],
                fontsize=14.0, style="italic", ha="right", va="bottom")

        for i, row in order.iterrows():
            col = self._color(row["category"])
            ax.barh(i, row["UAR"], height=0.62, color=col, alpha=0.85, edgecolor="none", zorder=2)
            ax.errorbar(row["UAR"], i,
                        xerr=[[row["UAR"] - row["UAR_ci_lower"]], [row["UAR_ci_upper"] - row["UAR"]]],
                        fmt="none", ecolor=col, elinewidth=1.2, capsize=3, zorder=3)

        use_ix = order.index[order["category"] == "Uses image"].tolist()
        y0, y1 = min(use_ix) - 0.35, max(use_ix) + 0.35
        x = 87.0
        tick_len = 1.2
        text_gap = 1.2

        ax.plot([x, x], [y0, y1], color="#555555", lw=0.9, clip_on=False)
        ax.plot([x - tick_len, x], [y0, y0], color="#555555", lw=0.9, clip_on=False)
        ax.plot([x - tick_len, x], [y1, y1], color="#555555", lw=0.9, clip_on=False)

        ax.text(
            x + text_gap,
            (y0 + y1) / 2,
            "image- \nusing \nband \n75-82%",
            fontsize=14.0,
            color="#333333",
            ha="left",
            va="center",
        )

        self.add_panel_label(ax, "c", "Answer preserved under image swap", x=-0.18, y=1.03, title_offset=0.06)

    def _panel_d(self, ax):
        self._clean_axes(ax)
        ax.set_xlim(-3, 103)
        ax.set_ylim(-3, 103)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Specificity (%)", fontsize=18.0)
        ax.set_ylabel("Sensitivity (%)", fontsize=18.0)
        ax.plot([0, 100], [100, 0], color="#CCCCCC", ls="--", lw=1.1, zorder=0)
        ax.text(17, 101, "always-Yes", color="#777777", fontsize=14.0, style="italic", ha="left", va="top")
        ax.text(90, 4, "always-No", color="#777777", fontsize=14.0, style="italic", ha="right", va="bottom")

        for _, row in self.df.sort_values("order_ix").iterrows():
            ax.scatter(row["specificity"], row["sensitivity"], s=120,
                       marker=self._shape(row["modality"]), color=self._color(row["category"]),
                       edgecolor="white", linewidth=0.8, zorder=3)

        llava = self._data_row("LLaVA-Med-7B")
        ax.annotate("LLaVA-Med-7B\nalways-Yes (spec 0.0)",
                    xy=(llava["specificity"], llava["sensitivity"]), xytext=(50, 87),
                    textcoords="data", fontsize=14.9, color="#333333", ha="left", va="top",
                    arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8, shrinkA=0, shrinkB=6))
        mistral = self._data_row("Mistral-Small-4-119B")
        ax.annotate("Mistral-Small-4-119B\nmostly-No",
                    xy=(mistral["specificity"], mistral["sensitivity"]), xytext=(80, 27),
                    textcoords="data", fontsize=15.0, color="#333333", ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8, shrinkA=0, shrinkB=6))

        self.add_panel_label(ax, "d", "How models reach their answers", x=-0.16, y=1.03, title_offset=0.07)

    @staticmethod
    def _short_model_label(model):
        return {
            "Gemma-4-26B": "Gemma-4-26B",
            "GPT-5": "GPT-5",
            "Qwen3-VL-32B": "Qwen3-VL-32B",
            "MedGemma-27B-text": "MedGemma-27B-text",
            "RAD-DINO": "RAD-DINO",
            "MedGemma-1.5-4B": "MedGemma-1.5-4B",
            "LLaVA-Med-7B": "LLaVA-Med-7B",
            "DeepSeek-R1-7B": "DeepSeek-R1-7B",
            "Mistral-Small-4-119B": "Mistral-Small-4-119B",
        }[model]

    def _add_bracket(self, ax, x1, x2, y, label, note, height=0.8, text_y=0.28):
        ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="#333333", lw=1.0, clip_on=False)
        ax.text((x1 + x2) / 2, y + height + text_y, f"{label}\n{note}", ha="center", va="bottom", fontsize=13.2, color="#333333", linespacing=1.16)

    def _panel_e(self, ax):
        self._clean_axes(ax)
        order = self.df.sort_values(["accuracy", "order_ix"], ascending=[False, True]).reset_index(drop=True)
        x = np.arange(len(order))
        ax.set_ylim(38, 72)
        ax.set_ylabel("Accuracy (%)", fontsize=18.0)
        ax.set_xticks(x)
        ax.set_xticklabels([self._short_model_label(m) for m in order["model"]], fontsize=13.0, rotation=45, ha="right", rotation_mode="anchor")

        for i, row in order.iterrows():
            col = self._color(row["category"])
            hatch = "///" if row["modality"] == "Text-only" else None
            ax.bar(i, row["accuracy"], width=0.42, color=col, alpha=0.88,
                   edgecolor="#333333" if hatch else "none", linewidth=0.6 if hatch else 0.0, hatch=hatch, zorder=2)
            ax.errorbar(i, row["accuracy"],
                        yerr=[[row["accuracy"] - row["accuracy_ci_lower"]], [row["accuracy_ci_upper"] - row["accuracy"]]],
                        fmt="none", ecolor=col, elinewidth=1.2, capsize=3, zorder=3)
            if row["modality"] == "Text-only":
                ax.text(i, row["accuracy"] + 2.4, "text-only", fontsize=12.4, color="#333333", ha="center", va="bottom", rotation=0)

        xpos = {m: i for i, m in enumerate(order["model"])}
        row1 = self._paired_row("MedGemma-1.5-4B", "MedGemma-27B-text")
        self._add_bracket(ax, xpos["MedGemma-27B-text"], xpos["MedGemma-1.5-4B"],
                          64.9, rf"$\Delta={row1['diff'] * 100:+.1f}$ pp, {self._fmt_p(row1['p_value_fdr'])}",
                          "ignores-image model beats a uses-image model", height=0.8, text_y=0.24)

        row2 = self._paired_row("DeepSeek-R1-7B", "Mistral-Small-4-119B")
        self._add_bracket(ax, xpos["DeepSeek-R1-7B"], xpos["Mistral-Small-4-119B"],
                          51.0, rf"$\Delta={row2['diff'] * 100:+.1f}$ pp, {self._fmt_p(row2['p_value_fdr'])}",
                          "119B multimodal ~ 7B text-only", height=0.75, text_y=0.22)

        self.add_panel_label(ax, "e", "Accuracy ranking cuts across behavioral categories", x=-0.065, y=1.02, title_offset=0.03)

    def _taxonomy_members(self, cat):
        members = self.df.loc[self.df["category"] == cat].sort_values("order_ix")
        return list(members.iterrows())

    def _draw_taxonomy_block(self, ax, x0, y0, w, h, cat, rule, members, two_cols=False):
        col = self.CAT_COLORS[cat]
        header_h = 0.075
        ax.add_patch(patches.FancyBboxPatch((x0, y0), w, h,
                                            boxstyle="round,pad=0.006,rounding_size=0.015",
                                            transform=ax.transAxes, facecolor="white",
                                            edgecolor="#DDDDDD", lw=0.9))
        ax.add_patch(patches.Rectangle((x0, y0 + h - header_h), w, header_h,
                                       transform=ax.transAxes, facecolor=col, edgecolor=col, lw=0))
        ax.text(x0 + 0.02, y0 + h - header_h / 2, cat, transform=ax.transAxes,
                color="white", fontsize=16.0, ha="left", va="center")
        ax.text(x0 + 0.02, y0 + h - header_h - 0.018, rule, transform=ax.transAxes,
                color="#333333", fontsize=12.8, family="DejaVu Sans Mono", ha="left", va="top")

        if two_cols:
            ncols = 2
            nrows = int(np.ceil(len(members) / ncols))
            x_positions = [x0 + 0.035, x0 + 0.48]
            for ci in range(ncols):
                yy = y0 + h - header_h - 0.075
                subset = members[ci * nrows:(ci + 1) * nrows]
                for _, row in subset:
                    xx = x_positions[ci]
                    ax.scatter(xx, yy, transform=ax.transAxes, s=48, marker=self._shape(row["modality"]),
                               color=col, edgecolor="white", linewidth=0.6, zorder=3)
                    ax.text(xx + 0.022, yy, row["model"], transform=ax.transAxes,
                            fontsize=13.4, color="#222222", ha="left", va="center")
                    yy -= 0.048
        else:
            yy = y0 + h - header_h - 0.075
            step = 0.052 if len(members) >= 3 else 0.060
            for _, row in members:
                ax.scatter(x0 + 0.035, yy, transform=ax.transAxes, s=48, marker=self._shape(row["modality"]),
                           color=col, edgecolor="white", linewidth=0.6, zorder=3)
                ax.text(x0 + 0.058, yy, row["model"], transform=ax.transAxes,
                        fontsize=13.4, color="#222222", ha="left", va="center")
                yy -= step

    def _panel_f(self, ax):
        ax.axis("off")
        self.add_panel_label(ax, "f", "Taxonomy defined by the causal triad", x=-0.06, y=0.985, title_offset=0.038)

        use_members = self._taxonomy_members("Uses image")
        ignore_members = self._taxonomy_members("Ignores image")
        unstable_members = self._taxonomy_members("Unstable")

        self._draw_taxonomy_block(ax, 0.02, 0.66, 0.96, 0.29,
                                  "Uses image", "CGR > 0 (CI excludes 0) and IS >= 90",
                                  use_members, two_cols=True)
        self._draw_taxonomy_block(ax, 0.02, 0.405, 0.96, 0.25,
                                  "Ignores image", "CGR = 0, UAR = 100, IS = 100",
                                  ignore_members, two_cols=True)
        self._draw_taxonomy_block(ax, 0.02, 0.21, 0.96, 0.17,
                                  "Unstable", "IS < 70 -> CGR uninterpretable",
                                  unstable_members, two_cols=False)

    def save(self):
        os.makedirs(self.out_pdf.parent, exist_ok=True)
        self.fig.savefig(self.out_pdf, format="pdf", bbox_inches="tight", facecolor="white")
        plt.close(self.fig)
        return str(self.out_pdf)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    Figure2CausalTriad().build().save()
