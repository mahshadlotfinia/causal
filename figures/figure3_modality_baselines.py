"""
figures/figure3_modality_baselines_2x2.py
Created May 27, 2026

Figure 3: Model design class, text baselines, and parameter count decouple from accuracy.

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


class Figure3ModalityBaselines:
    MAIN_CSV = "/PATH/all_metrics_mimic.csv"
    PAIRWISE_CSV = "/PATH/paired_comparisons.csv"
    OUT_PDF = "/PATH/figure3_modality_baselines.pdf"

    CLASS_COLORS = {
        "Frontier closed-source": "#542788",
        "General-purpose multimodal": "#2166AC",
        "Medical specialist multimodal": "#1B7837",
        "Text-only baseline": "#777777",
        "Vision-only probe": "#B35806",
    }
    SIG_COLORS = {
        "sig_positive": "#1A9850",
        "sig_negative": "#D73027",
        "nonsig": "#999999",
    }
    SHAPES = {
        "Multimodal": "o",
        "Text-only": "s",
        "Vision-only": "D",
    }
    BASELINES = {
        "strong": "MedGemma-27B-text",
        "weak": "DeepSeek-R1-7B",
    }
    MODEL_CLASS = {
        "GPT-5": "Frontier closed-source",
        "Gemma-4-26B": "General-purpose multimodal",
        "Qwen3-VL-32B": "General-purpose multimodal",
        "Mistral-Small-4-119B": "General-purpose multimodal",
        "MedGemma-1.5-4B": "Medical specialist multimodal",
        "LLaVA-Med-7B": "Medical specialist multimodal",
        "MedGemma-27B-text": "Text-only baseline",
        "DeepSeek-R1-7B": "Text-only baseline",
        "RAD-DINO": "Vision-only probe",
    }
    # Flagged metadata, not inferred from CSV values. GPT-5 remains undisclosed.
    PARAM_B = {
        "RAD-DINO": 0.05,
        "MedGemma-1.5-4B": 4.0,
        "LLaVA-Med-7B": 7.0,
        "DeepSeek-R1-7B": 7.0,
        "Gemma-4-26B": 26.0,
        "MedGemma-27B-text": 27.0,
        "Qwen3-VL-32B": 32.0,
        "Mistral-Small-4-119B": 119.0,
        "GPT-5": np.nan,
    }

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
            "font.size": 19,
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
    def _fmt_pp(x):
        return f"{x:+.1f}"

    @staticmethod
    def _fmt_p_value(p):
        if pd.isna(p):
            return r"$p$ = n/a"
        if p < 0.001:
            return r"$p$ < 0.001"
        return rf"$p$ = {p:.3f}"

    @staticmethod
    def _short_model(model):
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

    @staticmethod
    def _panel_label(ax, letter, title, x=-0.10, y=1.035, title_offset=0.04):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=27, fontweight="bold",
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
            ["model", "accuracy", "accuracy_ci_lower", "accuracy_ci_upper",
             "is_text_only", "is_vision_only"],
            self.main_csv.name,
        )
        self._require_columns(
            pw,
            ["comparison_type", "dataset", "model_a", "model_b", "metric", "diff",
             "diff_ci_lower", "diff_ci_upper", "p_value_fdr", "n_shared"],
            self.pairwise_csv.name,
        )
        for col in ["accuracy", "accuracy_ci_lower", "accuracy_ci_upper"]:
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

        df = df.assign(
            modality=modality,
            model_class=df["model"].map(self.MODEL_CLASS),
            param_b=df["model"].map(self.PARAM_B),
        )
        if df["model_class"].isna().any():
            missing = df.loc[df["model_class"].isna(), "model"].tolist()
            raise ValueError(f"Missing model-class metadata for: {missing}")
        self.df = df
        self.pw = pw
        self.clears_strong = self._compute_clears_strong()

    def _row(self, model):
        rows = self.df.loc[self.df["model"] == model]
        if rows.empty:
            raise KeyError(model)
        return rows.iloc[0]

    def _vs_baseline(self, baseline):
        rows = self.pw[
            (self.pw["comparison_type"] == "vs_text_baseline") &
            (self.pw["dataset"] == "mimic") &
            (self.pw["metric"] == "accuracy") &
            (self.pw["model_b"] == baseline)
        ].copy()
        rows = rows[rows["model_a"] != baseline].copy()
        rows["sig_class"] = np.select(
            [(rows["diff"] > 0) & (rows["p_value_fdr"] < 0.05),
             (rows["diff"] < 0) & (rows["p_value_fdr"] < 0.05)],
            ["sig_positive", "sig_negative"], default="nonsig",
        )
        rows["order_value"] = rows["diff"]
        return rows.sort_values("order_value", ascending=False).reset_index(drop=True)

    def _compute_clears_strong(self):
        strong = self.BASELINES["strong"]
        rows = self.pw[
            (self.pw["comparison_type"] == "vs_text_baseline") &
            (self.pw["dataset"] == "mimic") &
            (self.pw["metric"] == "accuracy") &
            (self.pw["model_b"] == strong)
        ].copy()
        clear = set(rows.loc[(rows["diff"] > 0) & (rows["p_value_fdr"] < 0.05), "model_a"])
        return clear

    def _baseline_values(self):
        med = self._row(self.BASELINES["strong"])["accuracy"]
        deep = self._row(self.BASELINES["weak"])["accuracy"]
        return float(med), float(deep)

    def _class_color(self, model):
        return self.CLASS_COLORS[self._row(model)["model_class"]]

    def build(self):
        self.fig = plt.figure(figsize=(18.8, 18.2), facecolor="white")
        gs_leg = self.fig.add_gridspec(1, 1, left=0.055, right=0.985, top=0.992, bottom=0.928)
        gs = self.fig.add_gridspec(2, 2, left=0.075, right=0.975, top=0.872, bottom=0.090,
                                   wspace=0.36, hspace=0.48, width_ratios=[1.05, 1.0])

        ax_leg = self.fig.add_subplot(gs_leg[0, 0])
        ax_a = self.fig.add_subplot(gs[0, 0])
        ax_b = self.fig.add_subplot(gs[0, 1])
        ax_c = self.fig.add_subplot(gs[1, 0])
        ax_d = self.fig.add_subplot(gs[1, 1])

        self._draw_legend(ax_leg)
        self._panel_a(ax_a)
        self._panel_b(ax_b)
        self._panel_c(ax_c)
        self._panel_d(ax_d)
        return self

    def _draw_legend(self, ax):
        ax.axis("off")
        class_order = [
            "Frontier closed-source", "General-purpose multimodal", "Medical specialist multimodal",
            "Text-only baseline", "Vision-only probe",
        ]
        class_display = {
            "Frontier closed-source": "Frontier",
            "General-purpose multimodal": "General multimodal",
            "Medical specialist multimodal": "Medical multimodal",
            "Text-only baseline": "Text-only baseline",
            "Vision-only probe": "Vision-only probe",
        }
        class_handles = [
            Line2D([0], [0], marker="o", lw=0, markersize=15.5,
                   mfc=self.CLASS_COLORS[c], mec="white", mew=0.8, label=class_display[c])
            for c in class_order
        ]
        modality_handles = [
            Line2D([0], [0], marker=m, lw=0, markersize=15.5,
                   mfc="#777777", mec="white", mew=0.8, label=k)
            for k, m in self.SHAPES.items()
        ]
        sig_handles = [
            Line2D([0], [0], marker="o", lw=0, markersize=14.5,
                   mfc=self.SIG_COLORS["sig_positive"], mec="white", mew=0.8, label="sig. positive"),
            Line2D([0], [0], marker="o", lw=0, markersize=14.5,
                   mfc=self.SIG_COLORS["sig_negative"], mec="white", mew=0.8, label="sig. negative"),
            Line2D([0], [0], marker="o", lw=0, markersize=14.5,
                   mfc=self.SIG_COLORS["nonsig"], mec="white", mew=0.8, label="n.s."),
        ]
        clear_handle = Line2D([0], [0], marker="o", lw=0, markersize=15.5,
                              mfc="white", mec="#111111", mew=1.8,
                              label="clears strong baseline")
        ax.text(0.03, 0.72, "Model class", transform=ax.transAxes,
                fontsize=16.5, ha="left", va="center", color="#222222")
        ax.text(0.03, 0.25, "Modality/status", transform=ax.transAxes,
                fontsize=16.5, ha="left", va="center", color="#222222")
        leg1 = ax.legend(class_handles, [h.get_label() for h in class_handles], ncol=5,
                         frameon=False, loc="center", bbox_to_anchor=(0.54, 0.72),
                         columnspacing=1.40, handletextpad=0.45)
        ax.add_artist(leg1)
        leg2 = ax.legend(modality_handles + sig_handles + [clear_handle],
                         [h.get_label() for h in modality_handles + sig_handles + [clear_handle]],
                         ncol=7, frameon=False, loc="center", bbox_to_anchor=(0.55, 0.25),
                         columnspacing=1.05, handletextpad=0.42)
        ax.add_artist(leg2)

    def _panel_a(self, ax):
        self._clean_axes(ax)
        order = self.df.sort_values("accuracy", ascending=False).reset_index(drop=True)
        x = np.arange(len(order))
        med_acc, deep_acc = self._baseline_values()

        ax.axhspan(deep_acc, med_acc, color="#777777", alpha=0.08, zorder=0)
        ax.axhline(med_acc, color=self.CLASS_COLORS["Text-only baseline"], lw=1.25, ls="--", zorder=1)
        ax.axhline(deep_acc, color=self.CLASS_COLORS["Text-only baseline"], lw=1.25, ls="--", zorder=1)
        ax.text(len(order) - 0.08, med_acc + 0.65, "MedGemma-27B-text baseline",
                ha="right", va="bottom", fontsize=15.5, color="#444444")
        # ax.text(len(order) - 0.08, deep_acc + 0.65, "DeepSeek-R1-7B baseline",
        #         ha="right", va="bottom", fontsize=15.5, color="#444444")

        for i, row in order.iterrows():
            model = row["model"]
            col = self.CLASS_COLORS[row["model_class"]]
            edge = "#111111" if model in self.clears_strong else "none"
            lw = 1.8 if model in self.clears_strong else 0.0
            ax.bar(i, row["accuracy"], width=0.54, color=col, alpha=0.90,
                   edgecolor=edge, linewidth=lw, zorder=2)
            ax.errorbar(i, row["accuracy"],
                        yerr=[[row["accuracy"] - row["accuracy_ci_lower"]],
                              [row["accuracy_ci_upper"] - row["accuracy"]]],
                        fmt="none", ecolor=col, elinewidth=1.35, capsize=3, zorder=3)

        ax.text(0.02, 0.985, "black outline: clears strong baseline",
                transform=ax.transAxes, fontsize=15.0, color="#222222", ha="left", va="top")
        ax.set_ylim(38, 73.2)
        ax.set_xlim(-0.65, len(order) - 0.35)
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(x)
        ax.set_xticklabels([self._short_model(m) for m in order["model"]],
                           rotation=45, ha="right", rotation_mode="anchor", fontsize=13.0)
        self._panel_label(ax, "a", "Accuracy landscape and useful-tier mark",
                          x=-0.085, y=1.035, title_offset=0.065)

    def _draw_forest(self, ax, baseline, title, letter, highlight_mistral=False, show_exact_p=False):
        self._clean_axes(ax)
        rows = self._vs_baseline(baseline)
        y = np.arange(len(rows))
        ax.axvline(0, color="#333333", lw=1.15, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([self._short_model(m) for m in rows["model_a"]], fontsize=13.5)
        ax.tick_params(axis="y", pad=2)
        ax.invert_yaxis()

        if highlight_mistral:
            idx = rows.index[rows["model_a"] == "Mistral-Small-4-119B"].tolist()
            if idx:
                ax.axhspan(idx[0] - 0.42, idx[0] + 0.42,
                           color=self.CLASS_COLORS["General-purpose multimodal"], alpha=0.08, zorder=0)

        for i, row in rows.iterrows():
            col = self.SIG_COLORS[row["sig_class"]]
            x = row["diff"]
            lo, hi = row["diff_ci_lower"], row["diff_ci_upper"]
            ax.errorbar(x, i, xerr=[[x - lo], [hi - x]], fmt="none",
                        ecolor=col, elinewidth=1.45, capsize=3, zorder=2)
            ax.scatter(x, i, s=110, color=col, edgecolor="white", linewidth=0.8, zorder=3)
            if show_exact_p:
                is_mistral = row["model_a"] == "Mistral-Small-4-119B"
                is_mistral_nonsig = highlight_mistral and is_mistral

                if not is_mistral_nonsig:
                    label_x = hi + 0.55 if x >= 0 else lo - 0.55
                    label_y = i - 0.12
                    ha = "left" if x >= 0 else "right"

                    # Move only the Mistral p-value in panel b.
                    if baseline == self.BASELINES["strong"] and is_mistral:
                        label_x += -4.0      # increase to move right, decrease to move left
                        label_y -= 0.25     # decrease to move upward, increase to move downward
                        ha = "left"

                    ax.text(
                        label_x,
                        label_y,
                        self._fmt_p_value(row["p_value_fdr"]),
                        fontsize=12.4,
                        ha=ha,
                        va="bottom",
                        color="#333333",
                    )

                ax.text(x, i - 0.34, self._fmt_pp(x), fontsize=13.2,
                        ha="center", va="top", color="#333333")
            else:
                label_x = hi + 1.0 if x >= 0 else lo - 1.0
                ax.text(label_x, i, self._fmt_p_value(row["p_value_fdr"]), fontsize=12.4,
                        ha="left" if x >= 0 else "right", va="center", color="#333333")
                ax.text(x, i - 0.26, self._fmt_pp(x), fontsize=13.2,
                        ha="center", va="top", color="#333333")

        if highlight_mistral:
            idx = rows.index[rows["model_a"] == "Mistral-Small-4-119B"].tolist()[0]
            mrow = rows.loc[rows["model_a"] == "Mistral-Small-4-119B"].iloc[0]
            ax.annotate(f"119B multimodal approx.\n7B text-only, {self._fmt_p_value(mrow['p_value_fdr'])}",
                        xy=(mrow["diff"], idx), xytext=(15.2, idx - 1.00),
                        textcoords="data", fontsize=14.0, ha="left", va="center", color="#333333",
                        arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8,
                                        shrinkA=0, shrinkB=6))

        if baseline == self.BASELINES["strong"]:
            ax.set_xlim(-28, 25)
        else:
            ax.set_xlim(-8, 28)
        ax.set_xlabel("Accuracy difference vs baseline (pp)")
        self._panel_label(ax, letter, title, x=-0.17, y=1.035, title_offset=0.060)

    def _panel_b(self, ax):
        self._draw_forest(ax, self.BASELINES["strong"],
                          "Paired change vs strong text baseline", "b",
                          highlight_mistral=False, show_exact_p=True)

    def _panel_c(self, ax):
        self._draw_forest(ax, self.BASELINES["weak"],
                          "Paired change vs weak text baseline", "c",
                          highlight_mistral=True, show_exact_p=True)

    def _panel_d(self, ax):
        self._clean_axes(ax)
        ax.set_xscale("log")
        med_acc, deep_acc = self._baseline_values()
        ax.axhline(med_acc, color="#777777", lw=1.15, ls="--", zorder=0)
        ax.axhline(deep_acc, color="#777777", lw=1.15, ls="--", zorder=0)
        ax.text(0.04, med_acc + 1.1, "strong text baseline", fontsize=14.5,
                ha="left", va="bottom", color="#555555")
        ax.text(0.04, deep_acc + 0.8, "weak text baseline", fontsize=14.5,
                ha="left", va="bottom", color="#555555")

        for _, row in self.df.sort_values("accuracy", ascending=False).iterrows():
            model = row["model"]
            if model == "GPT-5":
                continue
            x = self.PARAM_B[model]
            if model in self.clears_strong:
                ax.scatter(x, row["accuracy"], s=220, marker=self.SHAPES[row["modality"]],
                           color="none", edgecolor="#111111", linewidth=1.8, zorder=2.7)
            ax.scatter(x, row["accuracy"], s=140, marker=self.SHAPES[row["modality"]],
                       color=self.CLASS_COLORS[row["model_class"]], edgecolor="white",
                       linewidth=0.8, zorder=3)

        gpt = self._row("GPT-5")
        x_gpt = 320
        ax.axvline(205, color="#999999", lw=1.0, ls=":", zorder=0)
        ax.scatter(x_gpt, gpt["accuracy"], s=220, marker=self.SHAPES[gpt["modality"]],
                   color="none", edgecolor="#111111", linewidth=1.8, zorder=2.7)
        ax.scatter(x_gpt, gpt["accuracy"], s=140, marker=self.SHAPES[gpt["modality"]],
                   color=self.CLASS_COLORS[gpt["model_class"]], edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(x_gpt, gpt["accuracy"] + 2.3, "GPT-5\nundisclosed size",
                fontsize=13.8, ha="center", va="bottom", color="#333333")

        label_offsets = {
            "MedGemma-27B-text": (-56, -20),
            "RAD-DINO": (14, -6),
            "DeepSeek-R1-7B": (-46, -12),
            "Mistral-Small-4-119B": (-82, -8),
        }
        for model, off in label_offsets.items():
            r = self._row(model)
            ax.annotate(model, xy=(self.PARAM_B[model], r["accuracy"]), xytext=off,
                        textcoords="offset points", fontsize=13.2, color="#333333",
                        ha="left" if off[0] >= 0 else "right", va="center",
                        arrowprops=dict(arrowstyle="-", color="#666666", lw=0.7,
                                        shrinkA=0, shrinkB=5))

        mistral = self._row("Mistral-Small-4-119B")
        ax.annotate("119B multimodal approx.\n7B text-only",
                    xy=(119, mistral["accuracy"]), xytext=(18, 39.7), textcoords="data",
                    fontsize=14.2, color="#333333", ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8,
                                    shrinkA=0, shrinkB=6))
        ax.annotate("small probe,\nfrontier-tier accuracy",
                    xy=(0.05, self._row("RAD-DINO")["accuracy"]), xytext=(0.11, 66.0), textcoords="data",
                    fontsize=14.2, color="#333333", ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8,
                                    shrinkA=0, shrinkB=6))
        ax.text(0.04, 39.0, "No trend fitted (n=9)", fontsize=14.5,
                ha="left", va="bottom", color="#333333", style="italic")

        ax.set_xlim(0.035, 480)
        ax.set_ylim(38, 71)
        ax.set_xlabel("Parameter count (billions, log scale)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks([0.05, 1, 7, 26, 119, 320])
        ax.set_xticklabels(["0.05", "1", "7", "26", "119", "undisc."],
                           fontsize=13.2, rotation=25, ha="right")
        self._panel_label(ax, "d", "Parameter count does not order accuracy",
                          x=-0.12, y=1.035, title_offset=0.065)

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
    Figure3ModalityBaselines().build().save()
