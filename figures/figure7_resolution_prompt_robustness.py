"""
figures/figure7_resolution_prompt_robustness.py
Created May 27, 2026

Figure 7: Resolution and prompt robustness checks.

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
from matplotlib.colors import LinearSegmentedColormap, Normalize


class Figure7ResolutionPromptRobustness:
    RESOLUTION_CSV = "/PATH/resolution_check_results.csv"
    PROMPT_CSV = "/PATH/prompt_sensitivity_comparison.csv"
    MAIN_CSV = "/PATH/all_metrics_mimic.csv"
    OUT_PDF = "/PATH/figure7_resolution_prompt_robustness.pdf"

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
    VARIANT_ORDER = ["default", "brief", "clinical"]
    VARIANT_LABELS = {
        "default": "default",
        "brief": "terse",
        "clinical": "radiologist-framed",
    }
    VARIANT_ALPHA = {
        "default": 0.92,
        "brief": 0.62,
        "clinical": 0.35,
    }
    MODEL_ORDER = [
        "Gemma-4-26B", "GPT-5", "Qwen3-VL-32B", "MedGemma-27B-text", "RAD-DINO",
        "MedGemma-1.5-4B", "LLaVA-Med-7B", "DeepSeek-R1-7B", "Mistral-Small-4-119B",
    ]
    ROBUST_ORDER = ["GPT-5", "Qwen3-VL-32B", "RAD-DINO", "DeepSeek-R1-7B"]
    FRAGILE_ORDER = ["Gemma-4-26B", "MedGemma-1.5-4B", "MedGemma-27B-text", "LLaVA-Med-7B", "Mistral-Small-4-119B"]

    def __init__(self, resolution_csv=None, prompt_csv=None, main_csv=None, out_pdf=None):
        self.resolution_csv = Path(resolution_csv or self.RESOLUTION_CSV)
        self.prompt_csv = Path(prompt_csv or self.PROMPT_CSV)
        self.main_csv = Path(main_csv or self.MAIN_CSV)
        self.out_pdf = Path(out_pdf or self.OUT_PDF)
        self.log = logging.getLogger(self.__class__.__name__)
        self._set_rcparams()
        self._load_data()
        self.fig = None

    @staticmethod
    def _set_rcparams():
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 18.0,
            "axes.labelsize": 17.2,
            "axes.titlesize": 19.2,
            "xtick.labelsize": 14.2,
            "ytick.labelsize": 14.0,
            "legend.fontsize": 15.2,
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
    def _panel_label(ax, letter, title, x=-0.12, y=1.025, title_offset=0.050):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=26, fontweight="bold",
                ha="left", va="bottom")
        ax.text(x + title_offset, y, title, transform=ax.transAxes, fontsize=19.2,
                fontweight="normal", ha="left", va="bottom")

    @staticmethod
    def _clean_axes(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    @staticmethod
    def _require(df, cols, name):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {name}: {missing}")

    @staticmethod
    def _fmt_model(model):
        return model

    def _load_data(self):
        res = pd.read_csv(self.resolution_csv)
        prompt = pd.read_csv(self.prompt_csv)
        main = pd.read_csv(self.main_csv)

        self._require(res, [
            "model", "CGR_224", "CGR_224_ci_lower", "CGR_224_ci_upper", "CGR_224_n",
            "CGR_512", "CGR_512_ci_lower", "CGR_512_ci_upper", "CGR_512_n",
            "rank_224", "rank_512", "spearman_rho"
        ], self.resolution_csv.name)
        self._require(prompt, ["model", "variant", "accuracy", "ci_lower", "ci_upper", "n_cases", "parse_rate"],
                      self.prompt_csv.name)
        self._require(main, [
            "model", "CGR", "CGR_ci_lower", "UAR", "irrelevant_stable", "is_text_only", "is_vision_only"
        ], self.main_csv.name)

        for c in ["CGR_224", "CGR_224_ci_lower", "CGR_224_ci_upper", "CGR_512", "CGR_512_ci_lower", "CGR_512_ci_upper"]:
            res[c] = res[c].astype(float) * 100.0
        for c in ["accuracy", "ci_lower", "ci_upper", "parse_rate"]:
            prompt[c] = prompt[c].astype(float) * 100.0
        for c in ["CGR", "CGR_ci_lower", "UAR", "irrelevant_stable"]:
            main[c] = main[c].astype(float) * 100.0

        # Defragment after repeated column-wise updates.
        main = main.copy()

        modality = np.select(
            [main["is_text_only"].astype(bool), main["is_vision_only"].astype(bool)],
            ["Text-only", "Vision-only"],
            default="Multimodal",
        )

        category = main.apply(self._assign_category, axis=1)

        main = main.assign(
            modality=modality,
            category=category,
            order_ix=main["model"].map({m: i for i, m in enumerate(self.MODEL_ORDER)}),
        )

        # Keep stored dataframe defragmented too.
        main = main.copy()

        meta = main[["model", "modality", "category", "order_ix"]].copy()
        res = res.merge(meta, on="model", how="left")
        prompt = prompt.merge(meta, on="model", how="left")

        if res["category"].isna().any() or prompt["category"].isna().any():
            raise ValueError("Missing metadata after merge.")

        prompt["variant_order"] = prompt["variant"].map({v: i for i, v in enumerate(self.VARIANT_ORDER)})
        self.res = res
        self.prompt = prompt
        self.main = main
        robust = prompt.groupby("model")["parse_rate"].min().ge(90.0)
        self.parse_robust_models = set(robust[robust].index.tolist())

    def _assign_category(self, row):
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

    def _row_res(self, model):
        rows = self.res.loc[self.res["model"] == model]
        if rows.empty:
            raise KeyError(model)
        return rows.iloc[0]

    def _row_prompt(self, model, variant):
        rows = self.prompt[(self.prompt["model"] == model) & (self.prompt["variant"] == variant)]
        if rows.empty:
            raise KeyError((model, variant))
        return rows.iloc[0]

    def _color(self, model):
        cat = self.main.set_index("model").loc[model, "category"]
        return self.CAT_COLORS.get(cat, self.CAT_COLORS["Other"])

    def _shape(self, model):
        mod = self.main.set_index("model").loc[model, "modality"]
        return self.SHAPES.get(mod, "o")

    def build(self):
        self.fig = plt.figure(figsize=(17.6, 21.4), facecolor="white")
        gs_leg = self.fig.add_gridspec(1, 1, left=0.052, right=0.986, top=0.990, bottom=0.948)
        gs_r1 = self.fig.add_gridspec(1, 2, left=0.075, right=0.976, top=0.905, bottom=0.682,
                                      wspace=0.34, width_ratios=[1.0, 1.0])
        gs_r2 = self.fig.add_gridspec(1, 2, left=0.085, right=0.976, top=0.608, bottom=0.375,
                                      wspace=0.34, width_ratios=[1.0, 1.62])
        gs_r3 = self.fig.add_gridspec(1, 2, left=0.085, right=0.976, top=0.285, bottom=0.052,
                                      wspace=0.34, width_ratios=[1.0, 1.0])

        ax_leg = self.fig.add_subplot(gs_leg[0, 0])
        ax_a = self.fig.add_subplot(gs_r1[0, 0])
        ax_b = self.fig.add_subplot(gs_r1[0, 1])
        ax_c = self.fig.add_subplot(gs_r2[0, 0])
        ax_d = self.fig.add_subplot(gs_r2[0, 1])
        ax_e = self.fig.add_subplot(gs_r3[0, 0])
        ax_f = self.fig.add_subplot(gs_r3[0, 1])

        self._draw_legend(ax_leg)
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
            Line2D([0], [0], marker="o", lw=0, markersize=15,
                   mfc=self.CAT_COLORS[c], mec="white", mew=0.8, label=c)
            for c in ["Uses image", "Ignores image", "Unstable"]
        ]
        shape_handles = [
            Line2D([0], [0], marker=m, lw=0, markersize=15,
                   mfc="#777777", mec="white", mew=0.8, label=k)
            for k, m in self.SHAPES.items()
        ]
        variant_handles = [
            patches.Patch(facecolor="#777777", edgecolor="none", alpha=self.VARIANT_ALPHA[v],
                          label=self.VARIANT_LABELS[v])
            for v in self.VARIANT_ORDER
        ]
        hatch_handle = patches.Patch(facecolor="#dddddd", edgecolor="#777777", hatch="xxx", label="parse < 50%")
        ax.text(0.02, 0.65, "Behavioral category", transform=ax.transAxes, fontsize=16.0, ha="left", va="center")
        ax.text(0.02, 0.24, "Modality / prompt", transform=ax.transAxes, fontsize=16.0, ha="left", va="center")
        leg1 = ax.legend(cat_handles, [h.get_label() for h in cat_handles], ncol=3, frameon=False,
                         loc="center", bbox_to_anchor=(0.40, 0.65), columnspacing=1.9, handletextpad=0.5)
        ax.add_artist(leg1)
        leg2 = ax.legend(shape_handles + variant_handles + [hatch_handle],
                         [h.get_label() for h in shape_handles + variant_handles + [hatch_handle]],
                         ncol=7, frameon=False, loc="center", bbox_to_anchor=(0.60, 0.24),
                         columnspacing=1.25, handletextpad=0.45)
        ax.add_artist(leg2)

    def _panel_a(self, ax):
        self._clean_axes(ax)
        ax.set_xlim(-3, 65)
        ax.set_ylim(-3, 65)
        ax.set_aspect("equal", adjustable="box")
        ax.plot([-3, 65], [-3, 65], ls="--", lw=1.1, color="#888888", zorder=0)
        ax.axhline(0, color="#cccccc", lw=0.8, zorder=0)
        ax.axvline(0, color="#cccccc", lw=0.8, zorder=0)
        ignore_models = self.res.loc[self.res["category"] == "Ignores image", "model"].tolist()
        jitter = dict(zip(ignore_models, [-1.1, 0.0, 1.1]))
        for _, row in self.res.sort_values("order_ix").iterrows():
            x = row["CGR_224"]
            y = row["CGR_512"] + jitter.get(row["model"], 0.0)
            col = self.CAT_COLORS[row["category"]]
            xerr = [[x - row["CGR_224_ci_lower"]], [row["CGR_224_ci_upper"] - x]]
            yerr = [[y - (row["CGR_512_ci_lower"] + jitter.get(row["model"], 0.0))],
                    [(row["CGR_512_ci_upper"] + jitter.get(row["model"], 0.0)) - y]]
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor=col,
                        elinewidth=1.25, capsize=3, zorder=2)
            ax.scatter(x, y, s=135, marker=self.SHAPES[row["modality"]], color=col,
                       edgecolor="white", linewidth=0.8, zorder=3)
        rho = float(self.res["spearman_rho"].dropna().iloc[0])
        ax.text(0.05, 0.94, rf"$\rho$ = {rho:.3f}", transform=ax.transAxes,
                fontsize=14.8, ha="left", va="top")
        ax.text(5.5, 2.4, "ignores-image\ncluster at origin", fontsize=13.2,
                ha="left", va="center", color=self.CAT_COLORS["Ignores image"])
        ax.text(41.5, 54.0, "Mistral\nwide CI", fontsize=13.2,
                ha="left", va="center", color=self.CAT_COLORS["Unstable"])
        ax.set_xlabel("CGR at 224 px (%)")
        ax.set_ylabel("CGR at 512 px (%)")
        self._panel_label(ax, "a", "CGR values are resolution-stable", x=-0.16, y=1.035, title_offset=0.060)

    def _panel_b(self, ax):
        self._clean_axes(ax)
        data = self.res.sort_values(["rank_224", "order_ix"]).reset_index(drop=True)
        ax.set_xlim(-0.18, 1.30)
        ax.set_ylim(9.0, 0.55)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["224 rank", "512 rank"], fontsize=15.0)
        ax.set_yticks(range(1, 9))
        ax.set_ylabel("CGR rank")
        ax.tick_params(axis="x", length=0)
        ax.axvline(0, color="#BBBBBB", lw=1.0, zorder=0)
        ax.axvline(1, color="#BBBBBB", lw=1.0, zorder=0)
        for _, row in data.iterrows():
            col = self.CAT_COLORS[row["category"]]
            ax.plot([0, 1], [row["rank_224"], row["rank_512"]], color=col, lw=2.2, alpha=0.95, zorder=2)
            ax.scatter([0, 1], [row["rank_224"], row["rank_512"]], s=80,
                       marker=self.SHAPES[row["modality"]], color=col, edgecolor="white", linewidth=0.7, zorder=3)
            if row["rank_512"] <= 6 or row["model"] == "Mistral-Small-4-119B":
                ax.text(1.03, row["rank_512"], row["model"], fontsize=12.1,
                        ha="left", va="center", color="#333333")
        ax.text(0.10, 0.20, "only ranks 2-4 cross", transform=ax.transAxes,
                fontsize=13.0, ha="left", va="bottom", color="#333333",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92))
        self._panel_label(ax, "b", "Rank preservation despite small swaps", x=-0.14, y=1.035, title_offset=0.060)

    def _panel_c(self, ax):
        self._clean_axes(ax)
        data = self.res.sort_values("CGR_224_n", ascending=False).reset_index(drop=True)
        y = np.arange(len(data))
        h = 0.32
        ax.barh(y - h/2, data["CGR_224_n"], height=h, color="#2166AC", alpha=0.82, label="224 px", zorder=2)
        ax.barh(y + h/2, data["CGR_512_n"], height=h, color="#92C5DE", alpha=0.90, label="512 px", zorder=2)
        for i, row in data.iterrows():
            ax.text(row["CGR_224_n"] + 5, i - h/2, f"{int(row['CGR_224_n'])}", fontsize=11.8,
                    ha="left", va="center", color="#333333")
            ax.text(row["CGR_512_n"] + 5, i + h/2, f"{int(row['CGR_512_n'])}", fontsize=11.8,
                    ha="left", va="center", color="#333333")
        ax.set_xlim(0, 485)
        ax.set_ylim(-0.7, len(data)-0.3)
        ax.set_yticks(y)
        ax.set_yticklabels(data["model"].tolist(), fontsize=11.8)
        ax.invert_yaxis()
        ax.set_xlabel("CGR subset size (n)")
        ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.02, 0.01), ncol=1, fontsize=12.6)
        ax.text(0.47, 0.10, "512 checks use\nsmaller subsets", transform=ax.transAxes,
                fontsize=12.5, ha="left", va="center", color="#333333")
        self._panel_label(ax, "c", "512-pixel subset shrinkage", x=-0.22, y=1.030, title_offset=0.080)

    def _panel_d(self, ax):
        self._clean_axes(ax)
        order = self.ROBUST_ORDER + self.FRAGILE_ORDER
        xpos = np.arange(len(order))
        barw = 0.22
        variant_offsets = {"default": -barw, "brief": 0.0, "clinical": barw}
        for idx, model in enumerate(order):
            col = self._color(model)
            for v in self.VARIANT_ORDER:
                r = self._row_prompt(model, v)
                x = idx + variant_offsets[v]
                hatch = "xxx" if r["parse_rate"] < 50 else None
                edge = "#666666" if hatch else "none"
                lw = 0.7 if hatch else 0.0
                ax.bar(x, r["accuracy"], width=barw*0.92, color=col, alpha=self.VARIANT_ALPHA[v],
                       edgecolor=edge, linewidth=lw, hatch=hatch, zorder=2)
                ax.errorbar(x, r["accuracy"],
                            yerr=[[r["accuracy"] - r["ci_lower"]], [r["ci_upper"] - r["accuracy"]]],
                            fmt="none", ecolor=col, elinewidth=1.1, capsize=2.5, zorder=3)
                ytxt = min(102.5, r["accuracy"] + 11.0)
                ax.text(x, ytxt, f"{r['parse_rate']:.0f}", fontsize=9.7,
                        ha="center", va="bottom", rotation=0, color="#333333")
        divider = len(self.ROBUST_ORDER) - 0.5
        ax.axvline(divider, color="#666666", lw=1.0, zorder=1)
        ax.text((len(self.ROBUST_ORDER)-1)/2 - 0.75, 103.0, "parse-robust", fontsize=12.8,
                ha="center", va="bottom", color="#333333")
        ax.text((len(self.ROBUST_ORDER)+len(order)-1)/2, 107.0, "parse-fragile", fontsize=12.8,
                ha="center", va="bottom", color="#333333")
        # ax.text(-0.42, 96.5, "numbers = parse rate (%)", fontsize=12.2,
        #         ha="left", va="bottom", color="#333333")
        ax.set_xlim(-0.65, len(order)-0.35)
        ax.set_ylim(0, 108)
        ax.set_ylabel("Accuracy among parsed cases (%)")
        ax.set_xticks(xpos)
        ax.set_xticklabels(order, rotation=45, ha="right", rotation_mode="anchor", fontsize=11.1)
        self._panel_label(ax, "d", "Prompt accuracy split by parse robustness", x=-0.075, y=1.035, title_offset=0.042)

    def _panel_e(self, ax):
        order = self.ROBUST_ORDER + self.FRAGILE_ORDER
        mat = np.zeros((len(order), len(self.VARIANT_ORDER)))
        for i, model in enumerate(order):
            for j, v in enumerate(self.VARIANT_ORDER):
                mat[i, j] = self._row_prompt(model, v)["parse_rate"]
        cmap = LinearSegmentedColormap.from_list("parse_blue", ["#FFFFFF", "#08519C"])
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=100)
        ax.set_xticks(np.arange(len(self.VARIANT_ORDER)))
        ax.set_xticklabels([self.VARIANT_LABELS[v] for v in self.VARIANT_ORDER], fontsize=12.2)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(order, fontsize=11.8)
        for i in range(len(order)):
            for j in range(len(self.VARIANT_ORDER)):
                val = mat[i, j]
                color = "white" if val > 60 else "#222222"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=12.6, color=color)
                if val < 50:
                    rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, facecolor="none",
                                             edgecolor="#777777", hatch="xxx", lw=0.0)
                    ax.add_patch(rect)
        ax.axhline(len(self.ROBUST_ORDER)-0.5, color="#333333", lw=1.0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", length=0)
        ax.set_title("")
        cbar = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.018)
        cbar.ax.tick_params(labelsize=11.5)
        cbar.set_label("Parse rate (%)", fontsize=13.0)
        self._panel_label(ax, "e", "Parse-rate collapse under terse prompting", x=-0.17, y=1.035, title_offset=0.065)

    def _panel_f(self, ax):
        self._clean_axes(ax)
        ax.axvspan(0, 50, color="#cccccc", alpha=0.18, zorder=0)
        ax.axvline(50, color="#777777", lw=1.0, ls="--", zorder=1)
        for _, row in self.prompt.loc[self.prompt["variant"] == "brief"].sort_values("order_ix").iterrows():
            model = row["model"]
            col = self.CAT_COLORS[row["category"]]
            ax.scatter(row["parse_rate"], row["accuracy"], s=135, marker=self.SHAPES[row["modality"]],
                       color=col, edgecolor="white", linewidth=0.8, zorder=3)
        label_offsets = {
            "LLaVA-Med-7B": (8, 9),
            "MedGemma-27B-text": (8, 10),
            "Mistral-Small-4-119B": (8, -14),
            "MedGemma-1.5-4B": (-72, 10),
            "RAD-DINO": (-66, -8),
            "DeepSeek-R1-7B": (-82, -12),
        }
        for model, off in label_offsets.items():
            r = self._row_prompt(model, "brief")
            ax.annotate(model, xy=(r["parse_rate"], r["accuracy"]), xytext=off,
                        textcoords="offset points", fontsize=11.4, color="#333333",
                        ha="left" if off[0] >= 0 else "right", va="center",
                        arrowprops=dict(arrowstyle="-", color="#666666", lw=0.7, shrinkA=0, shrinkB=5))
        ax.text(4, 58, "low parse ->\nestimate unreliable", fontsize=13.0,
                ha="left", va="top", color="#333333",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.90))
        ax.set_xlim(-3, 105)
        ax.set_ylim(-3, 105)
        ax.set_xlabel("Terse-prompt parse rate (%)")
        ax.set_ylabel("Terse-prompt accuracy among parsed cases (%)")
        self._panel_label(ax, "f", "Formatting failure vs reasoning failure", x=-0.14, y=1.035, title_offset=0.060)

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
    Figure7ResolutionPromptRobustness().build().save()
