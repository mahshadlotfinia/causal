"""
figures/figure5_finding_grounding.py
Created May 27, 2026

Figure 5: Finding-level heterogeneity in causal grounding.

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
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D


class Figure5FindingGrounding:
    MAIN_CSV = "/PATH/all_metrics_mimic.csv"
    OUT_PDF = "/PATH/figure5_finding_grounding.pdf"

    ROW_MODELS = [
        "Gemma-4-26B",
        "MedGemma-1.5-4B",
        "GPT-5",
        "Qwen3-VL-32B",
        "RAD-DINO",
    ]
    MULTIMODAL_MODELS = [
        "Gemma-4-26B",
        "MedGemma-1.5-4B",
        "GPT-5",
        "Qwen3-VL-32B",
    ]
    FINDINGS = [
        "cardiomegaly",
        "pneumonia",
        "edema",
        "consolidation",
        "pleural_effusion",
        "pneumothorax",
        "atelectasis",
        "lung_opacity",
    ]
    GROUNDED_FINDINGS = [
        "cardiomegaly",
        "pneumonia",
        "edema",
        "consolidation",
        "pleural_effusion",
    ]
    INERT_FINDINGS = ["atelectasis", "lung_opacity"]

    FINDING_LABELS = {
        "cardiomegaly": "Cardiomegaly",
        "pneumonia": "Pneumonia",
        "edema": "Edema",
        "consolidation": "Consolidation",
        "pleural_effusion": "Pleural\neffusion",
        "pneumothorax": "Pneumo-\nthorax",
        "atelectasis": "Atelectasis",
        "lung_opacity": "Lung\nopacity",
    }
    SHORT_FINDING_LABELS = {
        "cardiomegaly": "Cardio.",
        "pneumonia": "Pneum.",
        "edema": "Edema",
        "consolidation": "Consol.",
        "pleural_effusion": "Effusion",
    }

    MODEL_COLORS = {
        "Gemma-4-26B": "#2166AC",
        "MedGemma-1.5-4B": "#1B7837",
        "GPT-5": "#542788",
        "Qwen3-VL-32B": "#D6604D",
        "RAD-DINO": "#B35806",
    }
    SHAPES = {
        "Multimodal": "o",
        "Text-only": "s",
        "Vision-only": "D",
    }

    def __init__(self, main_csv=None, out_pdf=None):
        self.main_csv = Path(main_csv or self.MAIN_CSV)
        self.out_pdf = Path(out_pdf or self.OUT_PDF)
        self.log = logging.getLogger(self.__class__.__name__)
        self._set_rcparams()
        self._load_data()
        self.fig = None
        self.cmap = LinearSegmentedColormap.from_list("cgr_blue", ["#FFFFFF", "#2166AC"])
        self.norm = Normalize(vmin=0, vmax=70)

    @staticmethod
    def _set_rcparams():
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 19,
            "axes.labelsize": 18.5,
            "axes.titlesize": 20.5,
            "xtick.labelsize": 15.5,
            "ytick.labelsize": 15.5,
            "legend.fontsize": 15.8,
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
    def _panel_label(ax, letter, title, x=-0.10, y=1.035, title_offset=0.040):
        ax.text(x, y, letter, transform=ax.transAxes,
                fontsize=28, fontweight="bold", ha="left", va="bottom")
        ax.text(x + title_offset, y, title, transform=ax.transAxes,
                fontsize=20.5, fontweight="normal", ha="left", va="bottom")

    @staticmethod
    def _clean_axes(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    @staticmethod
    def _wilson_lower_from_percent(percent, n, z=1.959963984540054):
        if pd.isna(percent) or pd.isna(n) or n <= 0:
            return np.nan
        k = int(round((percent / 100.0) * int(n)))
        n = int(n)
        phat = k / n
        denom = 1.0 + z * z / n
        center = (phat + z * z / (2 * n)) / denom
        half = z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
        return max(0.0, center - half) * 100.0

    @staticmethod
    def _fmt_model(model):
        return model

    def _require_columns(self, df, columns, name):
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {name}: {missing}")

    def _load_data(self):
        if not self.main_csv.exists():
            raise FileNotFoundError(self.main_csv)
        df = pd.read_csv(self.main_csv)
        required = ["model", "accuracy", "is_text_only", "is_vision_only"]
        for f in self.FINDINGS:
            required.extend([f"finding_cgr_{f}", f"finding_cgr_{f}_n"])
        self._require_columns(df, required, self.main_csv.name)

        # Convert all CGR fields and accuracy to percentage units.
        df["accuracy"] = df["accuracy"].astype(float) * 100.0
        for f in self.FINDINGS:
            df[f"finding_cgr_{f}"] = df[f"finding_cgr_{f}"].astype(float) * 100.0
            df[f"finding_cgr_{f}_n"] = df[f"finding_cgr_{f}_n"].astype(float)

        df["modality"] = np.select(
            [df["is_text_only"].astype(bool), df["is_vision_only"].astype(bool)],
            ["Text-only", "Vision-only"], default="Multimodal",
        )
        self.df = df

    def _row(self, model):
        rows = self.df.loc[self.df["model"] == model]
        if rows.empty:
            raise KeyError(model)
        return rows.iloc[0]

    def _finding_value(self, model, finding):
        row = self._row(model)
        val = row[f"finding_cgr_{finding}"]
        n = row[f"finding_cgr_{finding}_n"]
        return float(val) if not pd.isna(val) else np.nan, int(n) if not pd.isna(n) else 0

    def _finding_matrix(self):
        records = []
        for model in self.ROW_MODELS:
            for finding in self.FINDINGS:
                value, n = self._finding_value(model, finding)
                lower = self._wilson_lower_from_percent(value, n)
                records.append({
                    "model": model,
                    "finding": finding,
                    "cgr": value,
                    "n": n,
                    "wilson_lower": lower,
                    "significant": (not pd.isna(lower)) and lower > 0,
                    "small_n": n < 10,
                    "na": n == 0 or pd.isna(value),
                })
        return pd.DataFrame(records)

    def build(self):
        self.fig = plt.figure(figsize=(18.0, 20.0), facecolor="white")
        gs_leg = self.fig.add_gridspec(1, 1, left=0.055, right=0.985, top=0.992, bottom=0.932)
        gs_r1 = self.fig.add_gridspec(1, 1, left=0.105, right=0.970, top=0.875, bottom=0.630)
        gs_r2 = self.fig.add_gridspec(1, 2, left=0.105, right=0.970, top=0.535, bottom=0.330,
                                      width_ratios=[1.0, 1.20], wspace=0.30)
        gs_r3 = self.fig.add_gridspec(1, 1, left=0.105, right=0.970, top=0.245, bottom=0.065)

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

    def _draw_legend(self, ax):
        ax.axis("off")
        model_handles = [
            Line2D([0], [0], marker="o", lw=2.0, markersize=11.5,
                   color=self.MODEL_COLORS[m], mfc=self.MODEL_COLORS[m],
                   mec="white", mew=0.8, label=m)
            for m in self.ROW_MODELS
        ]
        marker_handles = [
            Line2D([0], [0], marker="o", lw=0, markersize=11.5,
                   mfc="#2166AC", mec="#08306B", mew=1.9,
                   label="Wilson lower > 0"),
            patches.Patch(facecolor="#CCCCCC", edgecolor="#999999", hatch="///", label="n < 10 or N/A"),
            Line2D([0], [0], marker="o", lw=0, markersize=12,
                   mfc="white", mec="#2166AC", mew=1.8, label="noise floor"),
        ]
        ax.text(0.02, 0.70, "Model colors", transform=ax.transAxes,
                fontsize=16.5, ha="left", va="center")
        ax.text(0.02, 0.28, "Cell / marker key", transform=ax.transAxes,
                fontsize=16.5, ha="left", va="center")
        leg1 = ax.legend(model_handles, [h.get_label() for h in model_handles],
                         ncol=5, frameon=False, loc="center", bbox_to_anchor=(0.56, 0.70),
                         columnspacing=1.50, handletextpad=0.45)
        ax.add_artist(leg1)
        ax.legend(marker_handles, [h.get_label() for h in marker_handles],
                  ncol=3, frameon=False, loc="center", bbox_to_anchor=(0.43, 0.28),
                  columnspacing=1.25, handletextpad=0.45)
        cax = ax.inset_axes([0.78, 0.13, 0.18, 0.22])
        cb = self.fig.colorbar(ScalarMappable(norm=self.norm, cmap=self.cmap), cax=cax,
                               orientation="horizontal")
        cb.set_label("CGR (%)", fontsize=12.5, labelpad=0.5)
        cb.ax.tick_params(labelsize=10.8, length=2)

    def _panel_a(self, ax):
        mat = self._finding_matrix()
        ax.set_xlim(-0.70, len(self.FINDINGS) - 0.50)
        ax.set_ylim(len(self.ROW_MODELS) - 0.50, -0.50)
        ax.set_aspect("auto")
        ax.set_xticks(np.arange(len(self.FINDINGS)))
        ax.set_xticklabels([self.FINDING_LABELS[f] for f in self.FINDINGS], fontsize=14.3)
        ax.set_yticks(np.arange(len(self.ROW_MODELS)))
        ax.set_yticklabels([self._fmt_model(m) for m in self.ROW_MODELS], fontsize=15.5)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for yi, model in enumerate(self.ROW_MODELS):
            # colored tab on the left to tie heatmap rows to model colors used in panels c and d
            ax.add_patch(patches.Rectangle((-0.62, yi - 0.42), 0.13, 0.84,
                                           facecolor=self.MODEL_COLORS[model], edgecolor="none", zorder=3))
            for xi, finding in enumerate(self.FINDINGS):
                rec = mat[(mat["model"] == model) & (mat["finding"] == finding)].iloc[0]
                val = rec["cgr"]
                n = int(rec["n"])
                na = bool(rec["na"])
                small_n = bool(rec["small_n"])
                significant = bool(rec["significant"]) and not small_n and not na

                if na:
                    face = "#D9D9D9"
                else:
                    face = self.cmap(self.norm(min(max(val, 0), 70)))
                rect = patches.Rectangle((xi - 0.5, yi - 0.5), 1.0, 1.0,
                                         facecolor=face, edgecolor="#FFFFFF", linewidth=1.2, zorder=1)
                ax.add_patch(rect)

                if significant:
                    ax.add_patch(patches.Rectangle((xi - 0.5, yi - 0.5), 1.0, 1.0,
                                                   facecolor="none", edgecolor="#08306B",
                                                   linewidth=2.2, zorder=4))
                if small_n:
                    ax.add_patch(patches.Rectangle((xi - 0.5, yi - 0.5), 1.0, 1.0,
                                                   facecolor="none", edgecolor="#999999",
                                                   linewidth=0.0, hatch="///", zorder=5))
                    ax.plot([xi - 0.39, xi + 0.39], [yi - 0.39, yi + 0.39],
                            color="#777777", lw=1.0, zorder=6)
                    ax.plot([xi - 0.39, xi + 0.39], [yi + 0.39, yi - 0.39],
                            color="#777777", lw=1.0, zorder=6)

                if na:
                    ax.text(xi, yi - 0.04, "N/A", fontsize=15.5, color="#555555",
                            ha="center", va="center", zorder=7)
                    ax.text(xi, yi + 0.27, f"n={n}", fontsize=10.8, color="#555555",
                            ha="center", va="center", zorder=7)
                else:
                    text_color = "white" if val >= 40 else "#222222"
                    ax.text(xi, yi - 0.12, f"{val:.1f}", fontsize=14.6,
                            color=text_color, ha="center", va="center", zorder=7)
                    ax.text(xi, yi + 0.23, f"n={n}", fontsize=10.8,
                            color=text_color, ha="center", va="center", zorder=7)

        # visual separators between grounded, pneumothorax, and inert columns
        ax.axvline(4.5, color="#666666", lw=0.8, alpha=0.55, zorder=8)
        ax.axvline(5.5, color="#666666", lw=0.8, alpha=0.55, zorder=8)
        ax.text(2.0, -0.72, "grounded block", ha="center", va="bottom",
                fontsize=14.2, color="#333333")
        ax.text(5.0, -0.72, "mixed\nsmall-n", ha="center", va="bottom",
                fontsize=13.2, color="#555555")
        ax.text(6.5, -0.72, "inert findings", ha="center", va="bottom",
                fontsize=14.2, color="#333333")

        self._panel_label(ax, "a", "Per-finding causal grounding and sample size",
                          x=-0.095, y=1.120, title_offset=0.045)

    def _max_cgr(self, model, findings):
        vals = []
        which = []
        for f in findings:
            v, n = self._finding_value(model, f)
            if not pd.isna(v) and n > 0:
                vals.append(v)
                which.append(f)
        if not vals:
            return np.nan, None
        idx = int(np.nanargmax(vals))
        return float(vals[idx]), which[idx]

    def _panel_b(self, ax):
        self._clean_axes(ax)
        models = self.ROW_MODELS
        y = np.arange(len(models))
        ax.axvline(0, color="#BBBBBB", lw=0.9, zorder=0)

        for i, model in enumerate(models):
            col = self.MODEL_COLORS[model]
            inert_max, inert_finding = self._max_cgr(model, self.INERT_FINDINGS)
            grounded_max, grounded_finding = self._max_cgr(model, self.GROUNDED_FINDINGS)
            if pd.isna(inert_max) or pd.isna(grounded_max):
                self.log.warning("Skipping panel b for %s because of NaN", model)
                continue
            ax.plot([inert_max, grounded_max], [i, i], color=col, lw=2.4, alpha=0.80, zorder=2)
            ax.scatter(inert_max, i, s=120, facecolor="white", edgecolor=col,
                       linewidth=1.9, zorder=3)
            ax.scatter(grounded_max, i, s=130, facecolor=col, edgecolor="white",
                       linewidth=0.8, zorder=4)
            ax.text(grounded_max + 1.0, i, f"{grounded_max:.1f}", fontsize=12.8,
                    ha="left", va="center", color="#333333")

        gemma_inert, gemma_finding = self._max_cgr("Gemma-4-26B", self.INERT_FINDINGS)
        ax.annotate(f"Gemma exception:\n{gemma_finding.replace('_', ' ')} {gemma_inert:.1f}%",
                    xy=(gemma_inert, 0), xytext=(35.5, 0.70), textcoords="data",
                    fontsize=13.1, color="#333333", ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8,
                                    shrinkA=0, shrinkB=5))
        ax.text(1.0, -0.52, "hollow: max inert", fontsize=13.0,
                ha="left", va="bottom", color="#333333")
        ax.text(31.0, -0.52, "filled: max grounded", fontsize=13.0,
                ha="left", va="bottom", color="#333333")

        ax.set_xlim(-1, 70)
        ax.set_ylim(-0.70, len(models) - 0.30)
        ax.set_yticks(y)
        ax.set_yticklabels([self._fmt_model(m) for m in models], fontsize=13.8)
        ax.invert_yaxis()
        ax.set_xlabel("Maximum finding-level CGR (%)")
        self._panel_label(ax, "b", "Inert findings vs grounded block",
                          x=-0.18, y=1.035, title_offset=0.065)

    def _panel_c(self, ax):
        self._clean_axes(ax)
        findings = self.GROUNDED_FINDINGS
        models = self.MULTIMODAL_MODELS
        x = np.arange(len(findings))
        ranks = pd.DataFrame(index=models, columns=findings, dtype=float)
        values = pd.DataFrame(index=models, columns=findings, dtype=float)

        for f in findings:
            vals = {m: self._finding_value(m, f)[0] for m in models}
            ser = pd.Series(vals, dtype=float)
            # Higher CGR = better rank. Ties are kept as average rank.
            r = ser.rank(ascending=False, method="average")
            for m in models:
                ranks.loc[m, f] = r.loc[m]
                values.loc[m, f] = ser.loc[m]

        for model in models:
            col = self.MODEL_COLORS[model]
            ax.plot(x, ranks.loc[model, findings].astype(float), color=col, lw=2.4,
                    marker="o", markersize=8.5, markeredgecolor="white", markeredgewidth=0.8,
                    zorder=3, label=model)
            ax.text(x[-1] + 0.08, float(ranks.loc[model, findings[-1]]), model,
                    fontsize=11.9, ha="left", va="center", color=col)

        ax.annotate("Gemma top pneumonia\nweakest consolidation",
                    xy=(1, ranks.loc["Gemma-4-26B", "pneumonia"]),
                    xytext=(0.45, 0.75), textcoords="data",
                    fontsize=12.5, ha="left", va="center", color=self.MODEL_COLORS["Gemma-4-26B"],
                    arrowprops=dict(arrowstyle="-", color="#666666", lw=0.7,
                                    shrinkA=0, shrinkB=5))
        ax.annotate("MedGemma top edema\nthird cardiomegaly",
                    xy=(2, ranks.loc["MedGemma-1.5-4B", "edema"]),
                    xytext=(1.8, 1.50), textcoords="data",
                    fontsize=12.5, ha="left", va="center", color=self.MODEL_COLORS["MedGemma-1.5-4B"],
                    arrowprops=dict(arrowstyle="-", color="#666666", lw=0.7,
                                    shrinkA=0, shrinkB=5))
        ax.annotate("GPT-5 top cardiomegaly\nweakest pneumonia",
                    xy=(0, ranks.loc["GPT-5", "cardiomegaly"]),
                    xytext=(3.00, 3.65), textcoords="data",
                    fontsize=12.5, ha="left", va="center", color=self.MODEL_COLORS["GPT-5"],
                    arrowprops=dict(arrowstyle="-", color="#666666", lw=0.7,
                                    shrinkA=0, shrinkB=5))
        ax.text(0.02, 0.02, "RAD-DINO excluded from ranks", transform=ax.transAxes,
                fontsize=12.5, ha="left", va="bottom", color="#333333", style="italic")

        ax.set_xlim(-0.25, len(findings) - 0.35)
        ax.set_ylim(4.35, 0.65)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_ylabel("Rank (1 = highest CGR)", labelpad=4)
        ax.set_xticks(x)
        ax.set_xticklabels([self.SHORT_FINDING_LABELS[f] for f in findings], fontsize=13.0)
        self._panel_label(ax, "c", "Rank inversion across grounded findings",
                          x=-0.16, y=1.035, title_offset=0.060)

    def _panel_d(self, ax):
        self._clean_axes(ax)
        findings = self.GROUNDED_FINDINGS
        y = np.arange(len(findings))
        rad_model = "RAD-DINO"
        multimodal = self.MULTIMODAL_MODELS

        for i, f in enumerate(findings):
            rad_val, rad_n = self._finding_value(rad_model, f)
            mm_vals = np.array([self._finding_value(m, f)[0] for m in multimodal], dtype=float)
            mm_mean = float(np.nanmean(mm_vals))
            mm_max = float(np.nanmax(mm_vals))
            max_model = multimodal[int(np.nanargmax(mm_vals))]
            min_mm = float(np.nanmin(mm_vals))

            if pd.isna(rad_val):
                # ax.text(0.6, i, "RAD-DINO n/a", fontsize=12.5,
                #         ha="left", va="center", color="#555555")
                rad_plot = 0.0
                rad_alpha = 0.20
            else:
                rad_plot = rad_val
                rad_alpha = 1.0
                ax.scatter(rad_plot, i, s=135, marker="D", color=self.MODEL_COLORS[rad_model],
                           edgecolor="white", linewidth=0.8, alpha=rad_alpha, zorder=4)

            ax.plot([rad_plot, mm_mean], [i, i], color="#888888", lw=2.0,
                    alpha=0.75, zorder=1)
            ax.plot([min_mm, mm_max], [i, i], color="#2166AC", lw=7.0,
                    alpha=0.10, solid_capstyle="round", zorder=0)
            ax.scatter(mm_mean, i, s=150, marker="o", facecolor="white", edgecolor="#2166AC",
                       linewidth=2.0, zorder=3)
            ax.scatter(mm_max, i, s=135, marker="o", color=self.MODEL_COLORS[max_model],
                       edgecolor="white", linewidth=0.8, zorder=4)
            ax.text(mm_mean + 1.0, i + 0.16, f"mean {mm_mean:.1f}", fontsize=12.2,
                    ha="left", va="center", color="#333333")
            ax.text(mm_max + 1.0, i + 0.17, f"max {mm_max:.1f}", fontsize=12.2,
                    ha="left", va="center", color=self.MODEL_COLORS[max_model])

        acc = float(self._row(rad_model)["accuracy"])
        rad_med = np.nanmedian([self._finding_value(rad_model, f)[0] for f in findings])
        ax.annotate(f"RAD-DINO accuracy {acc:.1f}%\nbut median grounded CGR {rad_med:.1f}%",
                    xy=(self._finding_value(rad_model, "consolidation")[0], findings.index("consolidation")),
                    xytext=(31, 4.25), textcoords="data", fontsize=13.4,
                    ha="left", va="center", color="#333333",
                    arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8,
                                    shrinkA=0, shrinkB=5))
        ax.text(1.0, -0.58, "diamond: RAD-DINO", fontsize=12.8,
                ha="left", va="bottom", color="#333333")
        ax.text(24.0, -0.58, "open circle: multimodal mean; filled circle: multimodal max",
                fontsize=12.8, ha="left", va="bottom", color="#333333")

        ax.set_xlim(-1, 72)
        ax.set_ylim(-0.75, len(findings) - 0.35)
        ax.set_yticks(y)
        ax.set_yticklabels([self.FINDING_LABELS[f].replace("\n", " ") for f in findings], fontsize=14.0)
        ax.invert_yaxis()
        ax.set_xlabel("Finding-level CGR (%)")
        self._panel_label(ax, "d", "RAD-DINO vs multimodal grounding on grounded findings",
                          x=-0.090, y=1.035, title_offset=0.040)

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
    Figure5FindingGrounding().build().save()
