"""
figures/figure7_radiologist_analysis.py
Created June 4, 2026

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


class Figure8RadiologistAnalysis:
    READER_CSV = "/PATH/analysis/all_reader_metrics.csv"
    PAIRED_CSV = "/PATH/analysis/task_D_failure_modes.csv"
    HUMAN_FINDING_CSV = "/PATH/analysis/per_finding_human.csv"
    BOX_FINDING_CSV = "/PATH/analysis/task_A_per_finding.csv"
    MODEL_CSV = "/PATH/all_metrics_mimic.csv"
    OUT_PDF = "/PATH/figure7_radiologist_analysis.pdf"

    CAT_COLORS = {
        "Uses image": "#2166AC",
        "Ignores image": "#B2182B",
        "Unstable": "#E08214",
        "Other": "#777777",
    }
    RAD_COLOR = "#333333"
    SHAPES = {
        "Multimodal": "o",
        "Text-only": "s",
        "Vision-only": "D",
        "Radiologist": "*",
    }
    PREFERRED_MODEL_ORDER = [
        "Gemma-4-26B", "GPT-5", "Qwen3-VL-32B", "MedGemma-27B-text",
        "RAD-DINO", "MedGemma-1.5-4B", "LLaVA-Med-7B",
        "DeepSeek-R1-7B", "Mistral-Small-4-119B",
    ]
    USES_IMAGE_MODELS = ["Gemma-4-26B", "MedGemma-1.5-4B", "GPT-5", "Qwen3-VL-32B", "RAD-DINO"]
    FINDINGS = ["cardiomegaly", "pneumonia", "edema", "consolidation",
                "pleural_effusion", "pneumothorax", "atelectasis", "lung_opacity"]
    FINDING_LABELS = {
        "cardiomegaly": "cardiomegaly",
        "pneumonia": "pneumonia",
        "edema": "edema",
        "consolidation": "consolidation",
        "pleural_effusion": "pleural\neffusion",
        "pneumothorax": "pneumothorax",
        "atelectasis": "atelectasis",
        "lung_opacity": "lung\nopacity",
    }

    def __init__(self, out_pdf=None):
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
            "axes.labelsize": 17.5,
            "axes.titlesize": 19.5,
            "xtick.labelsize": 14.5,
            "ytick.labelsize": 14.2,
            "legend.fontsize": 15.0,
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
    def _panel_label(ax, letter, title, x=-0.12, y=1.035, title_offset=0.05):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=25, fontweight="bold",
                ha="left", va="bottom")
        ax.text(x + title_offset, y, title, transform=ax.transAxes, fontsize=19.5,
                ha="left", va="bottom")

    @staticmethod
    def _wilson(p, n, z=1.96):
        if n is None or pd.isna(n) or n <= 0 or pd.isna(p):
            return (np.nan, np.nan)
        p = min(max(float(p), 0.0), 1.0)
        n = float(n)
        denom = 1 + z * z / n
        centre = p + z * z / (2 * n)
        half = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
        return ((centre - half) / denom, (centre + half) / denom)

    @staticmethod
    def _fmt_p(p):
        if pd.isna(p):
            return "p n/a"
        if p < 0.001:
            return "p < 0.001"
        return f"p = {p:.3f}"

    @staticmethod
    def _pct(x):
        return 100.0 * float(x)

    @staticmethod
    def _short_model(m):
        return m

    def _require(self, df, cols, name):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {name}: {missing}")

    def _load_data(self):
        self.reader = pd.read_csv(self.READER_CSV)
        self.paired = pd.read_csv(self.PAIRED_CSV)
        self.failure = pd.read_csv(self.FAILURE_CSV)
        self.human_find = pd.read_csv(self.HUMAN_FINDING_CSV)
        self.box_find = pd.read_csv(self.BOX_FINDING_CSV)
        self.model = pd.read_csv(self.MODEL_CSV)

        self._require(self.reader, ["task", "accuracy_point", "accuracy_ci_lower", "accuracy_ci_upper",
                                    "CGR_point", "CGR_ci_lower", "CGR_ci_upper",
                                    "n_accurate", "n_partial", "n_inaccurate", "n_cannot_tell",
                                    "percent_agreement", "percent_agreement_ci_lower", "percent_agreement_ci_upper",
                                    "cohens_kappa", "cohens_kappa_ci_lower", "cohens_kappa_ci_upper",
                                    "weighted_kappa_confidence", "weighted_kappa_confidence_ci_lower",
                                    "weighted_kappa_confidence_ci_upper", "n_shared_displays"], "all_reader_metrics")
        self._require(self.paired, ["model", "metric", "human_value", "model_value", "diff",
                                    "diff_ci_lower", "diff_ci_upper", "p_value_fdr", "n_paired"], "human_vs_model_paired")
        self._require(self.failure, ["model", "category", "n", "count", "fraction", "wilson_lower", "wilson_upper"], "task_D_failure_modes")
        self._require(self.human_find, ["finding", "metric", "point", "ci_lower", "ci_upper", "n", "reader"], "per_finding_human")
        self._require(self.box_find, ["finding", "n", "frac_accurate", "wilson_ci_lower", "wilson_ci_upper"], "task_A_per_finding")
        self._require(self.model, ["model", "accuracy", "accuracy_ci_lower", "accuracy_ci_upper",
                                   "CGR", "CGR_ci_lower", "CGR_ci_upper", "UAR",
                                   "irrelevant_stable", "irrelevant_stable_ci_lower", "irrelevant_stable_ci_upper",
                                   "is_text_only", "is_vision_only"], "all_metrics_mimic")

        self.model["modality"] = np.select(
            [self.model["is_text_only"].astype(bool), self.model["is_vision_only"].astype(bool)],
            ["Text-only", "Vision-only"], default="Multimodal",
        )
        self.model["category"] = self.model.apply(self._assign_category, axis=1)
        self.model["order_ix"] = self.model["model"].map({m: i for i, m in enumerate(self.PREFERRED_MODEL_ORDER)})

    def _assign_category(self, row):
        cgr = row["CGR"] * 100.0
        cgr_lo = row["CGR_ci_lower"] * 100.0
        uar = row["UAR"] * 100.0
        isv = row["irrelevant_stable"] * 100.0
        if pd.isna(cgr) or pd.isna(cgr_lo) or pd.isna(uar) or pd.isna(isv):
            return "Other"
        if isv < 70.0:
            return "Unstable"
        if np.isclose(cgr, 0.0) and np.isclose(uar, 100.0) and np.isclose(isv, 100.0):
            return "Ignores image"
        if (cgr > 0.0) and (cgr_lo > 0.0) and (isv >= 90.0):
            return "Uses image"
        return "Other"

    def _model_row(self, model):
        rows = self.model.loc[self.model["model"] == model]
        if rows.empty:
            raise KeyError(model)
        return rows.iloc[0]

    def _paired_rows(self, metric):
        return self.paired.loc[self.paired["metric"] == metric].copy()

    def _reader_task_row(self, task):
        rows = self.reader.loc[self.reader["task"] == task]
        if rows.empty:
            raise KeyError(task)
        return rows.iloc[0]

    def _reader_summary(self):
        return self._reader_task_row("B_finding_presence")

    def _marker_for_model(self, model):
        row = self._model_row(model)
        return self.SHAPES.get(row["modality"], "o")

    def _color_for_model(self, model):
        return self.CAT_COLORS.get(self._model_row(model)["category"], self.CAT_COLORS["Other"])

    def build(self):
        self.fig = plt.figure(figsize=(18.5, 21.8), facecolor="white")

        gs_leg = self.fig.add_gridspec(1, 1, left=0.045, right=0.985, top=0.992, bottom=0.936)
        gs_r1 = self.fig.add_gridspec(1, 2, left=0.090, right=0.985, top=0.895, bottom=0.670,
                                      width_ratios=[1.0, 1.0], wspace=0.34)
        gs_r2 = self.fig.add_gridspec(1, 3, left=0.070, right=0.985, top=0.590, bottom=0.360,
                                      width_ratios=[0.95, 1.05, 0.85], wspace=0.34)
        gs_r3 = self.fig.add_gridspec(1, 2, left=0.090, right=0.985, top=0.270, bottom=0.055,
                                      width_ratios=[0.85, 1.15], wspace=0.30)

        self._draw_legend(self.fig.add_subplot(gs_leg[0, 0]))
        self._panel_a(self.fig.add_subplot(gs_r1[0, 0]))
        self._panel_b(self.fig.add_subplot(gs_r1[0, 1]))
        self._panel_c(self.fig.add_subplot(gs_r2[0, 0]))
        self._panel_d(self.fig.add_subplot(gs_r2[0, 1]))
        self._panel_e(self.fig.add_subplot(gs_r2[0, 2]))
        self._panel_f(self.fig.add_subplot(gs_r3[0, 0]))
        self._panel_g(self.fig.add_subplot(gs_r3[0, 1]))
        return self

    def _draw_legend(self, ax):
        ax.axis("off")
        cat_handles = [
            Line2D([0], [0], marker="o", lw=0, markersize=15,
                   mfc=self.CAT_COLORS[c], mec="white", mew=0.8, label=c)
            for c in ["Uses image", "Ignores image", "Unstable"]
        ]
        shape_handles = [
            Line2D([0], [0], marker=m, lw=0, markersize=14 if name != "Radiologist" else 17,
                   mfc=self.RAD_COLOR if name == "Radiologist" else "#777777",
                   mec="white", mew=0.8, label=name)
            for name, m in self.SHAPES.items()
        ]
        failure_colors = {
            "Ambiguous": "#8DD3C7",
            "Image quality": "#FDB462",
            "Plausible confounder": "#B3DE69",
            "Clear model failure": "#FB8072",
            "Other": "#BEBADA",
        }
        failure_handles = [
            patches.Patch(facecolor=failure_colors[c], edgecolor="white", label=c)
            for c in ["Ambiguous", "Image quality", "Plausible confounder", "Clear model failure", "Other"]
        ]

        ax.text(0.02, 0.72, "Fill color", transform=ax.transAxes,
                fontsize=15.5, ha="left", va="center")
        leg1 = ax.legend(cat_handles, [h.get_label() for h in cat_handles],
                         ncol=3, frameon=False, loc="center", bbox_to_anchor=(0.30, 0.72),
                         columnspacing=1.6, handletextpad=0.45)
        ax.add_artist(leg1)
        leg2 = ax.legend(shape_handles, [h.get_label() for h in shape_handles],
                         ncol=4, frameon=False, loc="center", bbox_to_anchor=(0.76, 0.72),
                         columnspacing=1.15, handletextpad=0.45)
        ax.add_artist(leg2)

        ax.text(0.02, 0.24, "Failure mode", transform=ax.transAxes,
                fontsize=15.5, ha="left", va="center")
        ax.legend(failure_handles, [h.get_label() for h in failure_handles],
                  ncol=5, frameon=False, loc="center", bbox_to_anchor=(0.60, 0.24),
                  fontsize=15.0, columnspacing=1.35, handletextpad=0.42)

    def _dumbbell_panel(self, ax, metric, letter, title, xlim, band, order_by="model_value",
                        right_label="diff", margin_x=None, include_mistral_cgr=False):
        self._clean_axes(ax)
        rows = self._paired_rows(metric)
        if include_mistral_cgr and metric == "CGR" and "Mistral-Small-4-119B" not in set(rows["model"]):
            mr = self._model_row("Mistral-Small-4-119B")
            reader = self._reader_summary()
            rows = pd.concat([rows, pd.DataFrame([{
                "model": "Mistral-Small-4-119B",
                "metric": "CGR",
                "human_value": reader["CGR_point"],
                "model_value": mr["CGR"],
                "diff": reader["CGR_point"] - mr["CGR"],
                "diff_ci_lower": np.nan,
                "diff_ci_upper": np.nan,
                "p_value_fdr": np.nan,
                "n_paired": mr["CGR_n"],
                "source_note": "main_audit_model"
            }])], ignore_index=True)
        rows["model_pct"] = rows["model_value"].astype(float) * 100
        rows["human_pct"] = rows["human_value"].astype(float) * 100
        rows = rows.sort_values(order_by, ascending=False).reset_index(drop=True)
        y = np.arange(len(rows))
        if margin_x is None:
            margin_x = xlim[1] - 4

        ax.axvspan(band[0], band[1], color=self.RAD_COLOR, alpha=0.12, zorder=0)
        ax.text(band[1], -0.72, "reader band", fontsize=14, color=self.RAD_COLOR,
                ha="right", va="bottom")

        for i, row in rows.iterrows():
            model = row["model"]
            col = self._color_for_model(model)
            marker = self._marker_for_model(model)
            mv = row["model_pct"]
            hv = row["human_pct"]
            n = row["n_paired"]
            lo, hi = self._wilson(row["model_value"], n)
            lo, hi = lo * 100, hi * 100

            ax.plot([hv, mv], [i, i], color="#777777", lw=1.0, alpha=0.9, zorder=1)
            ax.scatter(hv, i, s=150, marker="*", color=self.RAD_COLOR,
                       edgecolor="white", linewidth=0.7, zorder=4)
            if np.isfinite(lo) and np.isfinite(hi):
                ax.errorbar(mv, i, xerr=[[mv - lo], [hi - mv]], fmt="none",
                            ecolor=col, elinewidth=1.3, capsize=3, zorder=2)
            ax.scatter(mv, i, s=120, marker=marker, color=col,
                       edgecolor="white", linewidth=0.8, zorder=5)

            if metric == "accuracy":
                delta = (row["model_value"] - row["human_value"]) * 100
                text = f"{delta:+.1f}, {self._fmt_p(row['p_value_fdr'])}"
            else:
                gap = (row["human_value"] - row["model_value"]) * 100
                if pd.isna(row["p_value_fdr"]):
                    text = f"gap {gap:+.1f}, no paired p"
                else:
                    text = f"gap {gap:+.1f}, {self._fmt_p(row['p_value_fdr'])}"

            y_text = i
            va = "center"
            if metric == "accuracy" and model in {"LLaVA-Med-7B", "RAD-DINO"}:
                y_text = i - 0.18
                va = "bottom"
            if metric == "CGR" and model in {"MedGemma-1.5-4B", "Mistral-Small-4-119B"}:
                y_text = i - 0.18
                va = "bottom"
            ax.text(margin_x, y_text, text, fontsize=11.5, color="#333333",
                    ha="right", va=va)

        ax.set_xlim(*xlim)
        ax.set_ylim(-0.85, len(rows) - 0.15)
        ax.set_yticks(y)
        ax.set_yticklabels([self._short_model(m) for m in rows["model"]], fontsize=12.6)
        ax.invert_yaxis()
        ax.set_xlabel(f"{'Accuracy' if metric == 'accuracy' else 'CGR'} (%)")
        self._panel_label(ax, letter, title, x=-0.13, y=1.035, title_offset=0.055)
        return rows

    def _panel_a(self, ax):
        self._dumbbell_panel(
            ax, "accuracy", "a", "Accuracy vs radiologist",
            xlim=(0, 116), band=(72.5, 90.0), order_by="model_pct", margin_x=115.0,
        )

    def _panel_b(self, ax):
        self._dumbbell_panel(
            ax, "CGR", "b", "Grounding vs radiologist",
            xlim=(0, 68), band=(13.8, 33.8), order_by="model_pct", margin_x=67.0,
            include_mistral_cgr=True,
        )

    def _panel_c(self, ax):
        self._clean_axes(ax)
        acc_rows = self._paired_rows("accuracy").set_index("model")
        cgr_rows = self._paired_rows("CGR").set_index("model")
        models = self.PREFERRED_MODEL_ORDER
        reader = self._reader_summary()
        r_acc = reader["accuracy_point"] * 100
        r_cgr = reader["CGR_point"] * 100

        ax.axvline(r_acc, color=self.RAD_COLOR, lw=1.0, alpha=0.45, zorder=0)
        ax.axhline(r_cgr, color=self.RAD_COLOR, lw=1.0, alpha=0.45, zorder=0)
        ax.scatter(r_acc, r_cgr, s=260, marker="*", color=self.RAD_COLOR,
                   edgecolor="white", linewidth=0.8, zorder=6)
        ax.text(r_acc + 1.0, r_cgr + 5.0, "radiologist", fontsize=12.4,
                ha="left", va="bottom", color=self.RAD_COLOR)

        for model in models:
            if model not in acc_rows.index:
                continue
            acc = acc_rows.loc[model, "model_value"] * 100
            n_acc = acc_rows.loc[model, "n_paired"]
            acc_lo, acc_hi = self._wilson(acc_rows.loc[model, "model_value"], n_acc)
            acc_lo, acc_hi = acc_lo * 100, acc_hi * 100

            if model in cgr_rows.index:
                cgr = cgr_rows.loc[model, "model_value"] * 100
                n_cgr = cgr_rows.loc[model, "n_paired"]
                cgr_lo, cgr_hi = self._wilson(cgr_rows.loc[model, "model_value"], n_cgr)
                cgr_lo, cgr_hi = cgr_lo * 100, cgr_hi * 100
            elif model == "Mistral-Small-4-119B":
                mr = self._model_row(model)
                cgr = mr["CGR"] * 100
                cgr_lo, cgr_hi = mr["CGR_ci_lower"] * 100, mr["CGR_ci_upper"] * 100
            else:
                continue
            col = self._color_for_model(model)
            marker = self._marker_for_model(model)
            ax.errorbar(acc, cgr,
                        xerr=[[acc - acc_lo], [acc_hi - acc]],
                        yerr=[[cgr - cgr_lo], [cgr_hi - cgr]],
                        fmt="none", ecolor=col, elinewidth=1.0, capsize=2.5, alpha=0.75, zorder=2)
            ax.scatter(acc, cgr, s=110, marker=marker, color=col, edgecolor="white",
                       linewidth=0.8, zorder=4)

        ax.text(76, 4.0, "similar accuracy\nbut little grounding", fontsize=11.8,
                color=self.CAT_COLORS["Ignores image"], ha="left", va="bottom")
        ax.text(50, 41.5, "uses-image systems\naround reader CGR", fontsize=11.8,
                color=self.CAT_COLORS["Uses image"], ha="left", va="center")
        ax.set_xlim(0, 105)
        ax.set_ylim(-2, 58)
        ax.set_xlabel("Accuracy (%)")
        ax.set_ylabel("CGR (%)")
        self._panel_label(ax, "c", "Accuracy-grounding plane", x=-0.19, y=1.035, title_offset=0.075)

    def _panel_d(self, ax):
        self._clean_axes(ax)
        data = self.box_find.copy()
        data["pct"] = data["frac_accurate"] * 100
        data["lo"] = data["wilson_ci_lower"] * 100
        data["hi"] = data["wilson_ci_upper"] * 100
        data = data.sort_values("pct", ascending=False).reset_index(drop=True)

        y = np.arange(len(data))
        col_bar = "#2C7FB8"
        ax.barh(y, data["pct"], height=0.56, color=col_bar, alpha=0.82, edgecolor="none", zorder=2)
        ax.errorbar(data["pct"], y, xerr=[data["pct"] - data["lo"], data["hi"] - data["pct"]],
                    fmt="none", ecolor=col_bar, elinewidth=1.1, capsize=3, zorder=3)

        ax.set_xlim(0, 106)
        ax.set_ylim(-0.65, len(data) - 0.35)
        ax.set_yticks(y)
        ax.set_yticklabels([self.FINDING_LABELS.get(f, f) for f in data["finding"]], fontsize=11.8)
        ax.invert_yaxis()
        ax.set_xlabel("Valid evidence boxes (%)")
        self._panel_label(ax, "d", "Box evidence validity by finding", x=-0.16, y=1.035, title_offset=0.065)

    def _panel_e(self, ax):
        self._clean_axes(ax)
        task = self.box_find.set_index("finding")
        human = self.human_find[(self.human_find["metric"] == "CGR") & (self.human_find["reader"] == "reader1")].set_index("finding")
        model_means = []
        for finding in self.FINDINGS:
            vals = []
            col = f"finding_cgr_{finding}"
            for m in self.USES_IMAGE_MODELS:
                row = self._model_row(m)
                if col in row.index and pd.notna(row[col]):
                    vals.append(row[col] * 100)
            model_means.append(np.nanmean(vals) if vals else np.nan)

        xs, ys_h, ys_m, labs = [], [], [], []
        for finding, model_mean in zip(self.FINDINGS, model_means):
            if finding not in task.index or finding not in human.index:
                continue
            xs.append(task.loc[finding, "frac_accurate"] * 100)
            ys_h.append(human.loc[finding, "point"] * 100)
            ys_m.append(model_mean)
            labs.append(finding)

        ax.plot([0, 100], [0, 100], color="#DDDDDD", lw=1.0, ls="--", zorder=0)
        ax.scatter(xs, ys_m, s=85, marker="o", color="#777777", edgecolor="white", linewidth=0.8,
                   label="uses-image model mean", zorder=3)
        ax.scatter(xs, ys_h, s=160, marker="*", color=self.RAD_COLOR, edgecolor="white", linewidth=0.7,
                   label="radiologist", zorder=4)
        label_offsets = {
            "cardiomegaly": (-12, 10),
            "pneumonia": (10, -5),
            "edema": (8, 2),
            "consolidation": (8, 7),
            "pleural_effusion": (8, -7),
            "pneumothorax": (-45, 4),
            "atelectasis": (8, 0),
            "lung_opacity": (8, -4),
        }
        for x, yh, ym, lab in zip(xs, ys_h, ys_m, labs):
            off = label_offsets.get(lab, (6, 6))
            ax.annotate(self.FINDING_LABELS.get(lab, lab).replace("\n", " "), xy=(x, max(yh, ym)),
                        xytext=off, textcoords="offset points", fontsize=9.8,
                        ha="left" if off[0] >= 0 else "right", va="center", color="#333333")

        ax.set_xlim(-2, 104)
        ax.set_ylim(-2, 66)
        ax.set_xlabel("Box validity (%)")
        ax.set_ylabel("Per-finding CGR (%)")
        ax.legend(frameon=False, loc="upper left", fontsize=10.8, handletextpad=0.3)
        self._panel_label(ax, "e", "Grounding tracks box validity", x=-0.18, y=1.035, title_offset=0.070)

    def _panel_f(self, ax):
        self._clean_axes(ax)
        row = self._reader_task_row("agreement_reader1_vs_reader2")
        metrics = [
            ("percent agreement", row["percent_agreement"], row["percent_agreement_ci_lower"], row["percent_agreement_ci_upper"]),
            ("Cohen's κ", row["cohens_kappa"], row["cohens_kappa_ci_lower"], row["cohens_kappa_ci_upper"]),
            ("weighted κ", row["weighted_kappa_confidence"], row["weighted_kappa_confidence_ci_lower"], row["weighted_kappa_confidence_ci_upper"]),
        ]
        y = np.arange(len(metrics))

        # Landis-Koch style bands behind the kappa rows only.
        bands = [(0.0, 0.20, "#EEEEEE", "slight"), (0.21, 0.40, "#E0E0E0", "fair"), (0.41, 0.60, "#D0D0D0", "moderate")]
        for xmin, xmax, color, label in bands:
            ax.add_patch(patches.Rectangle((xmin, 0.55), xmax - xmin, 1.9, transform=ax.transData,
                                           facecolor=color, edgecolor="none", alpha=0.55, zorder=0))
            ax.text((xmin + xmax) / 2, 2.45, label, fontsize=14, color="#666666",
                    ha="center", va="top")

        for i, (name, val, lo, hi) in enumerate(metrics):
            ax.errorbar(val, i, xerr=[[val - lo], [hi - val]], fmt="none",
                        ecolor=self.RAD_COLOR, elinewidth=1.4, capsize=3, zorder=2)
            ax.scatter(val, i, s=100, color=self.RAD_COLOR, edgecolor="white", linewidth=0.7, zorder=3)
            ax.text(min(0.98, hi + 0.025), i, f"{val:.3f}", fontsize=11.5,
                    ha="left", va="center", color="#333333")

        ax.set_xlim(0, 1.02)
        ax.set_ylim(-0.75, len(metrics) - 0.15)
        ax.set_yticks(y)
        ax.set_yticklabels([m[0] for m in metrics], fontsize=12.3)
        ax.invert_yaxis()
        ax.set_xlabel("Agreement metric")
        self._panel_label(ax, "f", "Inter-rater agreement caveat", x=-0.20, y=1.035, title_offset=0.080)

    def _panel_g(self, ax):
        self._clean_axes(ax)
        cats = ["Ambiguous", "Image quality", "Plausible confounder", "Clear model failure", "Other"]
        colors = {
            "Ambiguous": "#8DD3C7",
            "Image quality": "#FDB462",
            "Plausible confounder": "#B3DE69",
            "Clear model failure": "#FB8072",
            "Other": "#BEBADA",
        }
        models = ["GPT-5", "Gemma-4-26B", "RAD-DINO"]
        y = np.arange(len(models))
        for i, model in enumerate(models):
            rows = self.failure[(self.failure["model"] == model) & (self.failure["category"].isin(cats))]
            left = 0.0
            n = int(rows["n"].dropna().iloc[0]) if not rows.empty else 0
            for cat in cats:
                rr = rows.loc[rows["category"] == cat]
                frac = float(rr["fraction"].iloc[0]) if not rr.empty else 0.0
                width = frac * 100
                ax.barh(i, width, left=left, height=0.58, color=colors[cat],
                        edgecolor="white", linewidth=0.8, zorder=2)
                if width >= 8:
                    ax.text(left + width / 2, i, f"{width:.0f}", fontsize=11.0,
                            ha="center", va="center", color="#222222")
                left += width
            ax.text(102, i, f"n={n}", fontsize=12.0, ha="left", va="center", color="#333333")

        ax.set_xlim(0, 110)
        ax.set_ylim(-0.65, len(models) - 0.35)
        ax.set_yticks(y)
        ax.set_yticklabels(models, fontsize=12.8)
        ax.invert_yaxis()
        ax.set_xlabel("Failure-mode composition (%)")
        self._panel_label(ax, "g", "Failure-mode composition", x=-0.10, y=1.035, title_offset=0.050)

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
    Figure8RadiologistAnalysis().build().save()
