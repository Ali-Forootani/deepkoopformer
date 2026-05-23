#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze a single metrics.csv file in the current directory.

This version is adapted for CSV files with columns like:
    Model, PatchLen, Horizon, Set, MSE, MAE

Important change:
    The CSV does not need a separate Backbone column. Backbone, model family,
    and variant are inferred from the Model string, e.g.:
        Koop-PatchTST (constr.)        -> Family=Koop, Backbone=PatchTST, Variant=constr.
        ODE-Autoformer                -> Family=ODE, Backbone=Autoformer, Variant=base
        DLinear                       -> Family=Baseline, Backbone=DLinear, Variant=baseline

Outputs are saved in ./single_run_analysis/
"""

import os
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


# =========================================================
# USER SETTINGS
# =========================================================
CSV_FILE = "metrics.csv"
OUTPUT_DIR = "single_run_analysis"
SAVE_DPI = 600
FIGSIZE = (15, 7)
METRICS = ["MSE", "MAE", "RMSE"]

# Set to None to use Train + Test. For most papers, "Test" is usually best.
SET_FILTER = "Test"

# Exclude selected backbones from all plots
EXCLUDED_BACKBONES = ["SSM"]

PREFERRED_COLUMNS = [
    "Model",
    "Backbone",
    "Family",
    "Variant",
    "ModelDisplay",
    "PatchLen",
    "Horizon",
    "Set",
    "BestValMSE",
    "BestEpoch",
    "MSE",
    "MAE",
    "RMSE",
    "NumLayers",
    "LyapWeight",
]

BACKBONE_ORDER_DISPLAY = [
    "LSTM",
    "DLinear",
    "SSM",
    "PatchTST",
    "Autoformer",
    "Informer",
]

FAMILY_ORDER = ["Baseline", "Koop", "ODE"]

VARIANT_ORDER = [
    "baseline",
    "constr.",
    "unconstr.",
    "base",
    "learn:scalar",
    "learn:permode",
    "learn:mlp",
    "learn:lowrank16",
]

BACKBONE_COLORS = {
    "LSTM": "#4C72B0",
    "DLinear": "#C44E52",
    "SSM": "#8172B2",
    "PatchTST": "#55A868",
    "Autoformer": "#DA8BC3",
    "Informer": "#8C8C8C",
}

BACKBONE_HATCHES = {
    "LSTM": "///",
    "DLinear": "...",
    "SSM": "++",
    "PatchTST": "xxx",
    "Autoformer": "**",
    "Informer": "---",
}

FAMILY_COLORS = {
    "Baseline": "#4C72B0",
    "Koop": "#DD8452",
    "ODE": "#55A868",
}

FAMILY_MARKERS = {
    "Baseline": "o",
    "Koop": "s",
    "ODE": "X",
}

FAMILY_LINESTYLES = {
    "Baseline": "",
    "Koop": (4, 2),
    "ODE": (1, 2),
}


# =========================================================
# READER / MODEL PARSING
# =========================================================
def read_ragged_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError("CSV file is empty.")

        # Some pasted files may start with a shell line like: cat metrics.csv
        if len(header) == 1 and header[0].strip().startswith("cat "):
            header = next(reader, None)
            if header is None:
                raise ValueError("CSV header was not found after the pasted shell line.")

        header = [str(h).strip() for h in header]
        n_cols = len(header)

        for row in reader:
            if not row or all(str(x).strip() == "" for x in row):
                continue
            row = row[:n_cols]
            if len(row) < n_cols:
                row += [""] * (n_cols - len(row))
            rows.append(row)

    df = pd.DataFrame(rows, columns=header)

    for col in ["Model", "Backbone", "Set"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    numeric_cols = [
        "PatchLen", "Horizon", "BestValMSE", "BestEpoch",
        "MSE", "MAE", "RMSE", "LyapWeight",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Model" not in df.columns:
        raise ValueError("Column 'Model' is required in metrics.csv")

    parsed = df["Model"].apply(parse_model_name).apply(pd.Series)
    for col in parsed.columns:
        df[col] = parsed[col]

    if "Backbone" in df.columns:
        df["BackboneDisplay"] = df["Backbone"].fillna(df["BackboneDisplay"])

    if "RMSE" not in df.columns and "MSE" in df.columns:
        df["RMSE"] = df["MSE"] ** 0.5

    keep_cols = [c for c in PREFERRED_COLUMNS + ["BackboneDisplay"] if c in df.columns]
    df = df[keep_cols].copy()

    if SET_FILTER is not None and "Set" in df.columns:
        df = df[df["Set"].str.lower() == SET_FILTER.lower()].copy()

    required = ["ModelDisplay", "BackboneDisplay", "PatchLen", "Horizon", "MSE", "MAE"]
    df = df.dropna(subset=[c for c in required if c in df.columns]).reset_index(drop=True)


    if EXCLUDED_BACKBONES:
        df = df[~df["BackboneDisplay"].isin(EXCLUDED_BACKBONES)].copy()

    if df.empty:
        raise ValueError("No usable rows found in metrics.csv after parsing/filtering.")

    return df


def parse_model_name(model_name):
    name = str(model_name).strip()

    # Baselines in this CSV.
    baseline_backbones = {"LSTM", "DLinear", "SSM"}
    if name in baseline_backbones:
        return {
            "Family": "Baseline",
            "BackboneDisplay": name,
            "Variant": "baseline",
            "ModelDisplay": name,
        }

    # Koop-PatchTST (learn:scalar), Koop-Autoformer (constr.), etc.
    m = re.match(r"^(Koop)-(.+?)\s*\((.+)\)$", name)
    if m:
        family, backbone, variant = m.groups()
        return {
            "Family": family,
            "BackboneDisplay": normalize_backbone(backbone),
            "Variant": variant.strip(),
            "ModelDisplay": f"{family} ({variant.strip()})",
        }

    # ODE-PatchTST, ODE-Autoformer, ODE-Informer, etc.
    m = re.match(r"^(ODE)-(.+)$", name)
    if m:
        family, backbone = m.groups()
        return {
            "Family": family,
            "BackboneDisplay": normalize_backbone(backbone),
            "Variant": "base",
            "ModelDisplay": family,
        }

    # Fallback: preserve unknown models as their own backbone/family.
    return {
        "Family": "Other",
        "BackboneDisplay": normalize_backbone(name),
        "Variant": "other",
        "ModelDisplay": name,
    }


def normalize_backbone(backbone):
    aliases = {
        "patchtst": "PatchTST",
        "autoformer": "Autoformer",
        "informer": "Informer",
        "dlinear": "DLinear",
        "ssm": "SSM",
        "lstm": "LSTM",
    }
    key = str(backbone).strip().lower()
    return aliases.get(key, str(backbone).strip())


# =========================================================
# HELPERS
# =========================================================
def ordered_values(present, preferred):
    present = list(present)
    ordered = [x for x in preferred if x in present]
    remaining = [x for x in present if x not in ordered]
    return ordered + sorted(remaining)


def get_backbone_order(df):
    return ordered_values(df["BackboneDisplay"].dropna().unique(), BACKBONE_ORDER_DISPLAY)


def get_family_order(df):
    return ordered_values(df["Family"].dropna().unique(), FAMILY_ORDER)


def get_variant_order(df):
    return ordered_values(df["Variant"].dropna().unique(), VARIANT_ORDER)


def get_model_order(df):
    # Keeps readable labels such as LSTM, DLinear, SSM, Koop (...), ODE.
    return sorted(df["ModelDisplay"].dropna().unique())


def get_backbone_palette(backbone_order):
    return {name: BACKBONE_COLORS.get(name, "#999999") for name in backbone_order}


def get_family_palette(family_order):
    return {name: FAMILY_COLORS.get(name, "#999999") for name in family_order}


def save_current_figure(filename_base):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    png_path = os.path.join(OUTPUT_DIR, f"{filename_base}.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"{filename_base}.pdf")
    plt.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight", transparent=True)
    plt.savefig(pdf_path, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def print_summary(df):
    print("\n======================")
    print("DATA SUMMARY")
    print("======================")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Set filter:", SET_FILTER)
    print("Backbones:", sorted(df["BackboneDisplay"].dropna().unique()))
    print("Families:", sorted(df["Family"].dropna().unique()))
    print("Variants:", sorted(df["Variant"].dropna().unique()))
    print("Horizons:", sorted(df["Horizon"].dropna().unique()))
    print("PatchLens:", sorted(df["PatchLen"].dropna().unique()))
    print("\nCounts by Family, Backbone, Variant:")
    print(df.groupby(["Family", "BackboneDisplay", "Variant"]).size())


def make_custom_backbone_legend(ax, hue_order):
    handles = []
    for name in hue_order:
        handles.append(
            mpl.patches.Patch(
                facecolor=BACKBONE_COLORS.get(name, "#999999"),
                edgecolor="black",
                linewidth=1.4,
                hatch=BACKBONE_HATCHES.get(name, ""),
                label=name,
                alpha=0.9,
            )
        )
    legend = ax.legend(
        handles=handles,
        title="Backbone",
        fontsize=11,
        title_fontsize=13,
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )
    legend.get_frame().set_alpha(0.95)


def stylize_violin_bodies(ax, hue_order):
    violin_bodies = [c for c in ax.collections if isinstance(c, mpl.collections.PolyCollection)]
    n_hue = len(hue_order)
    if n_hue == 0:
        return
    for i, body in enumerate(violin_bodies):
        hue_name = hue_order[i % n_hue]
        body.set_edgecolor("black")
        body.set_linewidth(1.5)
        body.set_alpha(0.9)
        hatch = BACKBONE_HATCHES.get(hue_name, "")
        if hatch:
            body.set_hatch(hatch)
    for line in ax.lines:
        line.set_color("#444444")
        line.set_linewidth(1.2)
        line.set_linestyle("--")
        line.set_alpha(0.95)


# =========================================================
# PLOTS: BACKBONE COMPARISON
# =========================================================
def plot_violin_by_family(df, metric):
    family_order = get_family_order(df)
    backbone_order = get_backbone_order(df)
    palette = get_backbone_palette(backbone_order)

    plt.figure(figsize=FIGSIZE)
    ax = sns.violinplot(
        data=df,
        x="Family",
        y=metric,
        hue="BackboneDisplay",
        order=family_order,
        hue_order=backbone_order,
        inner="quart",
        cut=0,
        linewidth=1.3,
        palette=palette,
        saturation=1,
    )
    stylize_violin_bodies(ax, backbone_order)
    make_custom_backbone_legend(ax, backbone_order)
    plt.title(f"{metric} Distribution by Family and Backbone", fontsize=20)
    plt.xlabel("Model Family", fontsize=16)
    plt.ylabel(metric, fontsize=16)
    plt.xticks(rotation=20, ha="right", fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    save_current_figure(f"violin_{metric}_family_backbone")
    plt.close()


def plot_box_by_family(df, metric):
    family_order = get_family_order(df)
    backbone_order = get_backbone_order(df)
    palette = get_backbone_palette(backbone_order)

    plt.figure(figsize=FIGSIZE)
    ax = sns.boxplot(
        data=df,
        x="Family",
        y=metric,
        hue="BackboneDisplay",
        order=family_order,
        hue_order=backbone_order,
        showfliers=True,
        palette=palette,
        linewidth=1.3,
        saturation=1,
    )

    color_to_backbone = {}
    for backbone_name in backbone_order:
        rgba = mpl.colors.to_rgba(palette[backbone_name])
        rgb_key = tuple(round(v, 3) for v in rgba[:3])
        color_to_backbone[rgb_key] = backbone_name

    for patch in ax.findobj(mpl.patches.PathPatch):
        facecolor = patch.get_facecolor()
        rgb_key = tuple(round(v, 3) for v in facecolor[:3])
        backbone_name = color_to_backbone.get(rgb_key)
        if backbone_name is None:
            continue
        patch.set_edgecolor("black")
        patch.set_linewidth(1.3)
        patch.set_alpha(0.9)
        hatch = BACKBONE_HATCHES.get(backbone_name, "")
        if hatch:
            patch.set_hatch(hatch)

    for line in ax.lines:
        line.set_color("black")
        line.set_linewidth(1.1)

    make_custom_backbone_legend(ax, backbone_order)
    plt.title(f"{metric} Box Plot by Family and Backbone - Test - Wind Speeds", fontsize=26)
    plt.xlabel("Model Family", fontsize=24)
    plt.ylabel(metric, fontsize=24)
    plt.xticks(rotation=20, ha="right", fontsize=20)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    save_current_figure(f"box_{metric}_family_backbone_wind")
    plt.close()


def plot_horizon_by_backbone(df, metric):
    family_order = get_family_order(df)
    backbone_order = get_backbone_order(df)
    palette = get_family_palette(family_order)

    df_plot = df.groupby(["BackboneDisplay", "Family", "Horizon"], as_index=False)[metric].mean()
    df_plot["BackboneDisplay"] = pd.Categorical(df_plot["BackboneDisplay"], categories=backbone_order, ordered=True)
    df_plot["Family"] = pd.Categorical(df_plot["Family"], categories=family_order, ordered=True)
    df_plot = df_plot.sort_values(["BackboneDisplay", "Family", "Horizon"])

    g = sns.relplot(
        data=df_plot,
        kind="line",
        x="Horizon",
        y=metric,
        hue="Family",
        hue_order=family_order,
        style="Family",
        style_order=family_order,
        markers=FAMILY_MARKERS,
        dashes=FAMILY_LINESTYLES,
        col="BackboneDisplay",
        col_order=backbone_order,
        col_wrap=3,
        height=4,
        aspect=1.35,
        facet_kws={"sharey": False, "sharex": True},
        palette=palette,
        linewidth=2.5,
        markersize=8,
    )
    g.set_axis_labels("Horizon", metric, fontsize=14)
    g.set_titles("{col_name}", size=16)
    g.fig.subplots_adjust(top=0.88, right=0.84)
    g.fig.suptitle(f"{metric} vs Horizon by Backbone", fontsize=20)
    if g._legend is not None:
        g._legend.set_title("Family")
        g._legend.set_bbox_to_anchor((0.86, 0.5))
        g._legend.set_loc("center left")
    png_path = os.path.join(OUTPUT_DIR, f"horizon_line_{metric}_by_backbone.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"horizon_line_{metric}_by_backbone.pdf")
    g.fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight", transparent=True)
    g.fig.savefig(pdf_path, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    plt.close(g.fig)


def plot_patchlen_by_backbone(df, metric):
    family_order = get_family_order(df)
    backbone_order = get_backbone_order(df)
    palette = get_family_palette(family_order)

    df_plot = df.groupby(["BackboneDisplay", "Family", "PatchLen"], as_index=False)[metric].mean()
    df_plot["BackboneDisplay"] = pd.Categorical(df_plot["BackboneDisplay"], categories=backbone_order, ordered=True)
    df_plot["Family"] = pd.Categorical(df_plot["Family"], categories=family_order, ordered=True)
    df_plot = df_plot.sort_values(["BackboneDisplay", "Family", "PatchLen"])

    g = sns.relplot(
        data=df_plot,
        kind="line",
        x="PatchLen",
        y=metric,
        hue="Family",
        hue_order=family_order,
        style="Family",
        style_order=family_order,
        markers=FAMILY_MARKERS,
        dashes=FAMILY_LINESTYLES,
        col="BackboneDisplay",
        col_order=backbone_order,
        col_wrap=3,
        height=4,
        aspect=1.35,
        facet_kws={"sharey": False, "sharex": True},
        palette=palette,
        linewidth=2.5,
        markersize=8,
    )
    g.set_axis_labels("Patch Length", metric, fontsize=14)
    g.set_titles("{col_name}", size=16)
    g.fig.subplots_adjust(top=0.88, right=0.84)
    g.fig.suptitle(f"{metric} vs Patch Length by Backbone", fontsize=20)
    if g._legend is not None:
        g._legend.set_title("Family")
        g._legend.set_bbox_to_anchor((0.86, 0.5))
        g._legend.set_loc("center left")
    png_path = os.path.join(OUTPUT_DIR, f"patchlen_line_{metric}_by_backbone.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"patchlen_line_{metric}_by_backbone.pdf")
    g.fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight", transparent=True)
    g.fig.savefig(pdf_path, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    plt.close(g.fig)


def plot_variant_by_backbone(df, metric):
    # Focuses on Koop/ODE variants inside each backbone; baselines stay as baseline.
    variant_order = get_variant_order(df)
    backbone_order = get_backbone_order(df)

    plt.figure(figsize=(18, 8))
    ax = sns.boxplot(
        data=df,
        x="Variant",
        y=metric,
        hue="BackboneDisplay",
        order=variant_order,
        hue_order=backbone_order,
        palette=get_backbone_palette(backbone_order),
        linewidth=1.2,
        showfliers=True,
    )
    plt.title(f"{metric} by Variant and Backbone", fontsize=20)
    plt.xlabel("Variant", fontsize=16)
    plt.ylabel(metric, fontsize=16)
    plt.xticks(rotation=30, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    make_custom_backbone_legend(ax, backbone_order)
    plt.tight_layout()
    save_current_figure(f"box_{metric}_variant_backbone")
    plt.close()


# =========================================================
# MAIN
# =========================================================
def main():
    sns.set_theme(style="whitegrid", context="talk")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = read_ragged_csv(CSV_FILE)
    print_summary(df)

    parsed_csv = os.path.join(OUTPUT_DIR, "parsed_metrics_with_backbones.csv")
    df.to_csv(parsed_csv, index=False)
    print(f"Saved parsed CSV: {parsed_csv}")

    for metric in METRICS:
        if metric not in df.columns:
            print(f"Skipping {metric}: not found in CSV")
            continue

        plot_violin_by_family(df, metric)
        plot_box_by_family(df, metric)
        plot_horizon_by_backbone(df, metric)
        plot_patchlen_by_backbone(df, metric)
        plot_variant_by_backbone(df, metric)

    print(f"\nDone. All figures saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
