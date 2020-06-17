import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set(style="ticks")

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def define_pallete(
    DF,
    ay_pattern="[a|y|m]..?.?",
    ay_color="#9FADAD",
    old_pattern="o..?.?",
    old_color="#F57171",
    treated_pattern="[b|c|d|e|f]..?.?",
    treated_color="#539DC2",
    Other_treatment_pattern="[r|s]..?.?",
    Other_treatment_pattern_color="#009900",
    m_treatment_pattern="m..?.?",
    m_treatment_color="#BA55D3",
):
    Color_List = (
        DF.columns.str.replace(ay_pattern, ay_color)
        .str.replace(old_pattern, old_color)
        .str.replace(treated_pattern, treated_color)
        .str.replace(Other_treatment_pattern, Other_treatment_pattern_color)
        .str.replace(m_treatment_pattern, m_treatment_color)
        .tolist()
    )
    Cols = DF.columns.tolist()
    ColsColor_Dict = dict(zip(Cols, Color_List))

    return ColsColor_Dict


def create_strip_or_swarm_boxplot(
    DF,
    ExpName="Test",
    x_figSize=3.75,
    y_figSize=2.5,
    ScaleSize=1,
    y_label="not set",
    y_axis_limit=None,
    y_axis_start=0,
    UpperAxisLimit=None,
):
    fig, ax = plt.subplots()

    fig.set_size_inches(x_figSize * ScaleSize, y_figSize * ScaleSize)

    # ax = plt.gcf()

    ColsColor_Dict = define_pallete(DF)

    if DF.shape[0] < 30:
        ax = sns.swarmplot(
            data=DF,
            palette=ColsColor_Dict,
            alpha=0.8,
            zorder=1,
            edgecolor="gray",
            linewidth=0.5,
            size=5,
        )
        ax = sns.boxplot(
            data=DF,
            palette=ColsColor_Dict,
            fliersize=0,
            zorder=0,
            saturation=0.9,
            linewidth=1.5,
            notch=False,
        )
        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 0.5))

    else:
        if UpperAxisLimit == None:
            UpperAxisLimit = (
                DF.quantile(0.75) + (DF.quantile(0.75) -
                                     DF.quantile(0.25)) * 1.7
            ).max()

        ax = sns.stripplot(
            data=DF[DF < UpperAxisLimit],
            palette=ColsColor_Dict,
            alpha=0.2,
            zorder=0,
            jitter=0.3,
            edgecolor="gray",
            linewidth=0.5,
        )
        ax = sns.boxplot(
            data=DF,
            palette=ColsColor_Dict,
            fliersize=0,
            zorder=1,
            saturation=0.9,
            linewidth=1.5,
            notch=True,
        )
        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 0.8))

        if y_axis_limit == None:
            ax.set_ylim([0, UpperAxisLimit])

    if y_axis_limit != None:
        ax.set_ylim(top=y_axis_limit)

    ax.set_ylim(bottom=y_axis_start)
    ax.set_ylabel(y_label, fontsize=12)
    ax.xaxis.set_tick_params(width=1)
    ax.yaxis.set_tick_params(width=1)
    ax.tick_params(axis="both", which="major", pad=1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.setp(ax.spines.values(), linewidth=1)
    sns.despine()

    plt.savefig(f"{ExpName}.pdf", transparent=True, bbox_inches="tight")


def identify_y_axis_label(ExpAnalysisName):
    Name = pd.Series(ExpAnalysisName)
    y_axis_label = "y_axis_label"

    if Name.str.contains(r"cepia", case=False)[0]:
        if Name.str.contains(r"MitoTd", case=False)[0]:
            y_axis_label = "CEPIA3mt/MitoTdTomato\n(Relative Mito[Ca2+])"
        else:
            y_axis_label = "CEPIA3mt Intensity\n(Relative Mito[Ca2+]"

    if Name.str.contains(r"Perceval", case=False)[0]:
        y_axis_label = "PercevalHR 488/405\n(Relative ATP/ADP)"

    if Name.str.contains(r"SoNAR", case=False)[0]:
        y_axis_label = "SoNaR 405/488\n(Relative NADH/NAD+)"

    if Name.str.contains(r"cpYFP", case=False)[0]:
        y_axis_label = "CpYFP 405/488"

    if Name.str.contains(r"Mito.*orp1", case=False)[0]:
        y_axis_label = "MitoRoGFP2_Orp1 405/488\n(Relative Mito[H2O2])"

    if Name.str.contains(r"Mito.*Grx", case=False)[0]:
        y_axis_label = "MitoRoGFP2_Grx1 405/488\n(Relative Mito[EGSH])"

    if Name.str.contains(r"Cyto.*Grx", case=False)[0]:
        y_axis_label = "CytoRoGFP2_Grx1 405/488\n(Relative Cyto[EGSH])"

    if Name.str.contains(r"MitoNAD", case=False)[0]:
        y_axis_label = "MitoNAD+ sensor 405/488\n(Relative Mito[NAD+])"

    if Name.str.contains(r"Lacon", case=False)[0]:
        y_axis_label = "Laconic Em 480/550\n(Relative [Lactate])"

    return y_axis_label


def determine_fig_width(DF):
    """Input: Takes DF

    Function: Determine number of columns. If less than 3 cols, Fig_Width is 1.5. 
    If 4-5 cols, Fig_Width is 2.5. If more than 5 cols, Fig_Width is 3.75. 

    Output: Return Fig_Width"""

    if DF.shape[1] <= 3:
        Fig_Width = 1.5

    elif 3 < DF.shape[1] < 5:
        Fig_Width = 2.5

    else:
        Fig_Width = 3.75

    return Fig_Width


def define_pallette_tidy(
    series_id_category,
    color_mappings=dict(
        {
            "[a|y|m]..?.?": "#9FADAD",
            "o..?.?": "#F57171",
            "[b|c|d|e|f]..?.?": "#539DC2",
            "[r|s]..?.?": "#009900",
            "m..?.?": "#BA55D3",
        }
    ),
):

    series_id = pd.Series(series_id_category.cat.categories)

    for mapping, color in color_mappings.items():
        series_id = series_id.str.replace(mapping, color)

    return series_id.to_list()


def tidy_create_strip_box_plot(
    x_figSize=2.5,
    y_figSize=2.5,
    ScaleSize=1,
    y_axis_start=0,
    y_axis_limit=None,
    y_label="set y label",
    save_fig=False,
    ExpName="Test_plot",
    **kwargs,
):
    """Creates sns plots. Pass **kwargs to sns.stripplot and sns.boxplot."""

    fig, ax = plt.subplots()
    fig.set_size_inches(x_figSize * ScaleSize, y_figSize * ScaleSize)

    ax = sns.stripplot(
        alpha=0.2, zorder=0, jitter=0.3, edgecolor="gray", linewidth=0.5, **kwargs
    )
    ax = sns.boxplot(
        fliersize=0, zorder=1, saturation=0.9, linewidth=1.5, notch=True, **kwargs
    )

    if y_axis_limit != None:
        ax.set_ylim(top=y_axis_limit)

    ax.set_ylim(bottom=y_axis_start)
    ax.set_ylabel(y_label, fontsize=12)
    ax.xaxis.set_tick_params(width=1)
    ax.yaxis.set_tick_params(width=1)
    ax.tick_params(axis="both", which="major", pad=1)
    ax.xaxis.set_label_text("")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.setp(ax.spines.values(), linewidth=1)
    sns.despine()

    if save_fig:
        plt.savefig(f"{ExpName}.pdf", transparent=True, bbox_inches="tight")


def tidy_create_swarm_box_plot(
    x_figSize=2.5,
    y_figSize=2.5,
    ScaleSize=1,
    y_axis_start=0,
    y_axis_limit=None,
    y_label="set y label",
    save_fig=False,
    ExpName="Test_plot",
    **kwargs,
):
    """Creates sns plots. Pass **kwargs to sns.swarmplot and sns.boxplot."""

    fig, ax = plt.subplots()
    fig.set_size_inches(x_figSize * ScaleSize, y_figSize * ScaleSize)

    ax = sns.swarmplot(
        alpha=0.8, zorder=1, edgecolor="gray", linewidth=0.5, size=5, **kwargs
    )

    ax = sns.boxplot(
        fliersize=0, zorder=0, saturation=0.9, linewidth=1.5, notch=False, **kwargs
    )

    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.5))

    if y_axis_limit != None:
        ax.set_ylim(top=y_axis_limit)

    ax.set_ylim(bottom=y_axis_start)
    ax.set_ylabel(y_label, fontsize=12)
    ax.xaxis.set_tick_params(width=1)
    ax.yaxis.set_tick_params(width=1)
    ax.tick_params(axis="both", which="major", pad=1)
    ax.xaxis.set_label_text("")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.setp(ax.spines.values(), linewidth=1)
    sns.despine()

    if save_fig:
        plt.savefig(f"{ExpName}.pdf", transparent=True, bbox_inches="tight")


def determine_fig_width_from_palette(color_pal):
    """Input: Takes palette list to be used in plotting.

    Function: Determine number of items in list. If less than 3 list, Fig_Width is 1.5. 
    If 4-5 cols, Fig_Width is 2.5. If more than 5 list, Fig_Width is 3.75. 

    Output: Return Fig_Width"""

    if len(color_pal) <= 3:
        Fig_Width = 1.5

    elif 3 < len(color_pal) < 5:
        Fig_Width = 2.5

    else:
        Fig_Width = 3.75

    return Fig_Width
