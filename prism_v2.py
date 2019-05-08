import pandas as pd


def df_to_pzfx_func(DF_1, DF_2, DF_3, Index=0):
    """Function takes three pd dataframes and combines them into a prism file
    based on template located at Template_Path"""

    Template_Path = [
        "/Users/morriso1/Documents/MacVersion_Buck + Genentech Work/Buck + Genentech Lab Work/Mito Ca2+/Experiments/Prism files/Template_Prism_Files/Asamples.pzfx",
        "/Users/morriso1/Documents/MacVersion_Buck + Genentech Work/Buck + Genentech Lab Work/Mito Ca2+/Experiments/Prism files/Template_Prism_Files/AandBsamples.pzfx",
    ]
    with open(Template_Path[Index], "r") as f:
        Content = f.readlines()
        Indices = [i for i, Elements in enumerate(Content) if "sample" in Elements]
        # find the location of every sample in the template.to_pzfx

    DF_1 = "<d>" + DF_1.astype(str) + "</d>\n"
    DF_2 = "<d>" + DF_2.astype(str) + "</d>\n"
    DF_3 = "<d>" + DF_3.astype(str) + "</d>\n"

    Content_Head = Content[: Indices[0]]
    Content_Middle = [
        "</Subcolumn>\n",
        "</YColumn>\n",
        '<YColumn Width="224" Decimals="6" Subcolumns="1">\n',
    ]
    Content_TB = [
        "</Table>\n",
        '<Table ID="Table38" XFormat="none" TableType="OneWay" EVFormat="AsteriskAfterNumber">\n',
        "<Title>TableB</Title>\n",
        '<RowTitlesColumn Width="1">\n',
        "<Subcolumn></Subcolumn>\n",
        "</RowTitlesColumn>\n",
        '<YColumn Width="211" Decimals="6" Subcolumns="1">\n',
    ]
    Content_TC = [
        "</Table>\n",
        '<Table ID="Table41" XFormat="none" TableType="OneWay" EVFormat="AsteriskAfterNumber">\n',
        "<Title>TableC</Title>\n",
        '<RowTitlesColumn Width="1">\n',
        "<Subcolumn></Subcolumn>\n",
        "</RowTitlesColumn>\n",
        '<YColumn Width="211" Decimals="6" Subcolumns="1">\n',
    ]
    Content_Tail = Content[(Indices[-1] + 3) :]

    Temp_A = []
    Temp_B = []
    Temp_C = []

    for Key, Columns in DF_1.iteritems():
        Content_Up = [f"<Title>{Key}</Title>\n", "<Subcolumn>\n"]
        Temp_A = (
            Temp_A
            + Content_Up
            + DF_1[Key][DF_1[Key] != "<d>nan</d>\n"].tolist()
            + Content_Middle
        )
        Temp_B = (
            Temp_B
            + Content_Up
            + DF_2[Key][DF_2[Key] != "<d>nan</d>\n"].tolist()
            + Content_Middle
        )
        Temp_C = (
            Temp_C
            + Content_Up
            + DF_3[Key][DF_3[Key] != "<d>nan</d>\n"].tolist()
            + Content_Middle
        )

    del Temp_A[-1]
    del Temp_B[-1]
    del Temp_C[-1]
    # required to get rid of trailing subcolumn formatting

    Output = (
        Content_Head + Temp_A + Content_TB + Temp_B + Content_TC + Temp_C + Content_Tail
    )

    return Output
