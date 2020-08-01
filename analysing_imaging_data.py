import pandas as pd
import dask.dataframe as dd
import numpy as np
import glob
import os
import re

"""Module contains functions: read_csv_folder_into_tidy_df, grouped_tidy_data_summary_stats, exp_analysis_name, 
compile_DF_from_CSVdirectory, extract_gut_names, combining_gut_DFs, analyse_imagej_CSVs, df_to_pzfx, write_pzfx,
check_if_list_of_folders_exists, summarise_and_sort_list_of_DFs and compileCSVs_sortbycondition_apply_method"""


def read_csv_folder_into_tidy_df(csv_glob, drop_columns=[' '], sample_id_categories=None):
    """
    Input
    -----
    Takes glob (str) to csv folder as input. Optional sample_id_categories (e.g. list).

    Function
    --------
    Combines into tidy dataframe.

    Returns
    -------
    Returns tidy dataframe.
    """

    df = (
        dd.read_csv(csv_glob, include_path_column="sample_gut_id")
        .compute()
        .drop(columns=drop_columns)
    )

    if sample_id_categories is None:
        df = df.assign(
            sample_gut_id=lambda x: x["sample_gut_id"].str.findall(
                "[a-z]\dg\d\d?").str[-1],
            sample_id=lambda x: pd.Categorical(
                x["sample_gut_id"].str.split("g", expand=True)[0],
            ),
            gut_id=lambda x: x["sample_gut_id"].str.split("g", expand=True)[1],
        )

    else:
        df = df.assign(
            sample_gut_id=lambda x: x["sample_gut_id"].str.findall(
                "[a-z]\dg\d\d?").str[-1],
            sample_id=lambda x: pd.Categorical(
                x["sample_gut_id"].str.split("g", expand=True)[
                    0], categories=sample_id_categories
            ),
            gut_id=lambda x: x["sample_gut_id"].str.split("g", expand=True)[1],
        )

    return df


def grouped_tidy_data_summary_stats(
    tidy_df, group_col="image_key", categories=None, agg_funcs=['mean'], **agg_kwargs
):
    """
    Input
    -----
    Takes tidy DataFrame, group_col (str), categories (e.g. list) and aggregation functions.

    Returns
    -------
    Tidy DataFrame with 'summary_stats' performed on selected group.
    """

    tidy_df_grouped = (
        tidy_df.groupby(by=group_col)
        .agg(agg_funcs, **agg_kwargs, axis="column")
        .stack()
        .reset_index()
        .rename(columns={"level_1": "summary_stat"})
    )

    if categories is not None:
        tidy_df_grouped[["sample_id", "gut_id"]] = tidy_df_grouped[group_col].str.split(
            "g", expand=True
        )
        tidy_df_grouped["sample_id"] = pd.Categorical(
            tidy_df_grouped["sample_id"], categories=categories
        )

    return(tidy_df_grouped)


def exp_analysis_name(Exp_Folder=os.getcwd()):

    Exp_Folder_List = [items.replace(" ", "_")
                       for items in Exp_Folder.split("/")]
    if len(Exp_Folder_List) < 4:
        ExpAnalysisName = 'test'
        return ExpAnalysisName
    elif "Anterior" or "Posterior" in Exp_Folder_List:
        ExpAnalysisName = (
            Exp_Folder_List[-5]
            + "_"
            + Exp_Folder_List[-4]
            + "_"
            + Exp_Folder_List[-2]
            + "_"
            + Exp_Folder_List[-1]
        )
    else:
        ExpAnalysisName = (
            Exp_Folder_List[-4] + "_" + Exp_Folder_List[-3] +
            "_" + Exp_Folder_List[-1]
        )

    return ExpAnalysisName


def compile_DF_from_CSVdirectory(Path_Dir, usecolumns=["Mean"]):
    """
    Input: Function takes
    """
    CSV_FullPath_L = sorted(glob.glob(os.path.join(Path_Dir, "*.csv")))
    Compiled_DF = pd.DataFrame()
    for Files in CSV_FullPath_L:
        Temp_DF = pd.read_csv(Files, usecols=usecolumns)
        Temp_DF.columns = [os.path.basename(Files)]
        Compiled_DF = pd.concat([Compiled_DF, Temp_DF], axis=1)
    return Compiled_DF


def extract_gut_names(DF):
    Columns_DF = DF.columns.to_list()
    RE_Match_L = list()
    for Names in Columns_DF:
        RE_Match = re.search(r"\w\dg\d+", Names)
        RE_Match_L.append(RE_Match.group(0))
    return RE_Match_L


def combining_gut_DFs(DF):
    DF.columns = DF.columns.str[0:3]
    UniqueConditions = sorted(list(set(DF.columns.tolist())))

    Sorted_DF = pd.DataFrame()
    for Condition in UniqueConditions:
        DF_Temp = DF[Condition].T
        Sorted_DF_Temp = (
            pd.DataFrame(DF_Temp.values.flatten()
                         ).dropna().reset_index(drop=True)
        )
        Sorted_DF = pd.concat([Sorted_DF, Sorted_DF_Temp], axis=1)

    Sorted_DF.columns = UniqueConditions

    return Sorted_DF


def analyse_imagej_CSVs(
    Exp_Folder=os.getcwd(), Num_Dir="Output_C0", Denom_Dir=None, usecolumns=["Mean"]
):
    Path_Num_Dir = os.path.join(Exp_Folder, Num_Dir)

    if os.path.isdir(Path_Num_Dir):
        Num_DF = compile_DF_from_CSVdirectory(
            Path_Num_Dir, usecolumns=usecolumns)

    else:
        print(
            f"Num_Dir input '{Num_Dir}' is not a directory located in:\n{Exp_Folder}")
        return

    if Denom_Dir == None:
        Denom_DF = 1
        pass
    else:
        Path_Denom_Dir = os.path.join(Exp_Folder, Denom_Dir)
        if os.path.isdir(Path_Denom_Dir):
            Denom_DF = compile_DF_from_CSVdirectory(
                Path_Denom_Dir, usecolumns=usecolumns
            )
        else:
            print(
                f"Denom_Dir input '{Denom_Dir}' is not a directory located in:\n{Exp_Folder}"
            )
            return

    if Denom_Dir == None:
        Gut_Names_Num = extract_gut_names(Num_DF)
        Num_DF.columns = Gut_Names_Num

    else:
        Gut_Names_Num = extract_gut_names(Num_DF)
        Gut_Names_Denom = extract_gut_names(Denom_DF)

        if Gut_Names_Denom == Gut_Names_Num:
            Num_DF.columns = Gut_Names_Num
            Denom_DF.columns = Gut_Names_Denom
        else:
            print(f"{Num_Dir} and {Denom_Dir} do not contain matched CSVs")

    Divided_DF = Num_DF.div(Denom_DF)
    Sorted_Div_DF = combining_gut_DFs(Divided_DF)
    Sorted_Div_DF_Mean = combining_gut_DFs(pd.DataFrame(Divided_DF.mean()).T)
    Sorted_Div_DF_Median = combining_gut_DFs(
        pd.DataFrame(Divided_DF.median()).T)

    ExpAnalysisName = exp_analysis_name(Exp_Folder)
    return (
        ExpAnalysisName,
        Sorted_Div_DF,
        Sorted_Div_DF_Mean,
        Sorted_Div_DF_Median,
        Divided_DF,
    )


def df_to_pzfx(DF_1, DF_2, DF_3, Index=0):
    """Function takes three pd dataframes and combines them into a prism file
    based on template located at Template_Path"""

    Template_Path = [
        "/Users/morriso1/Documents/MacVersion_Buck + Genentech Work/Buck + Genentech Lab Work/Mito Ca2+/Experiments/Prism files/Template_Prism_Files/Asamples.pzfx",
        "/Users/morriso1/Documents/MacVersion_Buck + Genentech Work/Buck + Genentech Lab Work/Mito Ca2+/Experiments/Prism files/Template_Prism_Files/AandBsamples.pzfx",
    ]
    with open(Template_Path[Index], "r") as f:
        Content = f.readlines()
        Indices = [i for i, Elements in enumerate(
            Content) if "sample" in Elements]
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
    Content_Tail = Content[(Indices[-1] + 3):]

    Temp_A = []
    Temp_B = []
    Temp_C = []

    for Key, _ in DF_1.iteritems():
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

    Prism_Output = (
        Content_Head + Temp_A + Content_TB + Temp_B + Content_TC + Temp_C + Content_Tail
    )

    return Prism_Output


def write_pzfx(Prism_Output, Save_Dir=os.getcwd(), ExpAnalysisName=None):
    if ExpAnalysisName == None:
        print("Please provide ExpAnalysisName")
        return

    with open(os.path.join(Save_Dir, ExpAnalysisName) + ".pzfx", "w+") as f_out:
        for item in Prism_Output:
            f_out.write("{}".format(item))

    print(f"The prism file:\n'{ExpAnalysisName}' \nwas saved at\n'{Save_Dir}'")


def check_if_list_of_folders_exists(
    Exp_Folder=os.getcwd(), FoldersToCount=["Output_C0", "Output_C2"]
):
    """Input: Takes path to experiment folder and list of folders.

    Function: Checks if list of folders are in the experiment folder.

    Output: Returns boolean if they exist (first output) and list of full paths of input folders (second output)"""

    Path_Num_Dir_L = list()
    for Folder in FoldersToCount:
        Path_Num_Dir = os.path.join(Exp_Folder, Folder)
        if os.path.isdir(Path_Num_Dir):
            Path_Num_Dir_L.append(Path_Num_Dir)
            continue
        else:
            print(f"{Path_Num_Dir} is not a path to a directory.")
            return False
    return True, Path_Num_Dir_L


def summarise_and_sort_list_of_DFs(
    L_DFs, Method="count", Folders=["Output_C0", "Output_C2"]
):
    """Input: Takes as input list of pd.DataFrames, method to summarise to them (e.g. count, median or median) and 
    folders from which list of pd.DataFrames was constructed.

    Function: Checks if lists of pd.DataFrames and folders are same length, sorts and concats pd.DataFrames by unique
    condition and places them into single dictionary.

    Output: Returns sorted pd.DataFrames in dictionary with folder name as key."""

    if type(L_DFs) is list and type(Folders) is list and len(L_DFs) == len(Folders):

        Sorted_DFs_L = []
        for DFs in L_DFs:
            DFs.columns = DFs.columns.str.extract("([a-z][0-9]g[0-9][0-9]?)")[
                0
            ].tolist()
            if Method == "count":
                DFs_method = DFs.count()
            elif Method == "mean":
                DFs_method = DFs.mean()
            elif Method == "median":
                DFs_method = DFs.median()
            else:
                print("Available methods are 'count', 'mean' and 'median'.")
                return

            Split_Indexes = DFs_method.index.str.split("g")
            Split_Indexes = [x[0] for x in Split_Indexes]
            Unique_Conditions = sorted(set(Split_Indexes))

            DF_sorted = pd.DataFrame()

            # sorts by number of conditions e.g. 'a1' in unique conditions
            for Condition in Unique_Conditions:
                DF_temp = pd.DataFrame(
                    DFs_method[DFs_method.index.str.match(Condition)]
                )
                DF_temp.index = DF_temp.index.str.replace(Condition, "")
                DF_temp.index.name = "Guts"
                DF_temp.columns = [Condition]
                DF_sorted = pd.concat([DF_sorted, DF_temp], axis=1, sort=False)
                DF_sorted = DF_sorted.astype("float")

            Sorted_DFs_L.append(DF_sorted)

        return dict(zip(Folders, Sorted_DFs_L))

    else:
        print(f"L_DFs and {Folders} are not lists of the same length")


def compileCSVs_sortbycondition_apply_method(
    Exp_Folder=os.getcwd(),
    FoldersToApplyMethodTo=["Output_C0", "Output_C2"],
    Method="count",
):
    """Input: Takes path to experiment folder, list of folders to apply a summarise method and type of
    summarise method to apply (e.g. count, median or median).

    Function: Checks if list of folders exists, reads in and compiles CSVs by folder as list of pd.DataFrames,
    summarises and sorts list of pd.DataFrames.

    Returns: Returns sorted pd.DataFrames in dictionary with folder name as key.
    """

    if check_if_list_of_folders_exists(Exp_Folder, FoldersToApplyMethodTo):
        _, Path_Num_Dir_L = check_if_list_of_folders_exists(
            Exp_Folder, FoldersToApplyMethodTo
        )
        L_DFs = list()
        for Folder in Path_Num_Dir_L:
            DF_Temp = compile_DF_from_CSVdirectory(Folder)
            L_DFs.append(DF_Temp)

        Dict_of_DFs = summarise_and_sort_list_of_DFs(
            L_DFs=L_DFs, Folders=FoldersToApplyMethodTo, Method=Method
        )
        return Dict_of_DFs

    else:
        return
