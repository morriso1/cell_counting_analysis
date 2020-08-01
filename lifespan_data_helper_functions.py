import pandas as pd
import numpy as np


def excel_sheets_2_dict(File_Name):
    xl = pd.ExcelFile(File_Name)

    Dict_Excel_Sheets = {
        sheets: xl.parse(sheets, nrows=3, header=None, index_col=0).T.set_index("Days")
        for sheets in xl.sheet_names
    }
    return Dict_Excel_Sheets


def convert_to_lifespan_readable(
    DF, Col_Head="Unnamed", Col_Num_Dead="Total Deaths", Col_Num_Cens="Total Escape"
):
    """
    Input: Takes DataFrame and string Col_Head
    Functon: Converts counts dead and censors into binary format for survival curve (e.g. 11100)
    Output: DataFrame with string Col_Head as column title.
    """

    if Col_Num_Dead and Col_Num_Cens not in DF.columns:
        return print(f"'{Col_Num_Dead}' and '{Col_Num_Cens}' are not columns in DF")

    Binary_Num_Dead_Cens = np.empty(0)
    BinaryNoDays = np.empty(0)

    for index in DF.index:
        Number_Dead = DF.loc[index, Col_Num_Dead]
        Number_Cens = DF.loc[index, Col_Num_Cens]
        Number_Dead_Cens = np.append(np.ones(Number_Dead), np.zeros(Number_Cens))
        Binary_Num_Dead_Cens = np.append(Binary_Num_Dead_Cens, Number_Dead_Cens)
        BinaryNoDays = np.append(
            BinaryNoDays, np.repeat(index, Number_Dead + Number_Cens)
        )

    DF_Life = pd.DataFrame(data=Binary_Num_Dead_Cens, index=BinaryNoDays, dtype="int32")
    DF_Life.columns = [Col_Head]
    return DF_Life


def dict_concat_2_LifespanDF(Dict_Excel_Sheets):
    """
    Input: Takes Dictionary of excel sheet dataframes.
    Functon: 
    Output: 
    """

    DF = pd.DataFrame()

    for keys in Dict_Excel_Sheets:
        Temp_DF = convert_to_lifespan_readable(
            Dict_Excel_Sheets.get(keys), Col_Head=keys
        )
        DF = pd.concat([DF, Temp_DF], sort=True)

    return DF
