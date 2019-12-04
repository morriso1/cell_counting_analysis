import pandas as pd
import numpy as np
import glob
import os
import prism_v2
import time

# get current working directory and replace spaces
Cwd = os.getcwd()
L_Cwd_folders = [items.replace(" ", "_") for items in Cwd.split("/")]

# construct save file name
if "Anterior" or "Posterior" in L_Cwd_folders:
    Exp_Analysis_Name = (
        L_Cwd_folders[-5]
        + "_"
        + L_Cwd_folders[-4]
        + "_"
        + L_Cwd_folders[-2]
        + "_"
        + L_Cwd_folders[-1]
        + "C0"
    )
else:
    Exp_Analysis_Name = (
        L_Cwd_folders[-4] + "_" + L_Cwd_folders[-3] + "_" + L_Cwd_folders[-1] + "C0"
    )

# create path to "Output_C0"
Path_C0 = os.path.join(Cwd, "Output_C0")

# check path of Output_C0 exists
if os.path.isdir(Path_C0):
    Start_Time = time.time()
    # create a list of files in Output_C0 folder
    L_CSV_C0 = sorted(glob.glob(os.path.join(Path_C0, "*.csv")))
    # create empty DataFrame to store data from csv files
    DF_C0 = pd.DataFrame()
    # create L_Col. Will be sample list without gut info. Used later to aid merging columns
    L_Col = list()

    for Files in L_CSV_C0:
        S_Temp_C0 = pd.read_csv(Files, usecols=["Mean"])
        # Label columns with shorterned version of file name
        S_Temp_C0.columns = [Files[len(Path_C0) + 1 : len(Path_C0) + 6]]
        # concat temporary DF to grow DF.
        DF_C0 = pd.concat([DF_C0, S_Temp_C0], axis=1)
        L_Col = L_Col + [Files[len(Path_C0) + 1 : len(Path_C0) + 4]]

    DF_C0_Mean = pd.DataFrame(DF_C0.mean(axis="index")).T
    DF_C0_Median = pd.DataFrame(DF_C0.median(axis="index")).T

    # rename columns without gut info.
    DF_C0_Mean.columns = L_Col
    DF_C0_Median.columns = L_Col
    DF_C0.columns = L_Col

    # only unique sample names.
    L_Unique_Col = sorted(list(set(L_Col)))

    # create empty sorted DFs
    ST_DF_C0 = pd.DataFrame()
    ST_DF_C0_Mean = pd.DataFrame()
    ST_DF_C0_Median = pd.DataFrame()

    for i, Samples in enumerate(L_Unique_Col):
        DF_C0_T = DF_C0[L_Unique_Col[i]].T
        # column index DF_C0_T as series.
        ST_DF_C0_Temp = (
            pd.DataFrame(DF_C0_T.values.flatten()).dropna().reset_index(drop=True)
        )
        ST_DF_C0 = pd.concat([ST_DF_C0, ST_DF_C0_Temp], axis=1)

        DF_C0_Mean_T = DF_C0_Mean[L_Unique_Col[i]].T
        ST_DF_C0_Mean_Temp = pd.DataFrame(DF_C0_Mean_T.values)
        ST_DF_C0_Mean = pd.concat([ST_DF_C0_Mean, ST_DF_C0_Mean_Temp], axis=1)

        DF_C0_Median_T = DF_C0_Median[L_Unique_Col[i]].T
        ST_DF_C0_Median_Temp = pd.DataFrame(DF_C0_Median_T.values)
        ST_DF_C0_Median = pd.concat([ST_DF_C0_Median, ST_DF_C0_Median_Temp], axis=1)

    ST_DF_C0.columns = L_Unique_Col  # Name columns
    ST_DF_C0_Mean.columns = L_Unique_Col
    ST_DF_C0_Median.columns = L_Unique_Col

    ## saving to excel file.
    WRITER = pd.ExcelWriter(Exp_Analysis_Name + ".xlsx")
    ST_DF_C0.to_excel(WRITER, "All_ROIs")
    ST_DF_C0_Mean.to_excel(WRITER, "Mean")
    ST_DF_C0_Median.to_excel(WRITER, "Median")
    DF_C0.to_excel(WRITER, "IndGuts_AllROIs")
    DF_C0_Mean.to_excel(WRITER, "IndGuts_AllROIs_Mean")
    DF_C0_Median.to_excel(WRITER, "IndGuts_AllROIs_Median")
    WRITER.save()

    ## saving to csv files.
    ST_DF_C0.to_csv(Exp_Analysis_Name + ".csv")
    ST_DF_C0_Mean.to_csv(Exp_Analysis_Name + "_mean.csv")
    ST_DF_C0_Median.to_csv(Exp_Analysis_Name + "_median.csv")

    ## saving to prism file.
    # index type determines which prism template to use.
    if "b1g" in L_Unique_Col:
        Index_Type = 1
    else:
        Index_Type = 0

    Prism_Output = prism_v2.df_to_pzfx_func(
        ST_DF_C0, ST_DF_C0_Mean, ST_DF_C0_Median, Index=Index_Type
    )

    Save_Dir = Cwd

    with open(os.path.join(Save_Dir, Exp_Analysis_Name) + ".pzfx", "w+") as f_out:
        for item in Prism_Output:
            f_out.write("{}".format(item))

    print(f"The prism file:\n'{Exp_Analysis_Name}' \nwas saved at\n'{Save_Dir}'")
    print(f"Completion time after user input: {(time.time() - Start_Time)} seconds.")

elif not os.path.isdir(Path_C0):
    print("ERROR=> Check Output_C0 directory - does not exist.")
