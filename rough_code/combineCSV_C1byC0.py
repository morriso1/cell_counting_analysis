import pandas as pd
import numpy as np
import glob
import os
import prism_v2
import time

## Not yet checked.
Cwd = os.getcwd()
L_Cwd_folders = [items.replace(" ", "_") for items in Cwd.split("/")]

if "Anterior" or "Posterior" in L_Cwd_folders:
    Exp_Analysis_Name = (
        L_Cwd_folders[-5] + "_" + L_Cwd_folders[-4] + "_" + L_Cwd_folders[-2] + "_" + L_Cwd_folders[-1]
    )
else:
    Exp_Analysis_Name = L_Cwd_folders[-4] + "_" + L_Cwd_folders[-3] + "_" + L_Cwd_folders[-1]

Path_405 = os.path.join(Cwd, "Output_C0")
Path_488 = os.path.join(Cwd, "Output_C1")

# print(Exp_Analysis_Name, Path_405, Path_488, sep="\n")

if os.path.isdir(Path_405) and os.path.isdir(Path_488):
    # if statement checks to make sure that paths are ok.
    Start_Time = time.time()
    L_CSV_405 = sorted(glob.glob(os.path.join(Path_405, "*.csv")))
    L_CSV_488 = sorted(glob.glob(os.path.join(Path_488, "*.csv")))
    DF_405 = pd.DataFrame()
    DF_488 = pd.DataFrame()
    L_Col = list()

    for Files_1 in L_CSV_405:
        S_Temp_405 = pd.read_csv(Files_1, usecols=["Mean"])
        S_Temp_405.columns = [Files_1[len(Path_405) + 1 : len(Path_405) + 6]]
        DF_405 = pd.concat([DF_405, S_Temp_405], axis=1)
        L_Col = L_Col + [Files_1[len(Path_405) + 1 : len(Path_405) + 4]]
        # L_Col is sample list without gut info. Used later to aid merging columns

    for Files_2 in L_CSV_488:
        S_Temp_488 = pd.read_csv(Files_2, usecols=["Mean"])
        S_Temp_488.columns = [Files_2[len(Path_488) + 1 : len(Path_488) + 6]]
        DF_488 = pd.concat([DF_488, S_Temp_488], axis=1)

    DF_488_405 = DF_488.div(DF_405)
    DF_488_405_Mean = pd.DataFrame(DF_488_405.mean(axis="index")).T
    DF_488_405_Mean.columns = L_Col
    DF_488_405_Median = pd.DataFrame(DF_488_405.median(axis="index")).T
    DF_488_405_Median.columns = L_Col

    DF_488_405.columns = L_Col  # rename columns without gut info.
    L_Unique_Col = sorted(list(set(L_Col)))  # only unique sample names.

    ST_DF_488_405 = pd.DataFrame()
    ST_DF_488_405_Mean = pd.DataFrame()
    ST_DF_488_405_Median = pd.DataFrame()

    for i, Samples in enumerate(L_Unique_Col):
        DF_488_405_T = DF_488_405[L_Unique_Col[i]].T
        ST_DF_488_405_Temp = (
            pd.DataFrame(DF_488_405_T.values.flatten()).dropna().reset_index(drop=True)
        )  # column index DF_488_405_T as series.
        ST_DF_488_405 = pd.concat([ST_DF_488_405, ST_DF_488_405_Temp], axis=1)

        DF_488_405_Mean_T = DF_488_405_Mean[L_Unique_Col[i]].T
        ST_DF_488_405_Mean_Temp = pd.DataFrame(DF_488_405_Mean_T.values)
        ST_DF_488_405_Mean = pd.concat(
            [ST_DF_488_405_Mean, ST_DF_488_405_Mean_Temp], axis=1
        )

        DF_488_405_Median_T = DF_488_405_Median[L_Unique_Col[i]].T
        ST_DF_488_405_Median_Temp = pd.DataFrame(DF_488_405_Median_T.values)
        ST_DF_488_405_Median = pd.concat(
            [ST_DF_488_405_Median, ST_DF_488_405_Median_Temp], axis=1
        )

    ST_DF_488_405.columns = L_Unique_Col  # Name columns
    ST_DF_488_405_Mean.columns = L_Unique_Col
    ST_DF_488_405_Median.columns = L_Unique_Col

    ## saving to excel file.
    WRITER = pd.ExcelWriter(Exp_Analysis_Name + ".xlsx")
    ST_DF_488_405.to_excel(WRITER, "All_ROIs")
    ST_DF_488_405_Mean.to_excel(WRITER, "Mean")
    ST_DF_488_405_Median.to_excel(WRITER, "Median")
    DF_488_405.to_excel(WRITER, "IndGuts_AllROIs")
    DF_488_405_Mean.to_excel(WRITER, "IndGuts_AllROIs_Mean")
    DF_488_405_Median.to_excel(WRITER, "IndGuts_AllROIs_Median")
    WRITER.save()

    ## saving to csv files.
    ST_DF_488_405.to_csv(Exp_Analysis_Name + ".csv")
    ST_DF_488_405_Mean.to_csv(Exp_Analysis_Name + "_mean.csv")
    ST_DF_488_405_Median.to_csv(Exp_Analysis_Name + "_median.csv")

    ## saving to prism file.
    # index type determines which prism template to use.
    if "b1g" in L_Unique_Col:
        Index_Type = 1
    else:
        Index_Type = 0

    Prism_Output = prism_v2.df_to_pzfx_func(
        ST_DF_488_405, ST_DF_488_405_Mean, ST_DF_488_405_Median, Index=Index_Type
    )

    Save_Dir = Cwd

    with open(os.path.join(Save_Dir, Exp_Analysis_Name) + ".pzfx", "w+") as f_out:
        for item in Prism_Output:
            f_out.write("{}".format(item))

    print(f"The prism file:\n'{Exp_Analysis_Name}' \nwas saved at\n'{Save_Dir}'")
    print(f"Completion time after user input: {(time.time() - Start_Time)} seconds.")

    # print(ST_DF_488_405)
    # print(ST_DF_488_405_Mean)
    # print(ST_DF_488_405_Median)


elif os.path.isdir(Path_488) and not os.path.isdir(Path_405):
    print("ERROR=> Check Output_C0 directory - does not exist.")

elif os.path.isdir(Path_405) and not os.path.isdir(Path_488):
    print("ERROR=> Check Output_C1 directory - does not exist.")
