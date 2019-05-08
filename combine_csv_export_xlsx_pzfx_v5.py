import pandas as pd
import numpy as np
import glob
import os
import re
import prism_v2
import time

## Not yet checked.
Cwd = os.getcwd()
L_Cwd_folders = [items.replace(" ", "_") for items in Cwd.split("/")]

if "Anterior" in L_Cwd_folders:
    Exp_Analysis_Name = (
        L_Cwd_folders[-4] + "_" + L_Cwd_folders[-2] + "_" + L_Cwd_folders[-1]
    )
    Exp_Dir_Name = L_Cwd_folders[-4]
else:
    Exp_Analysis_Name = L_Cwd_folders[-3] + "_" + L_Cwd_folders[-1]
    Exp_Dir_Name = L_Cwd_folders[-3]

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

    DF_405_488 = DF_405.div(DF_488)
    DF_405_488_Mean = pd.DataFrame(DF_405_488.mean(axis="index")).T
    DF_405_488_Mean.columns = L_Col
    DF_405_488_Median = pd.DataFrame(DF_405_488.median(axis="index")).T
    DF_405_488_Median.columns = L_Col

    DF_405_488.columns = L_Col  # rename columns without gut info.
    L_Unique_Col = sorted(list(set(L_Col)))  # only unique sample names.

    ST_DF_405_488 = pd.DataFrame()
    ST_DF_405_488_Mean = pd.DataFrame()
    ST_DF_405_488_Median = pd.DataFrame()

    for i, Samples in enumerate(L_Unique_Col):
        DF_405_488_T = DF_405_488[L_Unique_Col[i]].T
        ST_DF_405_488_Temp = (
            pd.DataFrame(DF_405_488_T.values.flatten()).dropna().reset_index(drop=True)
        )  # column index DF_405_488_T as series.
        ST_DF_405_488 = pd.concat([ST_DF_405_488, ST_DF_405_488_Temp], axis=1)

        DF_405_488_Mean_T = DF_405_488_Mean[L_Unique_Col[i]].T
        ST_DF_405_488_Mean_Temp = pd.DataFrame(DF_405_488_Mean_T.values)
        ST_DF_405_488_Mean = pd.concat(
            [ST_DF_405_488_Mean, ST_DF_405_488_Mean_Temp], axis=1
        )

        DF_405_488_Median_T = DF_405_488_Median[L_Unique_Col[i]].T
        ST_DF_405_488_Median_Temp = pd.DataFrame(DF_405_488_Median_T.values)
        ST_DF_405_488_Median = pd.concat(
            [ST_DF_405_488_Median, ST_DF_405_488_Median_Temp], axis=1
        )

    ST_DF_405_488.columns = L_Unique_Col  # Name columns
    ST_DF_405_488_Mean.columns = L_Unique_Col
    ST_DF_405_488_Median.columns = L_Unique_Col

    ## saving to excel file.
    WRITER = pd.ExcelWriter(Exp_Analysis_Name + ".xlsx")
    ST_DF_405_488.to_excel(WRITER, "All_ROIs")
    ST_DF_405_488_Mean.to_excel(WRITER, "Mean")
    ST_DF_405_488_Median.to_excel(WRITER, "Median")
    WRITER.save()

    ## saving to pickle files.
    ST_DF_405_488.to_pickle(Exp_Analysis_Name + ".pkl")
    ST_DF_405_488_Mean.to_pickle(Exp_Analysis_Name + "_mean.pkl")
    ST_DF_405_488_Median.to_pickle(Exp_Analysis_Name + "_median.pkl")

    ## saving to prism file.
    # index type determines which prism template to use.
    if "b1g" in L_Unique_Col:
        Index_Type = 1
    else:
        Index_Type = 0

    Prism_Output = prism_v2.df_to_pzfx_func(
        ST_DF_405_488, ST_DF_405_488_Mean, ST_DF_405_488_Median, Index=Index_Type
    )

    Prism_Path = "/Users/morriso1/Documents/MacVersion_Buck + Genentech Work/Buck + Genentech Lab Work/Mito Ca2+/Experiments/Prism files"

    # L_Prism_Lower = [
    #     Name.lower()
    #     for Name in os.listdir(Prism_Path)
    #     if os.path.isdir(os.path.join(Prism_Path, Name))
    # ]

    if re.match("(?i).*RoGFP.*", Exp_Dir_Name):
        Temp_Dir = os.path.join(Prism_Path, "RoGFP2 experiments")
        if Exp_Dir_Name in os.listdir(Temp_Dir):
            Save_Dir = os.path.join(Temp_Dir, Exp_Dir_Name)
        else:
            os.mkdir(os.path.join(Temp_Dir, Exp_Dir_Name))
            Save_Dir = os.path.join(Temp_Dir, Exp_Dir_Name)

    if re.match("(?i).*Cepia.*|.*MitoGCaMP3.*", Exp_Dir_Name):
        Temp_Dir = os.path.join(Prism_Path, "Mito Ca2+ experiments")
        if Exp_Dir_Name in os.listdir(Temp_Dir):
            Save_Dir = os.path.join(Temp_Dir, Exp_Dir_Name)
        else:
            os.mkdir(os.path.join(Temp_Dir, Exp_Dir_Name))
            Save_Dir = os.path.join(Temp_Dir, Exp_Dir_Name)

    elif re.match("(?i).*SoNAR.*|.*cpYFP.*", Exp_Dir_Name):
        Temp_Dir = os.path.join(Prism_Path, "SoNAR + cpYFP experiments")
        if Exp_Dir_Name in os.listdir(Temp_Dir):
            Save_Dir = os.path.join(Temp_Dir, Exp_Dir_Name)
        else:
            os.mkdir(os.path.join(Temp_Dir, Exp_Dir_Name))
            Save_Dir = os.path.join(Temp_Dir, Exp_Dir_Name)
    else:
        Save_Dir = Cwd

    with open(os.path.join(Save_Dir, Exp_Analysis_Name) + ".pzfx", "w+") as f_out:
        for item in Prism_Output:
            f_out.write("{}".format(item))

    print(f"The prism file:\n'{Exp_Analysis_Name}' \nwas saved at\n'{Save_Dir}'")
    print(f"Completion time after user input: {(time.time() - Start_Time)} seconds.")

    # print(ST_DF_405_488)
    # print(ST_DF_405_488_Mean)
    # print(ST_DF_405_488_Median)


elif os.path.isdir(Path_488) and not os.path.isdir(Path_405):
    print("ERROR=> Check Output_C0 directory - does not exist.")

elif os.path.isdir(Path_405) and not os.path.isdir(Path_488):
    print("ERROR=> Check Output_C1 directory - does not exist.")
