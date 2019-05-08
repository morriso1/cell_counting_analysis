import pandas as pd
import numpy as np
import glob
import os
import time

Save_Path = os.path.join(input("Enter save directory path - must contain\
 Output_C1\n: "))
Exp_Name = os.path.join(input("Name of the experiment (no spaces): "))
Channel_Path = os.path.join(Save_Path, "Output_C1")

# Save_Path = "/Users/morriso1/Documents/Learning Python/Code/rough_code/Save_Path_Test"
# Exp_Name = "Test"
# Channel_Path = "/Users/morriso1/Documents/Learning Python/Code/rough_code/Output_C1"

if os.path.isdir(Save_Path) and os.path.isdir(Channel_Path):
    START = time.time()
    L_Channel_Path = sorted(glob.glob(os.path.join(Channel_Path, '*.csv')))
    DF_Channel = pd.DataFrame()
    L_Col = list()

    for Files in L_Channel_Path:
        S_Temp_Channel = pd.read_csv(Files, usecols = ['Mean'])
        S_Temp_Channel.columns = [Files[len(Channel_Path)+1 : len(Channel_Path)+8]]
        DF_Channel = pd.concat([DF_Channel, S_Temp_Channel], axis=1)
        L_Col = L_Col + [Files[len(Channel_Path)+1 : len(Channel_Path)+4]]

    DF_Channel_Mean = pd.DataFrame(DF_Channel.mean(axis='index')).T
    DF_Channel_Median = pd.DataFrame(DF_Channel.median(axis='index')).T
    DF_Channel.columns = L_Col
    DF_Channel_Mean.columns = L_Col
    DF_Channel_Median.columns = L_Col
    L_Col_unique = sorted(list(set(L_Col)))

    DF_Combined = pd.DataFrame()
    DF_Combined_Mean = pd.DataFrame()
    DF_Combined_Median = pd.DataFrame()

    for Samples in L_Col_unique:
        DF_Combined_Temp = pd.DataFrame(DF_Channel[Samples].T.values.flatten()\
        , columns=[Samples]).dropna().reset_index(drop=True)
        DF_Combined = pd.concat([DF_Combined, DF_Combined_Temp], axis=1)

        DF_Combined_Mean_Temp = pd.DataFrame(DF_Channel_Mean[Samples].T.values\
        .flatten(), columns=[Samples]).dropna().reset_index(drop=True)
        DF_Combined_Mean = pd.concat([DF_Combined_Mean, DF_Combined_Mean_Temp], axis=1)

        DF_Combined_Median_Temp = pd.DataFrame(DF_Channel_Median[Samples].T.values\
        .flatten(), columns=[Samples]).dropna().reset_index(drop=True)
        DF_Combined_Median = pd.concat([DF_Combined_Median, DF_Combined_Median_Temp], axis=1)

    WRITER = pd.ExcelWriter(os.path.join(Save_Path,Exp_Name) + ".xlsx")
    DF_Combined.to_excel(WRITER,'All_ROIs')
    DF_Combined_Mean.to_excel(WRITER,'Mean')
    DF_Combined_Median.to_excel(WRITER,'Median')
    WRITER.save()

    DF_Combined.to_pickle(os.path.join(Save_Path,Exp_Name) + ".pkl")
    DF_Combined_Mean.to_pickle(os.path.join(Save_Path,Exp_Name) + "_Mean.pkl")
    DF_Channel_Median.to_pickle(os.path.join(Save_Path,Exp_Name) + "_Median.pkl")

    print(f"Completion time after user input: {(time.time() - START)} seconds.")

elif not os.path.isdir(Save_Path):
    print("ERROR=> Check save directory path - does not exist.")

elif os.path.isdir(Save_Path) and not os.path.isdir(Channel_Path):
    print("ERROR=> Check Output_C1 directory - does not exist.")
