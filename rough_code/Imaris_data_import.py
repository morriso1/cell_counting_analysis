import pandas as pd
import numpy as np
import fnmatch
import os
import time
import sys

STATS_PATH = os.path.join(input("Enter path to Imaris statistic csv folder \n: "))
# check if STATS_PATH exists.
if os.path.isdir(STATS_PATH) == False:
    print("\nPath to Imaris statistic csv folder is incorrect.")
    sys.exit()

PARAMETER = input("What column parameter do you want to study :").capitalize()
EXP_NAME = os.path.join(input("Name of the experiment (no spaces): "))

START = time.time()

for FILE in os.listdir(STATS_PATH):
    if fnmatch.fnmatch(FILE, f'*{PARAMETER}*'):
        print(f"\nFound file {FILE}\n")
        PARAMETER_PATH = os.path.join(STATS_PATH, FILE)
        break
else:
    print("\nNo such parameter exists. Try again.")
    # exit script if file containing parameter name cannot be found.
    sys.exit()

SAVE_PATH = os.path.dirname(STATS_PATH)

DF = pd.read_csv(PARAMETER_PATH, skiprows=3, usecols=[PARAMETER, 'Original Image Name'])

DF.columns = DF.columns.str.replace(' ', '_')
L_IMAGE_NAMES = sorted(list(set(DF.Original_Image_Name)))

DF_GUTS = pd.DataFrame()
DF_GUT_SUM = pd.DataFrame()

for IMAGE_NAME in L_IMAGE_NAMES:
    S_TEMP_GUT = DF[DF.Original_Image_Name == IMAGE_NAME].Volume
    IMAGE_NAME = IMAGE_NAME[0:2]
    S_TEMP_GUT.reset_index(inplace=True, drop=True)
    S_TEMP_GUT.rename(IMAGE_NAME, inplace=True)
    S_TEMP_GUT_SUM = pd.Series(S_TEMP_GUT.sum(), name=IMAGE_NAME)
    DF_GUTS = pd.concat([DF_GUTS, S_TEMP_GUT], axis=1)
    DF_GUT_SUM = pd.concat([DF_GUT_SUM, S_TEMP_GUT_SUM], axis=1)

L_SAMPLES = sorted(list(set([elem[0:2] for elem in L_IMAGE_NAMES])))

ST_DF_GUTS = pd.DataFrame()
ST_DF_GUT_SUM = pd.DataFrame()

for SAMPLE in L_SAMPLES:
    DF_GUTS_T = DF_GUTS[SAMPLE].T
    ST_DF_GUTS_TEMP = pd.DataFrame(DF_GUTS_T.values.flatten(), columns=[SAMPLE])\
    .dropna().reset_index() # column index DF_GUTS_T as series.
    ST_DF_GUTS = pd.concat([ST_DF_GUTS, ST_DF_GUTS_TEMP[SAMPLE]], axis=1)

    DF_GUT_SUM_TEMP = pd.DataFrame(DF_GUT_SUM[SAMPLE].values.T, columns=[SAMPLE])
    ST_DF_GUT_SUM = pd.concat([ST_DF_GUT_SUM, DF_GUT_SUM_TEMP], axis=1)

WRITER = pd.ExcelWriter(os.path.join(SAVE_PATH,EXP_NAME) + ".xlsx")
ST_DF_GUTS.to_excel(WRITER,'All_ROIs')
ST_DF_GUT_SUM.to_excel(WRITER,f'Total {PARAMETER} per gut')
WRITER.save()
ST_DF_GUTS.to_pickle(os.path.join(SAVE_PATH,EXP_NAME) + ".pkl")
ST_DF_GUT_SUM.to_pickle(os.path.join(SAVE_PATH,EXP_NAME) + f"_total_{PARAMETER}_per_gut.pkl")

print(f"Completion time after user input: {(time.time() - START)} seconds.")
