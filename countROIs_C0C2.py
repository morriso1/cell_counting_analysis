import pandas as pd
import glob
import os
import time
import re

# Get cwd and define experiment name
Cwd = os.getcwd()
L_Cwd_folders = [items.replace(" ", "_") for items in Cwd.split("/")]

if "Anterior" or "Posterior" in L_Cwd_folders:
    Exp_Analysis_Name = (
        L_Cwd_folders[-5]
        + "_"
        + L_Cwd_folders[-4]
        + "_"
        + L_Cwd_folders[-2]
        + "_"
        + L_Cwd_folders[-1]
    )
else:
    Exp_Analysis_Name = (
        L_Cwd_folders[-4] + "_" + L_Cwd_folders[-3] + "_" + L_Cwd_folders[-1]
    )

PathOutC0 = os.path.join(Cwd, "Output_C0")
PathOutC2 = os.path.join(Cwd, "Output_C2")

if os.path.isdir(PathOutC0) and os.path.isdir(PathOutC2):
    StartTime = time.time()

    CSVpaths_C0 = sorted(glob.glob(os.path.join(PathOutC0, "*csv")))
    CSVpaths_C2 = sorted(glob.glob(os.path.join(PathOutC2, "*csv")))

# RowIndexes = [[Files_C0[len(PathOutC0) + 1 : len(PathOutC0) + 6]]]
ROINumbersC0 = [pd.read_csv(Files_C0).shape[0] for Files_C0 in CSVpaths_C0]
ROINumbersC2 = [pd.read_csv(Files_C2).shape[0] for Files_C2 in CSVpaths_C2]

RE_Match_C0List = list()
for Files in CSVpaths_C0:
    RE_Match = re.search("\w\dg\d+", Files)
    RE_Match_C0List.append(RE_Match.group(0))

RE_Match_C2List = list()
for Files in CSVpaths_C0:
    RE_Match = re.search(r"\w\dg\d+", Files)
    RE_Match_C2List.append(RE_Match.group(0))

if RE_Match_C2List != RE_Match_C0List:
    print("Check files lists. Lists not identical")

DF_C0C2Results = pd.DataFrame(
    list(zip(RE_Match_C0List, ROINumbersC0, ROINumbersC2)), columns=["Gut", "C0", "C2"]
)

DF_C0C2Results["Percentage"] = (DF_C0C2Results["C2"] / DF_C0C2Results["C0"]) * 100
DF_C0C2Results.to_csv(Exp_Analysis_Name + ".csv")
print(f"Completion time after user input: {(time.time() - StartTime)} seconds.")
