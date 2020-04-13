import pandas as pd
import numpy as np
import xpressplot as xp
from matplotlib import pyplot as plt
import os
import seaborn as sns

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def gene_name_search(df, gene=None):
    if not type(gene) == str:
        return print("Please provide identifier as string.")

    gene_name = df.index[df.index.str.contains(gene)]

    if gene_name.shape != (1,):
        return print("More than one  identifier match. Try again")

    return gene_name[0]


def create_xp_graphs_using_gene_lists(
    data_scaled_fil,
    df_metadata,
    txt_file_path="../Helen_RNAseq_Aging/Pathway_txt_files",
    colors=None,
):

    txt_file_list = [file for file in os.listdir(txt_file_path) if ".txt" in file]

    for txt_file in txt_file_list:
        file_name = txt_file.replace(".txt", "")
        df = pd.read_csv(os.path.join(txt_file_path, txt_file), sep="\t")
        print(file_name)
        gene_fly = df["current_id"].tolist()
        gene_list = [gene_name_search(data_scaled_fil, x) for x in gene_fly]
        num_in_data_scaled_fil = np.sum(data_scaled_fil.index.isin(gene_list))

        xp.multigene_overview(
            data_scaled_fil,
            df_metadata,
            title=f"{file_name}: {num_in_data_scaled_fil} genes",
            gene_list=gene_list,
            palette=colors,
            save_fig=f"{file_name}_{num_in_data_scaled_fil}_genes_violin_plot.pdf",
        )

        xp.heatmap(
            data_scaled_fil,
            df_metadata,
            sample_palette=colors,
            gene_list=gene_list,
            center=0,
            row_cluster=True,
            col_cluster=False,
            figsize=(15, 10),
            save_fig=f"{file_name}_{num_in_data_scaled_fil}_genes_heatmap.pdf",
        )
