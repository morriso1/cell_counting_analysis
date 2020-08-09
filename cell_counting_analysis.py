#!/usr/bin/env python
# coding: utf-8


from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from collections import OrderedDict
from skimage import io, morphology, filters, measure, feature, img_as_ubyte
from scipy import ndimage
from scipy.ndimage import morphology as scipy_morphology
from scipy.stats import mode
import cv2
from custom_plotting import custom_plotting as cp
from analysing_imaging_data import analysing_imaging_data as aid


def read_prob_image(fn):
    return io.imread(fn)[0]


def apply_binary_methods(
    binary_image,
    binary_method_list=[
        morphology.binary_erosion,
        morphology.binary_opening,
        [scipy_morphology.binary_fill_holes, morphology.selem.diamond(4)],
        [morphology.binary_erosion, morphology.selem.diamond(3)],
    ],
):
    edited_binary_image = np.copy(binary_image)

    for method in binary_method_list:
        if callable(method):
            edited_binary_image = method(edited_binary_image)

        if not callable(method) and len(method) == 2:
            edited_binary_image = method[0](edited_binary_image, method[1])

        else:
            "Check binary_method_list. Not in correct format"

    return edited_binary_image


def extract_regions_from_binary(
    binary_image, min_area=None, max_area=None, min_sphericity=None
):
    if max_area == None:
        max_area = binary_image.shape[0] * binary_image.shape[1]

    label_image = measure.label(binary_image)

    region_list = list()
    for region in measure.regionprops(label_image):
        if max_area > region.area > min_area:
            region_list.append(region)
    return region_list


def create_binary_image_from_region_list(original_image, region_list):

    new_image = np.zeros(original_image.shape, dtype="bool")

    for region in region_list:
        new_image[
            region.bbox[0]: region.bbox[2], region.bbox[1]: region.bbox[3]
        ] += region.image

    return new_image


def create_dict_of_binary_masks(
    input_dir="Prob_Map_C1",
    thresh_method=filters.threshold_otsu,
    binary_method_list=[
        [morphology.binary_erosion, morphology.selem.diamond(1)],
        [morphology.binary_opening, morphology.selem.diamond(3)],
        [scipy_morphology.binary_fill_holes, morphology.selem.diamond(4)],
        [morphology.binary_erosion, morphology.selem.diamond(1)],
    ],
    min_area=1000,
):

    ic = io.ImageCollection(
        load_pattern=os.path.join(input_dir, "*.tiff"), load_func=read_prob_image
    )

    dict_binary_masks = OrderedDict()

    for i, files in enumerate(ic.files):
        thresh_value = thresh_method(ic[i])
        binary_image = ic[i] > thresh_value
        edited_binary_image = apply_binary_methods(
            binary_image, binary_method_list=binary_method_list
        )
        region_list = extract_regions_from_binary(
            edited_binary_image, min_area=min_area
        )
        dict_binary_masks[files] = create_binary_image_from_region_list(
            edited_binary_image, region_list=region_list
        )

    return dict_binary_masks


def save_dict_binary_masks(
    dict_binary_masks, Save_Dir="Binaries_C1", method_img_type=img_as_ubyte, file_type='.tiff'
):

    if not os.path.isdir(Save_Dir):
        os.mkdir(Save_Dir)

    for key in dict_binary_masks:
        fn = os.path.split(key)[-1]
        fn_path = os.path.join(Save_Dir, fn)
        img = method_img_type((dict_binary_masks[key]))
        io.imsave(fn_path + file_type, img)

    return print(f"Binary images saved at '{Save_Dir}'")


def region_overlap(DF, label_img_outer=None, label_img_inner=None, overlap_thresh=0.5):
    overlap = label_img_outer[label_img_inner == DF["label"]]
    total_overlap_region = overlap.ravel().size
    non_zero_count = np.count_nonzero(overlap)
    ratio_non_zero = non_zero_count / total_overlap_region

    if ratio_non_zero > overlap_thresh:
        is_in = overlap.ravel()[np.nonzero(overlap.ravel())]
        is_in = mode(is_in)[0][0]

    else:
        is_in = 0

    return is_in


def in_region_three_channel(
    C0_img=None,
    C1_img=None,
    C2_img=None,
    C2_overlap_threshold=0.3,
    C1_overlap_threshold=0.5,
    pixel_size=0.4,
):
    label_C0_img = measure.label(C0_img)
    label_C1_img = measure.label(C1_img)
    label_C2_img = measure.label(C2_img)

    C0_props = measure.regionprops_table(
        label_C0_img, properties=["label", "centroid", "area"]
    )

    DF_C0 = pd.DataFrame.from_dict(C0_props, dtype=int)

    DF_C0["C0_in_C2"] = DF_C0.apply(
        region_overlap,
        axis=1,
        label_img_outer=label_C2_img,
        label_img_inner=label_C0_img,
        overlap_thresh=C2_overlap_threshold,
    )

    DF_C0["C0_in_C1"] = DF_C0.apply(
        region_overlap,
        axis=1,
        label_img_outer=label_C1_img,
        label_img_inner=label_C0_img,
        overlap_thresh=C1_overlap_threshold,
    )

    DF_C0.set_index("label", inplace=True)

    DF_C0["area"] = DF_C0["area"] * (pixel_size ** 2)

    return DF_C0


def create_RGB_image_overlapping_regions(DF_C0, C0_img=None):
    rgbArray = np.zeros((C0_img.shape[0], C0_img.shape[1], 3), dtype="uint8")

    label_C0_img = measure.label(C0_img)
    C0_img[C0_img > 0] = 255
    rgbArray[:, :, 2] = C0_img

    for ele in DF_C0[DF_C0["C0_in_C2"] > 0]["C0_in_C2"].index.to_numpy():
        rgbArray[:, :, 2][label_C0_img == ele] = 0

    for ele in DF_C0[DF_C0["C0_in_C1"] > 0]["C0_in_C1"].index.to_numpy():
        rgbArray[:, :, 2][label_C0_img == ele] = 0

    for ele in DF_C0[DF_C0["C0_in_C2"] > 0]["C0_in_C2"].index.to_numpy():
        rgbArray[:, :, 0][label_C0_img == ele] = 255

    for ele in DF_C0[DF_C0["C0_in_C1"] > 0]["C0_in_C1"].index.to_numpy():
        rgbArray[:, :, 1][label_C0_img == ele] = 255

    return rgbArray


def read_image_2d(fn):
    return io.imread(fn)


def test_imagecollections_same_files_and_order(
    *args, pattern=re.compile(r"[a-z][0-9]g[0-9][0-9]?\w[0-9]")
):
    extract_match_args = list()
    for args in args:
        extract_match = [re.findall(pattern, file)[0] for file in args.files]
        extract_match_args.append(extract_match)

    return (
        all(items == extract_match_args[0] for items in extract_match_args),
        extract_match_args[0],
    )


def marcm_save_CSVs_RGB_images_overlapping_regions(
    Bin_C0_Dir="Binaries_C0",
    Bin_C1_Dir="Binaries_C1",
    Bin_C2_Dir="Binaries_C2",
    C2_overlap_threshold=0.3,
    C1_overlap_threshold=0.5,
    pixel_size=0.4,
    csv_save_dir="CSVs_C0_in_C1C2",
    RGB_save_dir="RGB_C0_overlapping_regions",
):

    ic_C0 = io.ImageCollection(
        load_pattern=os.path.join(Bin_C0_Dir, "*.tif*"), load_func=read_image_2d
    )
    ic_C1 = io.ImageCollection(
        load_pattern=os.path.join(Bin_C1_Dir, "*.tif*"), load_func=read_image_2d
    )
    ic_C2 = io.ImageCollection(
        load_pattern=os.path.join(Bin_C2_Dir, "*.tif*"), load_func=read_image_2d
    )

    condition, img_names = test_imagecollections_same_files_and_order(
        ic_C0, ic_C1, ic_C2
    )

    if not condition:
        print(
            f"""{Bin_C0_Dir}, {Bin_C1_Dir}, {Bin_C2_Dir} do not contain the same number of files or 
            they are not in the same order."""
        )
        return

    if not os.path.isdir(csv_save_dir):
        os.mkdir(csv_save_dir)

    if not os.path.isdir(RGB_save_dir):
        os.mkdir(RGB_save_dir)

    Dict_DFs = OrderedDict()

    for C0_img, C1_img, C2_img, names in zip(ic_C0, ic_C1, ic_C2, img_names):
        DF = in_region_three_channel(
            C0_img=C0_img,
            C1_img=C1_img,
            C2_img=C2_img,
            C2_overlap_threshold=C2_overlap_threshold,
            C1_overlap_threshold=C1_overlap_threshold,
            pixel_size=pixel_size,
        )
        Dict_DFs[names] = DF
        DF.to_csv(os.path.join(csv_save_dir, names + ".csv"))
        RGB_img = create_RGB_image_overlapping_regions(DF, C0_img=C0_img)
        io.imsave(os.path.join(RGB_save_dir, names + ".tiff"), RGB_img)
    return Dict_DFs


def analyse_marcm_DFs(DF, DF_name=None, EC_min_area=40):
    DF_new = pd.DataFrame()

    for value in np.sort(DF["C0_in_C1"].unique()):
        DF_new.loc[f"{DF_name}_region_{value}", f"C2neg_C0area>{EC_min_area}um2"] = (
            (DF["C0_in_C1"] == value)
            & (DF["C0_in_C2"] == 0)
            & (DF["area"] > EC_min_area)
        ).sum(0)

        DF_new.loc[f"{DF_name}_region_{value}", f"C2neg_C0area<{EC_min_area}um2"] = (
            (DF["C0_in_C1"] == value)
            & (DF["C0_in_C2"] == 0)
            & (DF["area"] < EC_min_area)
        ).sum(0)

        DF_new.loc[f"{DF_name}_region_{value}", f"C2pos_C0area>{EC_min_area}um2"] = (
            (DF["C0_in_C1"] == value)
            & (DF["C0_in_C2"] != 0)
            & (DF["area"] > EC_min_area)
        ).sum(0)

        DF_new.loc[f"{DF_name}_region_{value}", f"C2pos_C0area<{EC_min_area}um2"] = (
            (DF["C0_in_C1"] == value)
            & (DF["C0_in_C2"] != 0)
            & (DF["area"] < EC_min_area)
        ).sum(0)

        DF_new.loc[f"{DF_name}_region_{value}", "Total"] = (
            DF["C0_in_C1"] == value
        ).sum(0)

    return DF_new


def analyse_marcm_DFs_alt(DF, DF_name=None, EC_min_area=40):
    DF_new = pd.DataFrame()

    for value in np.sort(DF["C0_in_C1"].unique()):
        DF_new.loc[f"{DF_name}_region_{value}", f"C2neg_C0area>{EC_min_area}um2"] = (
            (DF["C0_in_C1"] == value)
            & (DF["C0_in_C2"] == 0)
            & (DF["area"] > EC_min_area)
        ).sum(0)

        DF_new.loc[f"{DF_name}_region_{value}", f"C2neg_C0area<{EC_min_area}um2"] = (
            (DF["C0_in_C1"] == value)
            & (DF["C0_in_C2"] == 0)
            & (DF["area"] < EC_min_area)
        ).sum(0)

        DF_new.loc[f"{DF_name}_region_{value}", f"C2pos"] = (
            (DF["C0_in_C1"] == value) & (DF["C0_in_C2"] != 0)
        ).sum(0)

        DF_new.loc[f"{DF_name}_region_{value}", "Total"] = (
            DF["C0_in_C1"] == value
        ).sum(0)

    return DF_new


def combine_marcm_dict_DFs(
    Dict_DFs, analyse_method=analyse_marcm_DFs_alt, EC_min_area=50, total_col="Total"
):
    Output_DF_num = pd.DataFrame()

    for keys in Dict_DFs:
        DF = analyse_method(
            Dict_DFs[keys], DF_name=keys, EC_min_area=EC_min_area)
        Output_DF_num = Output_DF_num.append(DF)

    cols = Output_DF_num.columns.tolist()
    cols.remove(total_col)

    Output_DF_percentage = (
        Output_DF_num[cols].div(Output_DF_num[total_col], axis=0) * 100
    )

    Output_DF_num.reset_index(inplace=True)
    Output_DF_num.rename(columns={"index": "label"}, inplace=True)

    Output_DF_percentage.reset_index(inplace=True)
    Output_DF_percentage.rename(columns={"index": "label"}, inplace=True)

    return Output_DF_num, Output_DF_percentage


def define_sample_in_or_out_clone(foo):
    if foo.str.contains("a1g[0-9][0-9]?\w[0-9]_region_0")[0]:
        return "a1_outside_clone"
    if foo.str.contains("a1g[0-9][0-9]?\w[0-9]_region_[1-9][0-9]?")[0]:
        return "a1_inside_clone"
    if foo.str.contains("a2g[0-9][0-9]?\w[0-9]_region_0")[0]:
        return "a2_outside_clone"
    if foo.str.contains("a2g[0-9][0-9]?\w[0-9]_region_[1-9][0-9]?")[0]:
        return "a2_inside_clone"
    else:
        return "unknown"


def sorted_DFs_mean_sem(DF, set_index_col="Sample_in_or_out_clone", remove_col="Total"):
    DF = DF.set_index(set_index_col)

    DF_mean = pd.DataFrame()
    DF_sem = pd.DataFrame()

    for col in DF.index.unique():
        DF_mean[f"{col}"] = DF.loc[f"{col}"].mean()
        DF_sem[f"{col}"] = DF.loc[f"{col}"].sem()

    DF_mean = DF_mean.T
    DF_sem = DF_sem.T

    if remove_col in DF_mean.columns:
        DF_mean.drop(columns=remove_col, inplace=True)

    if remove_col in DF_sem.columns:
        DF_sem.drop(columns=remove_col, inplace=True)

    return DF_mean, DF_sem


def create_stack_bar_plot(
    DF,
    DF_error_bar=None,
    Exp_Name=aid.exp_analysis_name(),
    File_Name=None,
    Plot_Name="",
    x_figSize=None,
    y_figSize=2.5,
    y_label=cp.identify_y_axis_label(aid.exp_analysis_name()),
    y_axis_start=0,
    y_axis_limit=None,
    color_pal=sns.color_palette(palette="Blues_r"),
    bar_width=0.8,
):
    if x_figSize == None:
        x_figSize = cp.determine_fig_width(DF)

    fig, ax = plt.subplots(figsize=(x_figSize, y_figSize))

    sns.set(style="ticks")
    sns.despine()

    ax = DF.plot(
        kind="bar",
        stacked=True,
        color=color_pal,
        width=bar_width,
        ax=ax,
        yerr=DF_error_bar,
        capsize=4,
    )
    ax.set_ylabel(y_label, fontsize=12)
    sns.despine(ax=ax)
    ax.xaxis.set_tick_params(width=1)
    ax.yaxis.set_tick_params(width=1)
    ax.tick_params(axis="both", which="major", pad=1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.setp(ax.spines.values(), linewidth=1)

    if not y_axis_limit == None:
        ax.set_ylim(top=y_axis_limit)

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(
        reversed(handles), reversed(labels), bbox_to_anchor=(1, 1), loc="upper left"
    )

    if File_Name == None:
        File_Name = Exp_Name

    if not Plot_Name == "":
        Plot_Name = "_" + Plot_Name

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    plt.savefig(f"{File_Name}{Plot_Name}.pdf",
                transparent=True, bbox_inches="tight")


def watershed_binary_img_to_labelled_img(
    binary_img, gaussian_sigma=3, kernel_size_peak_local_max=np.ones((18, 18))
):
    distance = ndimage.distance_transform_edt(binary_img)
    distance = filters.gaussian(distance, sigma=gaussian_sigma)
    local_maxi = feature.peak_local_max(
        distance, indices=False, footprint=kernel_size_peak_local_max, labels=binary_img
    )
    markers = morphology.label(local_maxi)
    watershed_img = morphology.watershed(
        -distance, markers=markers, mask=binary_img
    )
    return watershed_img


def create_labelled_img_dict_from_folder(label_img_re=r"\w\dg\d\d?", **kwargs):
    img_collection = io.ImageCollection(**kwargs)
    labelled_img_dict = {
        re.search(label_img_re, file)[0]: measure.label(img_collection[i])
        for i, file in enumerate(img_collection.files)
    }
    return labelled_img_dict


def create_img_dict_from_folder(label_img_re=r"\w\dg\d\d?", **kwargs):
    img_collection = io.ImageCollection(**kwargs)
    labelled_img_dict = {
        re.search(label_img_re, file)[0]: img_collection[i]
        for i, file in enumerate(img_collection.files)
    }
    return labelled_img_dict


def measure_region_props_to_tidy_df(
    img_dictionary, labelled_dictionary_img_masks, **kwargs
):

    df_to_append = pd.DataFrame()

    for key, img in img_dictionary.items():
        df = pd.DataFrame(
            measure.regionprops_table(
                labelled_dictionary_img_masks[key], intensity_image=img, **kwargs
            )
        )
        df["image_key"] = key

        df_to_append = df_to_append.append(df).reset_index(drop=True)

    return df_to_append


def outline_contours_labelled_img(
    labelled_img,
    img_to_outline,
    constrast_stretch_low_bound=2,
    constrast_stretch_upper_bound=98,
):
    con_list = list()
    for value in np.unique(labelled_img):
        if not value == 0:
            contours, _ = cv2.findContours(
                img_as_ubyte(labelled_img == value),
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            con_list.extend(contours)
            input_image = cv2.cvtColor(
                img_as_ubyte(img_to_outline), cv2.COLOR_GRAY2RGB
            )

            p_low, p_high = np.percentile(
                input_image,
                (constrast_stretch_low_bound, constrast_stretch_upper_bound),
            )

            input_image = exposure.rescale_intensity(
                input_image, in_range=(p_low, p_high))

            outlined_img = cv2.drawContours(
                input_image, con_list, -1, (255, 0, 0), 2)

    return outlined_img


def num_div_denom_measure_region_props_to_tidy_df(
    num_img_dict,
    denom_img_dict,
    label_imgs,
    sample_id_categories=None,
    properties_num=["label", "area", "mean_intensity"],
    properties_denom=["label", "mean_intensity"],
):
    df = pd.merge(
        measure_region_props_to_tidy_df(
            num_img_dict, label_imgs, properties=properties_num
        ),
        measure_region_props_to_tidy_df(
            denom_img_dict, label_imgs, properties=properties_denom
        ),
        how="left",
        on=("image_key", "label"),
        suffixes=("_num", "_denom"),
    )
    if sample_id_categories is not None:
        df = df.assign(
            mean_intensity_num_div_denom=lambda x: x["mean_intensity_num"]
            / x["mean_intensity_denom"],
            sample_id=lambda x: pd.Categorical(
                x["image_key"].str.split("g", expand=True)[0],
                categories=sample_id_categories,
            ),
            gut_id=lambda x: x["image_key"].str.split("g", expand=True)[1],
        )

    else:
        df = df.assign(
            mean_intensity_num_div_denom=lambda x: x["mean_intensity_num"]
            / x["mean_intensity_denom"],
            sample_id=lambda x: pd.Categorical(
                x["image_key"].str.split("g", expand=True)[0]),
            gut_id=lambda x: x["image_key"].str.split("g", expand=True)[1],
        )

    return df
