import numpy as np
import pandas as pd
from skimage import io, measure, morphology, filters, img_as_ubyte, img_as_uint
from cell_counting_analysis import cell_counting_analysis as cca


def combine_label_images_return_df_of_regions(
    label_img_inner, label_img_outer, overlap_thresh=0.1
):
    df_lab_inner = pd.DataFrame(
        measure.regionprops_table(
            label_img_inner, properties=("label", "bbox", "image")
        )
    )
    df_lab_outer = pd.DataFrame(
        measure.regionprops_table(
            label_img_outer, properties=("label", "bbox", "image")
        )
    )
    overlapping_labels = df_lab_inner.apply(
        cca.region_overlap,
        axis=1,
        label_img_outer=label_img_outer,
        label_img_inner=label_img_inner,
        overlap_thresh=overlap_thresh,
    ).unique()

    labels_in_outer_not_inner = np.unique(label_img_outer)[
        ~np.isin(np.unique(label_img_outer), overlapping_labels)
    ]

    combined_df_of_regions = pd.concat(
        [df_lab_inner, df_lab_outer.query(
            "label in @labels_in_outer_not_inner")]
    ).reset_index(drop=True)

    combined_df_of_regions["label"] = combined_df_of_regions.index + 1

    return combined_df_of_regions


def create_label_image_from_df_of_regions(original_image, df):

    new_image = np.zeros(original_image.shape, dtype="uint16")

    for row in range(df.shape[0]):
        temp_single_img = img_as_uint(df.loc[row, "image"])
        temp_single_img[df.loc[row, "image"]] = df.loc[row, "label"]

        new_image[
            df.loc[row, "bbox-0"]: df.loc[row, "bbox-2"],
            df.loc[row, "bbox-1"]: df.loc[row, "bbox-3"],
        ][
            new_image[
                df.loc[row, "bbox-0"]: df.loc[row, "bbox-2"],
                df.loc[row, "bbox-1"]: df.loc[row, "bbox-3"],
            ]
            == 0
        ] = temp_single_img[
            new_image[
                df.loc[row, "bbox-0"]: df.loc[row, "bbox-2"],
                df.loc[row, "bbox-1"]: df.loc[row, "bbox-3"],
            ]
            == 0
        ]

    return new_image


def combine_label_image_dicts(
    label_img_inner_dict=None, label_img_outer_dict=None, overlap_thresh=0.1
):
    if type(label_img_inner_dict) is not dict:
        print(f"{label_img_inner_dict} is not a dict")
        return

    if type(label_img_outer_dict) is not dict:
        print(f"{label_img_outer_dict} is not a dict")
        return

    combined_lab_img_dict = dict()
    for key in label_img_inner_dict.keys():
        #         print(key)
        df_combined = combine_label_images_return_df_of_regions(
            label_img_inner=label_img_inner_dict[key],
            label_img_outer=label_img_outer_dict[key],
            overlap_thresh=overlap_thresh,
        )
        combined_lab_img_dict[key] = create_label_image_from_df_of_regions(
            label_img_inner_dict[key], df_combined
        )
    return combined_lab_img_dict
