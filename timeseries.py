from skimage import io, img_as_float, img_as_uint, img_as_ubyte, measure
from matplotlib import pyplot as plt
import pandas as pd
import os
import re


def create_dict_of_multi_dim_imgs(
    img_re=r"\w\dg\d", path_to_dir="../z_stack_TIFFs", f_extension="C0.tif"
):
    C0_imgs = {
        re.search(img_re, files)[0]: img_as_float(
            io.imread(os.path.join(path_to_dir, files))
        )
        for files in os.listdir(path_to_dir)
        if files.endswith(f_extension)
    }
    return C0_imgs


def save_dict_of_imgs_as_tiff(
    img_dict, save_dir=os.getcwd(), additional_identifier="C", img_dtype_function=img_as_uint,
):
    if type(img_dict) is not dict:
        return print(f"{img_dict} is not a dictionary.")

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for key, value in img_dict.items():
        io.imsave(os.path.join(
            save_dir, f"{key}_{additional_identifier}.tiff"), img_as_uint(value))


def stack_label_images_to_tidy_df(
    label_img_stack,
    num_intensity_img_stack,
    denom_intensity_img_stack,
    properties=["label", "mean_intensity", "area", "centroid"],
):
    features = pd.DataFrame()
    for i, lab_img in enumerate(label_img_stack):
        df = pd.DataFrame(
            measure.regionprops_table(
                lab_img,
                intensity_image=img_as_uint(num_intensity_img_stack[i, :, :]),
                properties=properties,
            )
        )
        df["frame"] = i
        df.rename(
            columns={
                "mean_intensity": "mean_intensity_num",
                "centroid-0": "y",
                "centroid-1": "x",
            },
            inplace=True,
        )
        df["mean_intensity_denom"] = pd.DataFrame(
            measure.regionprops_table(
                lab_img,
                intensity_image=img_as_uint(
                    denom_intensity_img_stack[i, :, :]),
                properties=["mean_intensity"],
            )["mean_intensity"]
        )
        df["mean_intensity_num_denom"] = (
            df["mean_intensity_num"] / df["mean_intensity_denom"]
        )
        features = features.append(df)

    return features
