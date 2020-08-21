from skimage import io, img_as_float, img_as_uint, img_as_ubyte
from matplotlib import pyplot as plt
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
