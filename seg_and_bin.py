from scipy import ndimage
from skimage import filters, feature, morphology, segmentation
import numpy as np


def watershed_binary_img_to_labelled_img(
    binary_img, gaussian_sigma=3, kernel_size_peak_local_max=np.ones((18, 18))
):
    distance = ndimage.distance_transform_edt(binary_img)
    distance = filters.gaussian(distance, sigma=gaussian_sigma)
    local_maxi = feature.peak_local_max(
        distance, indices=False, footprint=kernel_size_peak_local_max, labels=binary_img
    )
    markers = morphology.label(local_maxi)
    watershed_img = segmentation.watershed(
        -distance, markers=markers, mask=binary_img
    )
    return watershed_img


def gaussian_threshold_remove_small_objects_and_holes(
    img,
    threshold_method,
    gaussian_sigma=3,
    min_hole_size=10,
    min_object_size=100,
    **kwargs_thresh
):
    return morphology.remove_small_holes(
        morphology.remove_small_objects(
            img
            > threshold_method(
                filters.gaussian(img, sigma=gaussian_sigma), **kwargs_thresh
            ),
            min_size=min_object_size,
        ),
        area_threshold=min_hole_size,
    )
