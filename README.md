# Image analysis tools:

## cell_counting_analysis
Performs cell counting or cell lineage analysis e.g. number of cells of type X inside clone and outside clone.

Takes as input binary segmentations of each channel: nuclei (e.g. DAPI), clonal marker (e.g. GFP) and cell-type marker (e.g. Delta - intestinal stem cell marker). Automatic segmentations are best perform using either classical machine-learning tools (e.g. Ilastik) or deep-learning tools (e.g. cellpose, stardist).

![example_pose](/cca_images/example_pose.png)
