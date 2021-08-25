# HPointLoc
Framework for visual place recognition and localization. This project is a complete implementation of paper "HPointLoc: open dataset and framework for indoor visual localizationbased on synthetic RGB-D images"

This repository provides a novel framework **PNTR** for exploring new indoor datasets - **HPointLoc**, specially designed to explore detection and loop closure capabilities in Simultaneous Localization and Mapping (SLAM).

Our paper on deep learning-based visual place recognition contains detailed information about these datasets:
> BibteX: HPointLoc: open dataset and framework for indoor visual localizationbased on synthetic RGB-D images*

#![image](https://user-images.githubusercontent.com/68793107/130793350-4ccb203d-ebbf-4c0d-aa65-ad3f49d0a874.png)
<div style="text-align:center"><img src="https://user-images.githubusercontent.com/68793107/130793350-4ccb203d-ebbf-4c0d-aa65-ad3f49d0a874.png" /></div>

## Quick start to evaluate PNTR pipeline

```bash
git clone --recurse-submodules https://github.com/cds-mipt/HPointLoc
conda env create -f environment.yml
python pipelines/pipeline_evaluate.py --dataset_root /path/to/dataset --image-retrieval 'patchnetvlad' --keypoints-matching 'superpoint_superglue' --optimizer-cloud 'teaser' -f  
```

