# HPointLoc
Framework for visual place recognition and localization. This project is a complete implementation of paper "HPointLoc: open dataset and framework for indoor visual localizationbased on synthetic RGB-D images"


## Quick start to evaluate PNTR pipeline

```bash
git clone --recurse-submodules https://github.com/cds-mipt/HPointLoc
```

```bash
conda env create -f environment.yml
```

```bash
python pipelines/pipeline_evaluate.py --dataset_root /path/to/dataset --image-retrieval 'patchnetvlad' --keypoints-matching 'superpoint_superglue' --optimizer-cloud 'teaser' -f  
```
