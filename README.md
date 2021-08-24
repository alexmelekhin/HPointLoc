# HPointLoc
Framework for visual place recognition and localization. This project is a complete implementation of paper "HPointLoc: open dataset and framework for indoor visual localizationbased on synthetic RGB-D images"

```bash
git clone --recurse-submodules https://github.com/cds-mipt/HPointLoc
```


```bash
python pipelines/pipeline_evaluate.py --dataset_root ~/Myfolder/habitat/ --image-retrieval 'patchnetvlad' --keypoints-matching 'superpoint_superglue' --optimizer-cloud 'g2o' -f  
```
