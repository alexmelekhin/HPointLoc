# HPointLoc: open dataset and framework for indoor visual localizationbased on synthetic RGB-D images*
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
This repository provides a novel framework **PNTR** for exploring capabilities of new indoor datasets - **HPointLoc**, specially designed to explore detection and loop closure capabilities in Simultaneous Localization and Mapping (SLAM).
Our paper on deep learning-based visual place recognition contains detailed information about these datasets:
 ```bibtex
 HPointLoc: open dataset and framework for indoor visual localizationbased on synthetic RGB-D images*
 ```

**HPointLoc** is based on the popular Habitat simulator from 49 photorealistic indoor scenes from the Matterport3D dataset and contains 76,000 frames.
<p align="center">
  <img src="https://user-images.githubusercontent.com/68793107/130797278-615f72c7-0528-4eff-af95-a7e07bf1fea3.png" />
</p> 

When forming the dataset, considerable attention was paid to the presence of instance segmentation of scene objects, which will allow it to be used in new emerging semantic methods for place recognition and localization
<p align="center">
  <img src="https://user-images.githubusercontent.com/68793107/130794869-ea0388e6-f19c-4c83-989a-64d79622db2a.png" />
</p>



The experiments were held on the **HPointLoc-Val** and **HPointLoc-ALL** datasets.
<p align="center">
  <img src="https://user-images.githubusercontent.com/68793107/130799354-25caaa4e-2156-432e-80df-b6a2becbe8ba.png" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/68793107/130799671-c938881b-faf6-435a-8aea-c3ae006e76a0.png" />
</p>


The image retrieval on **HPointLoc-Val** dataset 

<p align="center">
  <img src="https://user-images.githubusercontent.com/68793107/130798397-4c4eea5a-1b55-4a0a-9f99-7d498c7b8dfc.png" />
</p>

## Quick start to evaluate PNTR pipeline

```bash
git clone --recurse-submodules https://github.com/cds-mipt/HPointLoc
conda env create -f environment.yml
python pipelines/pipeline_evaluate.py --dataset_root /path/to/dataset --image-retrieval 'patchnetvlad' --keypoints-matching 'superpoint_superglue' --optimizer-cloud 'teaser' -f  
```

