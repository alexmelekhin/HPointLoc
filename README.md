# HPointLoc: open dataset and framework for indoor visual localization based on synthetic RGB-D images
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a novel framework PNTR for exploring the capabilities of a new indoor dataset - **HPointLoc**, specially designed to explore detection and loop closure capabilities in Simultaneous Localization and Mapping (SLAM).

**HPointLoc** is based on the popular Habitat simulator from 49 photorealistic indoor scenes from the Matterport3D dataset and contains 76,000 frames.
<p align="center">
  <img src="https://user-images.githubusercontent.com/68793107/130797278-615f72c7-0528-4eff-af95-a7e07bf1fea3.png" />
</p> 

When forming the dataset, considerable attention was paid to the presence of instance segmentation of scene objects, which will allow it to be used in new emerging semantic methods for place recognition and localization
<p align="center">
  <img src="https://user-images.githubusercontent.com/68793107/130794869-ea0388e6-f19c-4c83-989a-64d79622db2a.png" />
</p>

The dataset is split into two parts: the validation HPointLoc-Val, which contains only one scene, and the complete HPointLoc-All dataset, containing all 49 scenes, including HPointLoc-Val
<p align="center">
  <img src="https://user-images.githubusercontent.com/68793107/130804077-ac2665fe-0f1f-4229-9486-af7c0e0a762e.png" />
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/68793107/130805029-d76ce041-10a4-47c4-91dd-52a50908ff39.png" />
</p>

**HPointLoc Val** dataset is available by the link:
https://drive.google.com/drive/folders/14_UBWF4CgLiwdmbj0GFFWzgHPITjCcNx?usp=sharing

## Experimental results

The experiments were held on the **HPointLoc-Val** and **HPointLoc-ALL** datasets.
<p align="center">
  <img src="https://user-images.githubusercontent.com/68793107/130799354-25caaa4e-2156-432e-80df-b6a2becbe8ba.png" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/68793107/130799671-c938881b-faf6-435a-8aea-c3ae006e76a0.png" />
</p>


<!-- The image retrieval problem on **HPointLoc-Val** dataset (NetVLAD case)

<p align="center">
  <img src="https://user-images.githubusercontent.com/68793107/130798397-4c4eea5a-1b55-4a0a-9f99-7d498c7b8dfc.png" />
</p> -->


## Quick start to evaluate PNTR pipeline

```bash
git clone --recurse-submodules https://github.com/cds-mipt/HPointLoc.git
```
```bash
Download models from https://drive.google.com/drive/folders/192c_XEn12Pz0pmD3aEwV8t3QBqwlBPo1?usp=sharing to PNTR folder
cd HPointLoc
conda env create -f environment.yml
conda activate PTNR_pipeline 
python /path/to/HPointLoc_repo/pipelines/utils/exctracting_dataset.py --dataset_path /path/to/dataset/HPointLoc_dataset
python pipelines/pipeline_evaluate.py --dataset_root /path/to/extracted_dataset --image-retrieval patchnetvlad --keypoints-matching superpoint_superglue --optimizer-cloud teaser
```

