#!/bin/bash

python main.py --mode test --cacheBatchSize 24 --resume /datasets/netvlad_v100_datasets/models/vgg16_netvlad_checkpoint --metadata_json_file /datasets/Habitat/metadata.json --output_json_file /datasets/Habitat/results.json 
