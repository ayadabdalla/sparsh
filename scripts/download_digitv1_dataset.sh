#!/bin/bash

# set path to the directory to donwload the dataset
path_dataset="/mnt/dataset_drive/touch_datasets"

mkdir -p $path_dataset/digitv1
cd $path_dataset/digitv1

echo "Downloading the backbone dataset for DIGITv1..."
gdown https://drive.google.com/drive/folders/19vs-5dSqakiJ96ykBdHbhDuc8EoYK0eg?usp=sharing --folder --remaining-ok

# extract files
cd ./Object-Slide
# extracting using tar in this way doesn't work, it is done manually
# for i in */; do tar -xvf "${i%/}.tar.gz"; done
# rm *.tar.gz

echo "Done!"
