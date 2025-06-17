#!/bin/bash

# set path to the directory to donwload the dataset
path_dataset="/mnt/dataset_drive/touch_datasets"

mkdir -p $path_dataset/gelsight
cd $path_dataset/gelsight

# # download object_folder dataset =>done
# echo "Downloading the backbone dataset gelsight/object_folder"
# gdown https://drive.google.com/drive/folders/1kgKj3BhvSN8bF1hI2bjeqhcaCHyxJUss?usp=sharing --folder


# download touch_go dataset
# downloaded only first 5 batches, due to problems accessing the rest
echo "Downloading the backbone dataset gelsight/touch_go"
# gdown https://drive.google.com/drive/folders/1Rpy9ZHCfJjwycj7TMuEbwwHEtH79ls8D?usp=sharing --folder
# gdown https://drive.google.com/file/d/1ggS3unOR2ZIdlj4J2PDCphgE1zDHiDHb/view?usp=drive_link --fuzzy
gdown https://drive.google.com/file/d/1yNP-d4ZEWM3WL3v9CFlpUkw9jc98jYBS/view?usp=drive_link --fuzzy
# # extract files
# cd ./touch_go
# for i in */; do tar -xvf "${i%/}.tar.gz"; done
# rm *.tar.gz

echo "Done!"
