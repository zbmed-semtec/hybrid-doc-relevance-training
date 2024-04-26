sudo pip install gdown
cd data

mkdir -p Split_Dataset/Data
mkdir -p Split_Dataset/Groundtruth

cd Split_Dataset/Data/
gdown https://drive.google.com/uc?id=1GkSgtiiX7BG5LygBq3I0HKRRGEBQ7_Lz -O test.npy
gdown https://drive.google.com/uc?id=1oIPGaplji5jgn3qtlm_HE8lL5HTpeHXl -O train.npy
gdown https://drive.google.com/uc?id=1IhDxuD3jhhXwapeEOBmCgqcp4iFc_nWG -O valid.npy

cd ../Groundtruth
gdown https://drive.google.com/uc?id=15z0lWO2kUR8irIyzV5g3yn9wv2SaoRx0 -O test.tsv
gdown https://drive.google.com/uc?id=1hTS76X6S_slnZZoZRBmlpYqq6T3vEiWF -O train.tsv
gdown https://drive.google.com/uc?id=1FFXIH70kPKtwhtF7TvTQ7ue3M0zGSnuZ -O valid.tsv