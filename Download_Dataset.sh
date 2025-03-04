pip install gdown
cd data

mkdir -p Split_Dataset/Data
mkdir -p Split_Dataset/Ground_truth

cd Split_Dataset/Data/

# Pre-annotation apporaches
gdown https://drive.google.com/uc?id=1p7noO4lOgf942FXBVwd40OrmFO5Go0Hn -O test_annotated.npy
gdown https://drive.google.com/uc?id=1u089d0apWGPCsoRu7NR8ubbBfxWNvrvi -O train_annotated.npy
gdown https://drive.google.com/uc?id=120QdvlnaG_hD_X1IHObCAdd2zjPpJ7KA -O valid_annotated.npy


# Post/Post-reduction annotation approaches
gdown  https://drive.google.com/uc?id=1IgxzmajC09aUTz_awABPD_iEJ2s1VR0E -O test.npy
gdown  https://drive.google.com/uc?id=1xCoXFl0GTc5VbW7L-7joAL9ViudD0Nbh -O train.npy
gdown https://drive.google.com/uc?id=1De4n5hf6kirLpMlILD2LTFTuwsgpVhKQ -O valid.npy

cd ../Ground_truth
gdown  https://drive.google.com/uc?id=1y9T41Faf9Oq2XOtWMD1U9fZe9OHLgLjv -O test.tsv
gdown  https://drive.google.com/uc?id=1R1i74XWzILnlozwCfYItlequKIhMnHmB -O train.tsv
gdown https://drive.google.com/uc?id=1ZupxAdTOWxmKPWlD5FOwEKbkdavt5Zxk -O valid.tsv