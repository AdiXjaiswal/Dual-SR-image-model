"""Download the Kaggle dataset for this project.

Before running this script, create a Kaggle API token:
1. Go to https://www.kaggle.com/ and sign in.
2. Open your account settings and create a new API token.
3. Download kaggle.json and save it to:
   C:/Users/<YourUsername>/.kaggle/kaggle.json
4. Make sure the file permissions allow read access.
"""
    
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

api.dataset_download_files(
    "dozeradi007/patch-dataset-for-paired-hr-lr-satellite-imagery",
    path="H://Dual-SR-Dataset-Patched", #outpput path
    unzip=True,
)