import kagglehub


import json
import os

# Load the credentials from your custom location
with open('kaggle.json') as file:
    creds = json.load(file)

# Set the environment variables
os.environ['KAGGLE_USERNAME'] = creds['username']
os.environ['KAGGLE_KEY'] = creds['key']

# Download latest version
path = kagglehub.dataset_download("subrahmanyambhvsp/faceforensics")

print("Path to dataset files:", path)