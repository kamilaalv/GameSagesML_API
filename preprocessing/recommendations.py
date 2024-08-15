import pandas as pd
import numpy as np
from google.cloud import storage
import os
import io
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import ast
import torch


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/credentials"

client = storage.Client()
bucket_name = 'games_raw'
bucket = client.bucket(bucket_name)

blob_name = 'games_preprocessed_embeddings.csv'
blob = bucket.blob(blob_name)

csv_data = blob.download_as_string()

df = pd.read_csv(io.StringIO(csv_data.decode('utf-8')))

def string_to_array(s):
    return np.fromstring(s.strip('[]'), sep=',')
for column in ['tags_embedding1', 'combined_tag_score_embedding1', 'combined_short_desc_score_embedding']:
    df[column] = df[column].apply(string_to_array)    