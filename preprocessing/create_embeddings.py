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

blob_name = 'games_preprocessed.csv'
blob = bucket.blob(blob_name)

csv_data = blob.download_as_string()

df = pd.read_csv(io.StringIO(csv_data.decode('utf-8')))

#converts str to list
df['clean_tags'] = df['clean_tags'].apply(ast.literal_eval)

model = SentenceTransformer('all-mpnet-base-v2')

#method 1: calculates embedding of each tag separately, then gets embeddings' average for each game.
def get_average_embedding(tags):
    # Encode all tags at once and compute the mean
    if not tags or all(tag == "" for tag in tags):
        return np.zeros(model.get_sentence_embedding_dimension())  # Return a zero vector if no tags
    tag_embeddings = model.encode(tags, convert_to_tensor=True).cpu().numpy()
    return np.mean(tag_embeddings, axis=0)

tqdm.pandas()

csv_filename = 'games_preprocessed_embeddings.csv'

# Calculate embeddings for tags
df['tags_embedding1'] = df['clean_tags'].progress_apply(get_average_embedding)

# Calculate embeddings for short descriptions and score ranks
def encode_short_description(desc):
    if desc == "":
        return np.zeros(model.get_sentence_embedding_dimension())
    return model.encode(desc, convert_to_tensor=True).cpu().numpy()

df['short_description'] = df['short_description'].astype(str)
short_desc_embeddings = df['short_description'].progress_apply(encode_short_description)

score_rank_embeddings = model.encode(
    df['score_rank'].astype(str),
    convert_to_tensor=True,
).cpu()

score_rank_embeddings_np = score_rank_embeddings.numpy() if isinstance(score_rank_embeddings, torch.Tensor) else score_rank_embeddings
short_desc_embeddings_np = short_desc_embeddings.numpy() if isinstance(short_desc_embeddings, torch.Tensor) else short_desc_embeddings

# Convert the columns to numpy arrays
tags_embedding1_np = np.vstack(df['tags_embedding1'].values)
short_desc_embeddings_np = np.vstack(short_desc_embeddings.values)

# Ensure the number of rows and dimensions in score_rank_embeddings_np match the other embeddings
assert tags_embedding1_np.shape == score_rank_embeddings_np.shape, "Shape mismatch: tags_embedding1_np and score_rank_embeddings_np"
assert short_desc_embeddings_np.shape == score_rank_embeddings_np.shape, "Shape mismatch: short_desc_embeddings_np and score_rank_embeddings_np"

# Stack the arrays along a new dimension and compute the mean
combined_tag_score_embedding1_np = np.mean(np.stack((tags_embedding1_np, score_rank_embeddings_np), axis=1), axis=1)
combined_short_desc_score_embedding_np = np.mean(np.stack((short_desc_embeddings_np, score_rank_embeddings_np), axis=1), axis=1)

# Assign the combined embeddings back to the dataframe
df['combined_tag_score_embedding1'] = list(combined_tag_score_embedding1_np)
df['combined_short_desc_score_embedding'] = list(combined_short_desc_score_embedding_np)

def array_to_string(arr):
    return np.array2string(arr, separator=',', formatter={'float_kind': lambda x: "%.8f" % x})

for column in ['tags_embedding1', 'combined_tag_score_embedding1', 'combined_short_desc_score_embedding']:
    df[column] = df[column].apply(array_to_string)

df.to_csv(csv_filename, index=False)


csv_blob_name = 'games_preprocessed_embeddings.csv'
csv_blob = bucket.blob(csv_blob_name)
csv_blob.upload_from_filename(csv_filename)

print(f"File {csv_filename} uploaded to {bucket_name} as {csv_blob_name}.")