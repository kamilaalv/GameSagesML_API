import pandas as pd
import numpy as np
from google.cloud import storage
import json
import os
import io

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/credentials"

client = storage.Client()
bucket_name = 'games_raw'
bucket = client.bucket(bucket_name)

json_blob_name = 'games_raw.json'
blob = bucket.blob(json_blob_name)

data = json.loads(blob.download_as_string())

df = pd.read_json(io.StringIO(json.dumps(data)))

df = df.T.reset_index()
df.columns = ['appid'] + list(df.columns[1:])

#important: removing duplicates
df['name_developers'] = df.apply(lambda row: (row['name'], tuple(row['developers'])), axis=1)
df.sort_values(by='appid', inplace=True)
duplicates = df[df.duplicated(subset=['name_developers'], keep=False)]
df.drop_duplicates(subset=['name_developers'], keep='first', inplace=True)

#removing useless for ml rows
empty_rows_df = df[(df['tags'].apply(lambda x: len(x) == 0)) & (df['short_description'] == "")]
df = df.drop(empty_rows_df.index)

#metacritic score of 0 does not mean 0; it means that there was no data in the api
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['required_age'] = pd.to_numeric(df['required_age'], errors='coerce')
df['metacritic_score'] = pd.to_numeric(df['metacritic_score'], errors='coerce')
df['peak_ccu'] = pd.to_numeric(df['peak_ccu'], errors='coerce')

df.drop(columns=['user_score', 'score_rank',  'average_playtime_forever', 'average_playtime_2weeks',
       'median_playtime_forever', 'median_playtime_2weeks', 'name_developers'], inplace=True)

def get_score_rank(row):
    total = row['positive'] + row['negative']
    
    if total == 0:
        return 'Unknown'
        
    positive_percent = (row['positive'] / total) * 100

    if positive_percent >= 95 and total >= 500:
        return 'Overwhelmingly Positive'
        
    elif positive_percent >= 80:
        if total >= 50:
            return 'Very Positive'
        elif total >= 10:
            return 'Positive'
    
    elif positive_percent >= 70:
        return 'Mostly Positive'
    
    elif positive_percent >= 40:
        return 'Mixed'

    elif positive_percent >= 20:
        return 'Mostly Negative'
    
    elif positive_percent >= 0:
        if total >= 500:
            return 'Overwhelmingly Negative'
        elif total >= 50:
            return 'Very Negative'
        elif total >= 10:
            return 'Negative'
    
    return 'Unknown'

df['score_rank'] = df.apply(get_score_rank, axis=1)
df['clean_tags'] = df['tags'].apply(lambda x: list(x.keys()) if isinstance(x, dict) else x)

csv_filename = 'games_preprocessed.csv'
df.to_csv(csv_filename, index=False)

csv_blob_name = 'games_preprocessed.csv'
csv_blob = bucket.blob(csv_blob_name)
csv_blob.upload_from_filename(csv_filename)

print(f"File {csv_filename} uploaded to {bucket_name} as {csv_blob_name}.")


#save clean df to json for backend
df_reversed = df.T.reset_index()
df_reversed_index = df_reversed.iloc[1:,0]
df_reversed = df_reversed.iloc[: , 1:]
df_reversed.columns = df_reversed.iloc[0]
df_reversed = df_reversed[1:].reset_index(drop=True)
df_reversed.index=df_reversed_index

json_clean_filename = 'games_clean.json'
df_reversed.to_json('games_clean.json', orient='columns')

json_clean_blob_name = 'games_clean.json'
json_clean_blob = bucket.blob(json_clean_blob_name)
json_clean_blob.upload_from_filename(json_clean_filename)

print(f"File {json_clean_filename} uploaded to {bucket_name} as {json_clean_blob_name}.")
