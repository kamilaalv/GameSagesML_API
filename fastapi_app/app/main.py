from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import pandas as pd
import numpy as np
from mangum import Mangum
import os
from sentence_transformers import SentenceTransformer, util
from utils import get_suggestions


# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081", "https://localhost:8081",  "http://localhost", "http://localhost:3000", "https://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"], 
    allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin", "User-Agent", "DNT", "Cache-Control", "X-Mx-ReqToken", "Keep-Alive", "X-Requested-With", "If-Modified-Since", "X-CSRF-Token"], 

)

handler = Mangum(app)

os.environ['TRANSFORMERS_CACHE'] = '/tmp'
os.environ['HF_HOME'] = '/tmp'

model = SentenceTransformer('./app/model')

# # Load data
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getcwd(), 'gamesages-43d85821b61c.json')
# client = storage.Client()
# bucket_name = 'games_raw'
# bucket = client.bucket(bucket_name)
# blob_name = 'games_preprocessed_embeddings.csv'
# blob = bucket.blob(blob_name)

# Load data in chunks
def load_data_in_chunks(chunk_size=10000):
    #csv_data = blob.download_as_string()
    path = os.path.join(os.path.dirname(__file__), 'games_preprocessed_embeddings.csv')
    chunks = pd.read_csv(path, usecols=['appid', 'score_rank', 'tags', 'short_description', 'combined_tag_score_embedding1', 'combined_short_desc_score_embedding'], chunksize=chunk_size)
    for chunk in chunks:
        yield chunk

df = pd.concat(load_data_in_chunks())



def string_to_array(s):
    return np.fromstring(s.strip('[]'), sep=',').astype('float32')

for column in ['combined_tag_score_embedding1', 'combined_short_desc_score_embedding']:
    df[column] = df[column].apply(string_to_array)    

# Pydantic models
class GameRequest(BaseModel):
    appid: int
    tags: Optional[List[str]] = []

#root endpoint
@app.post("/")
async def read_root(request:GameRequest):
    return request

# FastAPI route
@app.post("/suggestions")
async def suggestions(request: GameRequest):
    appid = request.appid
    tags = request.tags
    try:
        result = get_suggestions(df=df, input_appid=appid, specific_tags=tags)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
