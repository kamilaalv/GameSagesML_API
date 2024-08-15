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

# Configure CORS middleware to allow specific origins and methods
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8081", 
        "https://localhost:8081",  
        "http://localhost", 
        "http://localhost:3000", 
        "https://localhost:3000"
    ], 
    allow_credentials=True,
    allow_methods=[
        "GET", "POST", "PUT", "DELETE", 
        "OPTIONS", "PATCH", "HEAD"
    ], 
    allow_headers=[
        "Content-Type", "Authorization", "X-Requested-With", 
        "Accept", "Origin", "User-Agent", "DNT", 
        "Cache-Control", "X-Mx-ReqToken", "Keep-Alive", 
        "X-Requested-With", "If-Modified-Since", "X-CSRF-Token"
    ]
)

# AWS Lambda handler using Mangum
handler = Mangum(app)

# Set environment variables for transformers cache and HF home
os.environ['TRANSFORMERS_CACHE'] = '/tmp'
os.environ['HF_HOME'] = '/tmp'

# Load the pre-trained SentenceTransformer model from the specified directory
model = SentenceTransformer('./app/model')

# Function to load data in chunks to handle large datasets efficiently
def load_data_in_chunks(chunk_size=10000):
    """
    Load the CSV data in chunks to manage memory usage efficiently.
    
    Args:
        chunk_size (int): The number of rows to process at a time.
    
    Yields:
        pd.DataFrame: A chunk of the CSV data.
    """
    path = os.path.join(os.path.dirname(__file__), 'games_preprocessed_embeddings.csv')
    chunks = pd.read_csv(
        path, 
        usecols=[
            'appid', 'score_rank', 'tags', 
            'short_description', 
            'combined_tag_score_embedding1', 
            'combined_short_desc_score_embedding'
        ], 
        chunksize=chunk_size
    )
    for chunk in chunks:
        yield chunk

# Load the full dataset into a DataFrame by concatenating all chunks
df = pd.concat(load_data_in_chunks())

# Convert string representations of arrays into actual NumPy arrays
def string_to_array(s):
    """
    Convert a string representation of an array into a NumPy array.
    
    Args:
        s (str): String representation of the array.
    
    Returns:
        np.ndarray: The converted NumPy array.
    """
    return np.fromstring(s.strip('[]'), sep=',').astype('float32')

# Apply the string_to_array function to relevant columns
for column in ['combined_tag_score_embedding1', 'combined_short_desc_score_embedding']:
    df[column] = df[column].apply(string_to_array)    

# Define Pydantic models for request validation
class GameRequest(BaseModel):
    """
    Pydantic model for validating game request data.
    
    Attributes:
        appid (int): The ID of the app/game.
        tags (Optional[List[str]]): A list of tags associated with the game.
    """
    appid: int
    tags: Optional[List[str]] = []

#root endpoint
@app.post("/")
async def read_root(request:GameRequest):
    return request

# Endpoint to get game suggestions based on input data
@app.post("/suggestions")
async def suggestions(request: GameRequest):
    """
    Endpoint to generate game suggestions based on the provided app ID and tags.
    
    Args:
        request (GameRequest): The incoming game request data.
    
    Returns:
        dict: A dictionary containing the suggested games.
    
    Raises:
        HTTPException: If an error occurs during suggestion generation.
    """
    appid = request.appid
    tags = request.tags
    try:
        result = get_suggestions(df=df, input_appid=appid, specific_tags=tags)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Main entry point for running the FastAPI app locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
