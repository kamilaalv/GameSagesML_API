# GameSages ML API

## Overview

This repository contains the Machine Learning API used in a mobile application to recommend Steam games. The API is built with FastAPI and is deployed using AWS Lambda.

## Project Structure

- **fastapi_app/**: Contains the FastAPI application code.
  - **app/**: 
    - `main.py`: Entry point for the API.
    - `utils.py`: Utility functions used in the API.
    - `models.py`: Data models for the API.
    - **model/**: Model files and configurations used by the API.
  - **Dockerfile**: Configuration for Docker container.
  - **requirements.txt**: Python dependencies.
- **preprocessing/**: Scripts used for data preprocessing and embedding creation.
  - `create_embeddings.py`: Script to create game embeddings.
  - `data_preprocessing.py`: Data cleaning and preparation script.
  - `recommendations.py`: Logic for generating game recommendations.

## Data Flow

1. **Data Scraping**: Data was initially scraped from Steam and Steam Spy using [this Scraper](https://github.com/FronkonGames/Steam-Games-Scraper).

2. **Data Storage**: The scraped data was stored in Google Cloud Storage (GCS).

3. **Data Preprocessing**: The data was preprocessed and saved in GCS.

4. **API Development**: The FastAPI application was developed to serve the model's predictions. For the final deployment, the preprocessed data is stored in `games_preprocessed_embeddings.csv` within the `app/` folder due to Docker memory constraints.

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r fastapi_app/requirements.txt
