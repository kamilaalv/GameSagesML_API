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

2. **Install Dependencies**
  ```bash pip install -r fastapi_app/requirements.txt

3. **Download the Model**
  ```bash cd fastapi_app
  py /app/download_model.py

4. **Add Preprocessed Data**
The games_preprocessed_embeddings.csv file is not included in this repository due to its size. You must download it from the provided link and place it in fastapi_app/app/ before running the application.

Download Link: [Link to games_preprocessed_embeddings.csv](https://drive.google.com/file/d/1bx9BP1Pv14MsFI0RMSRGy7GCWwteXfKu/view?usp=sharing).

5. **Run the API Locally**
  ```bash uvicorn fastapi_app.app.main:app --reload


## Deployment
To test the deployed API, you can use the following Lambda link: [Lambda Endpoint] (https://ep4js2tqr3bhiy3m3xoqyydkim0qrvvg.lambda-url.eu-west-2.on.aws/docs#/default/suggestions_suggestions_post)

## Notes
- The games_preprocessed_embeddings.csv file is not included in this repository due to its size. Ensure this file is placed in fastapi_app/app/ before running the application.

- For detailed information about the model, refer to the model/README.md.

## Acknowledgments
[FronkonGames Steam Games Scraper] (https://github.com/FronkonGames/Steam-Games-Scraper) for data collection. 
