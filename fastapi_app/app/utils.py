import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import torch

model = SentenceTransformer('./app/model')

def get_average_embedding(tags):
    # Encode all tags at once and compute the mean
    if not tags or all(tag == "" for tag in tags):
        return np.zeros(model.get_sentence_embedding_dimension())  # Return a zero vector if no tags
    tag_embeddings = model.encode(tags, convert_to_tensor=True).cpu().numpy()
    return np.mean(tag_embeddings, axis=0)

def pad_embedding(embedding, target_dim=768):
    current_dim = len(embedding)
    if current_dim < target_dim:
        return np.pad(embedding, (0, target_dim - current_dim), mode='constant')
    elif current_dim > target_dim:
        return embedding[:target_dim]
    else:
        return embedding

tqdm.pandas()

def get_suggestions(df, input_appid, specific_tags=[], top_n=20):
    input_tags_score_embedding = False
    input_description_score_embedding = False
    suggestions = []
    similar = []
    app_data = df[df['appid'] == input_appid]
    
    if specific_tags:
        input_tags_embedding = get_average_embedding(specific_tags).astype(np.float32)
        score_rank_embedding = model.encode(str(app_data['score_rank'].iloc[0]), convert_to_tensor=True).cpu().numpy()
        input_tags_score_embedding = (input_tags_embedding + score_rank_embedding) / 2
    else:
        if not app_data['tags'].empty and app_data['tags'].iloc[0]:
            input_tags_score_embedding = np.array(app_data['combined_tag_score_embedding1'].iloc[0], dtype=np.float32)

    if not app_data['short_description'].empty and app_data['short_description'].iloc[0]:
        input_description_score_embedding = np.array(app_data['combined_short_desc_score_embedding'].iloc[0], dtype=np.float32)

    if input_tags_score_embedding is not False:
        input_tags_score_embedding = pad_embedding(input_tags_score_embedding, target_dim=768)
        input_tags_score_embedding_tensor = torch.tensor(input_tags_score_embedding)  # Convert to tensor
        for i in tqdm(range(len(df)), desc="Finding suggestions"):
            if df.iloc[i]['appid'] != input_appid:
                other_embedding = np.array(df.iloc[i]['combined_tag_score_embedding1'], dtype=np.float32)
                other_embedding_tensor = torch.tensor(other_embedding)  # Convert to tensor
                similarity_score = util.pytorch_cos_sim(input_tags_score_embedding_tensor, other_embedding_tensor).item()
                suggestions.append((df.iloc[i]['appid'], similarity_score))
        
        # Sort the similarities and get the top N
        suggestions.sort(key=lambda x: x[1], reverse=True)
        suggestions = suggestions[:top_n]

    if input_description_score_embedding is not False:
        input_description_score_embedding_tensor = torch.tensor(input_description_score_embedding)  # Convert to tensor
        for i in tqdm(range(len(df)), desc="Finding similar"):
            if df.iloc[i]['appid'] != input_appid:
                other_embedding = np.array(df.iloc[i]['combined_short_desc_score_embedding'], dtype=np.float32)
                other_embedding_tensor = torch.tensor(other_embedding)  # Convert to tensor
                similarity_score = util.pytorch_cos_sim(input_description_score_embedding_tensor, other_embedding_tensor).item()
                similar.append((df.iloc[i]['appid'], similarity_score))
        
        # Sort the similarities and get the top N
        similar.sort(key=lambda x: x[1], reverse=True)
        similar = similar[:top_n]

    suggestions = [{'appid': int(s[0]), 'score': float(s[1])} for s in suggestions]
    similar = [{'appid': int(s[0]), 'score': float(s[1])} for s in similar]
     
    return {'suggestions':suggestions,'similar': similar}