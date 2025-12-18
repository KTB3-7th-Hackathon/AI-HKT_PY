from youtube_loader import load_youtube_script
from cleaner import clean_script
from embedding import embed_texts
from rag_pipeline import cosine_similarity

video_id = "TCaDxE3wXsI"

raw = load_youtube_script(video_id)
cleaned = clean_script(raw)

query_vec = embed_texts([cleaned])[0]

# 기존 벡터 DB 예시
stored_vec = ...  # DB에서 가져온 벡터

score = cosine_similarity(query_vec, stored_vec)
print("similarity:", score)