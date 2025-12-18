import vertexai
from vertexai.generative_models import GenerativeModel
import faiss
import numpy as np
import sys

# Import functions from existing modules
from full_pipeline import get_video_transcript, refine_script
from vector import load_data, get_embeddings, build_faiss_index, CSV_PATH

def main():
    print("=== YouTube RAG Pipeline Start ===\n")

    # 1. Prepare Vector Database (Knowledge Base)
    import os
    index = None
    
    # Check if index exists
    if os.path.exists("faiss_index.bin"):
        print(">> 1. Loading existing Vector DB from 'faiss_index.bin'...")
        index = faiss.read_index("faiss_index.bin")
        
        # Load CSV just for metadata (titles/content) lookup, no embedding needed
        df = load_data(CSV_PATH) 
        print(">> Vector DB Loaded.\n")
    else:
        print(">> 1. Building Vector DB from CSV (First Run)...")
        df = load_data(CSV_PATH)
        if df is None:
            print("Error: Could not load article.csv")
            return

        # Embed Content from CSV
        article_texts = df['content'].tolist()
        article_vectors = get_embeddings(article_texts)
        
        if article_vectors is None:
            print("Error: Failed to embed articles.")
            return

        # Build FAISS Index
        faiss.normalize_L2(article_vectors) # Ensure normalization for IP/Cosine
        index = build_faiss_index(article_vectors)
        
        # Save Index
        faiss.write_index(index, "faiss_index.bin")
        print(">> Vector DB Saved to 'faiss_index.bin'.\n")


    # 2. Process YouTube Video (Query)
    # Using a fixed video ID for demo, or could take input
    VIDEO_ID = "TCaDxE3wXsI" 
    print(f">> 2. Fetching & Refining YouTube Video ({VIDEO_ID})...")
    
    raw_script = get_video_transcript(VIDEO_ID)
    if not raw_script:
        print("Error: Failed to fetch transcript.")
        return

    # Try to refine, but handle the 429 Quota logic (if it fails, maybe use raw?)
    refined_script = refine_script(raw_script)
    
    final_query_text = ""
    if refined_script:
        print(">> Refinement Successful.")
        final_query_text = refined_script
    else:
        print("!! Refinement failed (likely 429 Quota). Using raw script for search fallback.")
        final_query_text = raw_script

    # 3. Perform Search
    print(f"\n>> 3. Searching for Top 4 Relevant Articles...")
    print(f"Query Length: {len(final_query_text)} chars")

    # Embed the query (YouTube script)
    # We treat the entire script as one query vector here.
    query_vector = get_embeddings([final_query_text])
    
    if query_vector is None:
        print("Error: Failed to embed query.")
        return

    # Normalize query vector for Cosine Similarity (Index is already IP with normalized vectors)
    faiss.normalize_L2(query_vector)

    # Search Top 4
    k = 4
    distances, indices = index.search(query_vector, k)

    print("\n" + "="*30)
    print(f"   RAG Search Results (Top {k})")
    print("="*30)
    
    for i in range(k):
        idx = indices[0][i]
        score = distances[0][i]
        
        if idx < len(df):
            row = df.iloc[idx]
            title = row.get('title', 'No Title')
            content = row.get('content', '')[:100] + "..." # Preview
            print(f"\n{i+1}. [Score: {score:.4f}] {title}")
            print(f"   Context: {content}")
        else:
            print(f"\n{i+1}. [Unknown Index: {idx}]")

if __name__ == "__main__":
    main()
