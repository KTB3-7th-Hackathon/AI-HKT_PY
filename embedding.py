from google import genai
import test
from google.genai import types

client = genai.Client(api_key="AIzaSyAa-TXqxVWLX3ITxz0FJf_3tAfEKuoibck")

print(test.filtered_script)




def embed_texts(texts: list[str], dim=768):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(
            output_dimensionality=dim
        )
    )
    print(result.embeddings)
    return [e.values for e in result.embeddings]
