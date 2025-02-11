# RAG_with_reranking
RAG with reranking from Cohere reranker. The large language model is GPT-2. This project is about text-generation to generate a short story from such a model.

You need to register the Upstash and Cohere to get
UPSTASH_VECTOR_REST_URL, UPSTASH_VECTOR_REST_TOKEN, COHERE_API_KEY

Run: embed_to_upstash.py to upsert vectors from Stanford dataset: https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/viewer/stanford

Once your Upstash database is filled with vectors, you can generate a story by
Run: gen_txt.py
