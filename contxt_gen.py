from embed_to_upstash import create_embeddings
from rerank_by_cohere import rerank_res_from_upstash

def query_vec_from_upstash(index, prompt_from_usr):
    ## Create embeddings to embed user's question to Upstash vector
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"
    model_kwargs = {'device':'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = create_embeddings(modelPath, model_kwargs, encode_kwargs)
    ## Embed question & select top 20 similar vectors from DB
    ## by using Upstash index
    question_embedding = embeddings.embed_documents([prompt_from_usr])
    res = index.query(vector=question_embedding[0], top_k=20, include_metadata=True)
    return res
    

def gen_contxt(index, prompt_from_usr):
    ## Get similar 20 vectors from user's question
    res = query_vec_from_upstash(index, prompt_from_usr)
    ## Rerank such 20 vecs to 5
    ## then generate context (string)
    reranked_res_str_list = rerank_res_from_upstash(prompt_from_usr, res)
    context = "Context: "+", ".join(reranked_res_str_list)+"."
    return context
