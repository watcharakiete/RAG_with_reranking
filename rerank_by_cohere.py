import cohere
import constants as const

COHERE_API_KEY = const.COHERE_API_KEY

def res_from_upstash_to_dict(res):
    ## Get strings from Upstash & transform into dictionary
    ## to be compatible with cohere reranker
    str_list_from_upstash = list(set([r.metadata["text"] for r in res]))
    res_dict = [{"text": e} for e in str_list_from_upstash]
    return res_dict


def rerank_res_from_upstash(query, res):
    res_dict = res_from_upstash_to_dict(res)
    ## Initialize cohere to rerank dictionary result 
    ## from Upstash vector (dot similarity) using user's original query
    cohere_client = cohere.Client(COHERE_API_KEY)
    reranked_docs = cohere_client.rerank(
        query=query,
        documents=res_dict, # Dictionary result from Upstash vector
        model="rerank-english-v3.0",
        rank_fields=list(res_dict[0].keys()),
        top_n=5
    )
    ## Combine 5 top-reranked strings into a list
    reranked_str_list = [(res_dict[hit.index])["text"] for hit in reranked_docs.results]
    return reranked_str_list

