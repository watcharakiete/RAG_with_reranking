from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from functools import partial
from datasets import Dataset
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
import copy
from upstash_vector import Index, Vector
import datetime, json
from tqdm import tqdm

import constants as const ## Tokens and stuff here

"""
Functions
"""
def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds

def extract_only_text(data):
    ext_txt_data = copy.deepcopy(data)
    for dat in ext_txt_data:
        split_str = (dat.page_content).split("text:")
        split_str = (split_str[1].split("seed_data:"))
        dat.page_content = split_str[0]
    return ext_txt_data

def create_embeddings(modelPath, model_kwargs, encode_kwargs):
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

if __name__ == "__main__":

    """ 
    Batch upsert for each day
    """
    start_date = datetime.datetime(2025, 1, 27, 16, 0, 0, 0)
    now = datetime.datetime.now()
    days_from_start = (now-start_date).days
    print(f"Start date is: {start_date}. Now is {now}, so days from start = {days_from_start}\n")

    batch = 10
    for i in range(5):
    ## Load dataset from Huggingface Stanford
        data = load_dataset("HuggingFaceTB/cosmopedia", "stanford", split="train", streaming=True)
        num = 50
        start_iter_ds = num*days_from_start
        end_iter_ds = start_iter_ds+num

        ## Read from ./status_to_upsert_into_Upstash.json
        with open('./status_to_upsert_into_Upstash.json', 'r') as file:
            status = json.load(file)
        prev_last_id = status["Current last id (in Upstash)"]
        
        iterable_ds = data.take(start_iter_ds+(i+1)*batch)
        iterable_ds = iterable_ds.skip(start_iter_ds+i*batch)

        ds = Dataset.from_generator(partial(gen_from_iterable_dataset, iterable_ds), features=iterable_ds.features)
        data = ds
        data = data.to_pandas()
        data.to_csv("stanford_dataset.csv")

        ## Load from saved .csv
        loader = CSVLoader(file_path='./stanford_dataset.csv')
        data = loader.load()
        ext_txt_docs = extract_only_text(data)

        ## Split page_content.text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=40)
        docs = text_splitter.split_documents(ext_txt_docs)

        ## Create model for embeddings 
        modelPath = "sentence-transformers/all-MiniLM-l6-v2"
        model_kwargs = {'device':'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = create_embeddings(modelPath, model_kwargs, encode_kwargs)

        ## Embed docs into vecs
        # Generate embeddings for each chunk of text
        doc_func = lambda x: x.page_content
        docs = list(map(doc_func, docs))
        doc_embeddings = embeddings.embed_documents(docs)
        print(f"Length of split docs: {len(docs)}.")

        ### Batch vecs to Upstash
        # Write days_from_start and curr_last_id in .txt
        curr_last_id = prev_last_id+len(docs)
        dictionary = {
            "Date": str(now),
            "Days from start": days_from_start,
            "Current last id (in Upstash)": curr_last_id
        }
        with open("./status_to_upsert_into_Upstash.json", "w") as outfile:
            json.dump(dictionary, outfile)

        ## Initialize Upstash and index
        UPSTASH_VECTOR_REST_URL = const.UPSTASH_VECTOR_REST_URL
        UPSTASH_VECTOR_REST_TOKEN = const.UPSTASH_VECTOR_REST_TOKEN

        index = Index(
            url=UPSTASH_VECTOR_REST_URL,
            token=UPSTASH_VECTOR_REST_TOKEN
        )

        ## Upsert vectors to Upstash
        vectors = []
        for j in range(len(doc_embeddings)):
            vec = Vector(id=f"{prev_last_id+1+j}", vector = doc_embeddings[j], metadata = {"text": docs[j]})
            vectors.append(vec)
        index.upsert(vectors=vectors)
