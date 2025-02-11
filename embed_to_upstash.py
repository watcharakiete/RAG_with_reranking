from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from functools import partial
from datasets import Dataset
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
import copy
from upstash_vector import Index, Vector
import datetime, json

