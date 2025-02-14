from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline
import torch
from upstash_vector import Index

from contxt_gen import gen_contxt
import constants as const

model_name = "openai-community/gpt2" # The possible to use here

print(f"We use the model:{model_name}.")

## Initialize the model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    #return_tensors='pt',
    max_length=250,
    max_new_tokens=150,
    model_kwargs={"torch_dtype": torch.bfloat16},
    #device="cuda",
    truncation=True
)

prompt_from_usr = "Write a fantasy story for young children. Once upon a time, there was a girl living with a dragon"

# Initialize Upstash
UPSTASH_VECTOR_REST_URL = const.UPSTASH_VECTOR_REST_URL
UPSTASH_VECTOR_REST_TOKEN = const.UPSTASH_VECTOR_REST_TOKEN

index = Index(
    url=UPSTASH_VECTOR_REST_URL,
    token=UPSTASH_VECTOR_REST_TOKEN
)

## Generate context
## Query vectors from Upstash for k=20,
## then reranking by Cohere down to 5
## & return text
context = gen_contxt(index, prompt_from_usr)
print(f"{context}")
prompt = f"{context}\n\n{prompt_from_usr}"

### Making inference from the model
res = (pipe(prompt))[0]["generated_text"]
print(res)

f = open("Stories.txt", "a")
f.write(f"{res}\n\n")
f.close()
