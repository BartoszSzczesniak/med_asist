import os
from tqdm.autonotebook import tqdm
from datetime import datetime

import torch

from beir.datasets.data_loader import GenericDataLoader
from beir.util import download_and_unzip

from langsmith import Client

from utils.config import CONFIG
from llm_chain_with_rag import build_chain
from rag_document_embedding import format_document

os.environ['LANGCHAIN_PROJECT'] = "med_assist_training_data_prep"

DATASET_NAME = "med_assist_training_dataset" + datetime.now().strftime('_%m%d-%H%M')
DATASET_DESCR = "Medical assistant training dataset"

client = Client()

data_path = download_and_unzip(
    url = CONFIG['beir']['url'],
    out_dir = CONFIG['beir']['path']
    )
corpus, queries, qrels = GenericDataLoader(
    data_folder = data_path
    ).load(split="train")

if client.has_dataset(dataset_name=DATASET_NAME):
    raise NameError("Dataset name already exist")

dataset = client.create_dataset(
    dataset_name=DATASET_NAME,
    description=DATASET_DESCR
)

chain = build_chain()

make_input = chain.steps[0] | chain.steps[1]
make_output = chain.steps[2] | chain.steps[3]

ids = list(corpus.keys())
docs = [format_document(corpus.get(id)) for id in ids]

for doc in tqdm(docs):

    doc_input = make_input.invoke(doc)
    doc_output = make_output.invoke(doc_input)

    example_input = {"prompt": doc_input.to_string()}
    example_output = {"output": doc_output}

    client.create_example(
        dataset_name=DATASET_NAME,
        inputs=example_input,
        outputs=example_output
        )
    
    torch.cuda.empty_cache()