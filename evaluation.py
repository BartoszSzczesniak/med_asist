from datetime import datetime
from dotenv import load_dotenv

from beir.datasets.data_loader import GenericDataLoader
from beir.util import download_and_unzip

from langsmith import Client
from langchain.smith import RunEvalConfig
from langchain_core.runnables import RunnableLambda

from utils.config import CONFIG
from llm_chain_with_rag import build_chain

load_dotenv()

# Prepare dataset 

data_path = download_and_unzip(
    url = CONFIG['beir']['url'],
    out_dir = CONFIG['beir']['path']
    )
corpus, queries, qrels = GenericDataLoader(
    data_folder = data_path
    ).load(split="test")

dataset_input = [{"question": question} for question in list(queries.values())[:100]]

# configure langsmith evaluation

client = Client()

dataset_name = "med_assist_evaluation"
    
if client.has_dataset(dataset_name=dataset_name):
    client.delete_dataset(dataset_name=dataset_name)

dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="Test of med_assist project evaluation"
)

client.create_examples(
    dataset_name=dataset_name, 
    inputs=dataset_input)

eval_config = RunEvalConfig(
    evaluators=[
        RunEvalConfig.Criteria("conciseness"),
        RunEvalConfig.Criteria("harmfulness")
        ]
)

chain = build_chain()
format_input = RunnableLambda(lambda d: d.get("question"))

eval_chain = format_input | chain

client.run_on_dataset(
    dataset_name=dataset_name,
    llm_or_chain_factory=eval_chain,
    evaluation=eval_config,
    verbose=True,
    project_name="evaluation_test_"+datetime.now().strftime('%H%M'),
    concurrency_level=1
)