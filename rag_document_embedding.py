import configparser
import chromadb
from beir.datasets.data_loader import GenericDataLoader
from beir import util
from sentence_transformers import SentenceTransformer

def format_document(doc_data: dict):

    title = doc_data.get('title')
    text = doc_data.get('text')

    return f"{title}; {text}"

def run():

    config = configparser.ConfigParser()
    config.read("config.cfg")

    # Download dataset

    data_path = util.download_and_unzip(
        url = config['beir']['BEIR_URL'],
        out_dir = config['beir']['BEIR_PATH']
        )
    corpus, _, _ = GenericDataLoader(
        data_folder = data_path
        ).load(split="train")

    # Prepare keys and documents for a vector database

    ids = list(corpus.keys())
    docs = [format_document(corpus.get(id)) for id in ids]

    # Prepare embedding model

    # GIST Embedding: Guided In-sample Selection of Training Negatives for Text Embedding
    # https://huggingface.co/avsolatorio/GIST-Embedding-v0
    #
    # According to MTEB English leaderboard on 21-02-2024:
    # - #14 best embedding model in the Overall rating
    # - #18 best embedding model in Retrieval tasks
    # - #1  best embedding model with the size below 1GB
    # https://huggingface.co/spaces/mteb/leaderboard

    emb_model = SentenceTransformer("avsolatorio/GIST-Embedding-v0")

    # Embed documents

    doc_embeddings = emb_model.encode(docs, show_progress_bar=True).tolist()

    # Load embeddings to a vetor database 

    chromadb. \
        PersistentClient(path = config['chromadb']['DB_PATH']). \
        get_or_create_collection(name = config['chromadb']['DB_COLL']). \
        add(
            ids=ids,
            embeddings=doc_embeddings,
            documents=docs
            )
    
if __name__ == "__main__":
    run()