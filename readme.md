## Goal

Build a chatbot providing medical information.

## Stack

- **Methods:** LLM + RAG
- **Tools:** HuggingFace + LangChain + ChromaDB

## Dataset
Beir dataset:  
https://github.com/beir-cellar/beir

## Model for RAG Embedding
GIST Embedding: Guided In-sample Selection of Training Negatives for Text Embedding  
https://huggingface.co/avsolatorio/GIST-Embedding-v0

According to MTEB English leaderboard on 21-02-2024:
- #14 best embedding model in the Overall rating
- #18 best embedding model in Retrieval tasks
- #1  best embedding model with the size below 1GB

https://huggingface.co/spaces/mteb/leaderboard