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

## LLM for answer Generation

4-bit quantized Llama2-7b-chat-hf model.

Original model description:
- **Series:** LLama2
- **Version:** Llama-2-7b-chat-hf
- **Parameters:** 7B
- **Use case:** Tuned for assistant-like chat

https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

Quantization description:
- **Version:** llama-2-7b-chat.Q4_K_M.gguf  
- **Quant method:** Q4_K_M  
- **Bits:** 4  
- **Size:** 4.08 GB  
- **Max RAM required:** 6.58 GB  
- **Use case:** medium, balanced quality - recommended  

https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF