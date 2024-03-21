## Goal

Build a chatbot providing medical information.

## Stack

- **Methods:** LLM + RAG
- **Tools:** HuggingFace + LangChain + ChromaDB

## Dataset

NFCorpus from BEIR datasets.

NFCorpus is a full-text English retrieval data set for Medical Information Retrieval. It contains a total of 3,244 natural language queries (written in non-technical English, harvested from the NutritionFacts.org site) with 169,756 automatically extracted relevance judgments for 9,964 medical documents (written in a complex terminology-heavy language), mostly from PubMed.

NFCorpus details: https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/  
BEIR details: https://github.com/beir-cellar/beir

## Model for RAG Embedding
GIST Embedding: Guided In-sample Selection of Training Negatives for Text Embedding  

According to MTEB English leaderboard on 21-02-2024:
- #14 best embedding model in the Overall rating
- #18 best embedding model in Retrieval tasks
- #1  best embedding model with the size below 1GB

Details: https://huggingface.co/avsolatorio/GIST-Embedding-v0  
Leaderboard: https://huggingface.co/spaces/mteb/leaderboard  

## LLM for answer Generation

Fine tuned 4-bit quantized Llama2 model.

**Base model description:**

- **Series:** LLama2
- **Version:** Llama-2-7b-chat-hf
- **Parameters:** 7B
- **Use case:** Tuned for assistant-like chat

Details: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

**Quantization:**   

- 4 bits  

**Fine tuning:** 
- **Method:** PEFT QLora  
- **Lora rank:** 64
- **Lora alpha:** 16 
- **Epochs:** 5
- **Training dataset:** Training dataset was obtained by running the base model on a series of queries relevant to the task and reviewing all the answers by human. Revised and corrected data (in format *prompt:answer*) was used for fine tuning. 2000 records were utilized as a training dataset and ~500 records as a validation dataset.