import chromadb

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.llms.llamacpp import LlamaCpp

from sentence_transformers import SentenceTransformer

from utils.config import CONFIG
from utils.classes import STCDBRetriever

def build_chain():

    # RAG collection

    collection = chromadb.PersistentClient(
        path=CONFIG['chromadb']["path"]
    ).get_collection(
        name=CONFIG["chromadb"]["collection"]
    )

    # LLM

    llm = LlamaCpp(
        model_path=CONFIG['llama']['path'],
        temperature=0.25,
        max_tokens=1024,
        top_p=1,
        n_ctx=1024
    )
    llm.client.verbose = False

    # RAG model / retriever

    emb_model = SentenceTransformer(model_name_or_path=CONFIG['gist']['path'])
    retriever = STCDBRetriever(model=emb_model, collection=collection, n_results=1)

    # prompt template

    llm_prompt_template = """

[INST] <<SYS>>
You are a helpful and concise assistant. Always return a concise list of facts regarding the question based on the provided context. 
The list should not include any harmful, unethical or illegal content, it should be socially unbiased and positive in nature.
The list should be based only on the provided context information and no prior knowledge.
Include only information relevant to the question and include all the details.
If the provided context does not contain relevant information, concisely answer that there is no information available on this topic.
If the question does not make any sense, or is not factually coherent, explain that the question is invalid.
<</SYS>>
Context: {context}

Question: {question} 
[/INST]

Answer: Based on the provided context, here is the list of facts regarding your question: 

"""

    llm_prompt = PromptTemplate(
        input_variables=['question', 'context'],
        template=llm_prompt_template
        )

    # Output parser

    parser = StrOutputParser()

    # Med assist chain

    return {"context": retriever, "question": RunnablePassthrough()} | llm_prompt | llm | parser

if __name__ == "__main__":

    chain = build_chain()

    while True:
        question = input("\nQuestion: ")
        
        print("Answer: ", end="", flush=True)
        for s in chain.stream(question):
            print(s, end="", flush=True)