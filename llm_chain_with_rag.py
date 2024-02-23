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
        Generate the answer to the question using only given context and no prior knowledge.
        Your answer should be as concise as possible.
        If the context is empty, answer only that you have no information on this topic.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
    """

    llm_prompt = PromptTemplate(
        input_variables=['question', 'context'],
        template=llm_prompt_template
        )

    # Output parser

    parser = StrOutputParser()

    # Med assist chain

    return {"context": retriever, "question": RunnablePassthrough()} | llm_prompt | llm | parser

def run_io_script(chain):

    while True:
        
        question = input("\nQuestion: ")

        print("Answer: ", end="", flush=True)

        for s in chain.stream(question):
            print(s, end="", flush=True)

if __name__ == "__main__":

    chain = build_chain()

    run_io_script(chain)