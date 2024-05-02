import yaml
import chromadb
import importlib.resources
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
from med_assist import data
from med_assist.config import CONFIG
from med_assist.components.retriever import Retriever
from med_assist.components.llm import Llama2
from med_assist.components.parsers import BooleanOutputParser

def build_chain():

    # RAG collection

    collection = chromadb.PersistentClient(
        path=CONFIG['chromadb']["path"]
    ).get_collection(
        name=CONFIG["chromadb"]["collection"]
    )

    # RAG model / retriever

    emb_model = SentenceTransformer(model_name_or_path=CONFIG['gist']['path'])
    retriever = Retriever(model=emb_model, collection=collection, n_results=1)
        
    # LLM

    llm = Llama2(model=CONFIG['llama']['tuned_path'])

    # prompt template

    with open(importlib.resources.files(data) / 'prompt_templates.yaml', "r") as file:
        prompt_templates = yaml.safe_load(file)
    
    relevance_prompt = PromptTemplate(
        input_variables=['question', 'context'],
        template=prompt_templates.get("relevance")
        )

    answer_prompt = PromptTemplate(
        input_variables=['question', 'context'],
        template=prompt_templates.get("answer")
        )

    no_answer_prompt = PromptTemplate(
        input_variables=['question',],
        template=prompt_templates.get("no_answer")
        )

    conditional_prompt = RunnableLambda(
        lambda inputs: \
            answer_prompt.invoke(inputs.get("prompt_vars")) \
            if inputs.get('relevance') \
            else no_answer_prompt.invoke(inputs.get("prompt_vars"))
        )
    
    # parsers

    boolean_output_parser = BooleanOutputParser()
    answer_parser = StrOutputParser()

    # Med assist chain

    relevance_checker = (relevance_prompt | llm | boolean_output_parser)

    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | RunnableParallel({"relevance": relevance_checker, "prompt_vars": RunnablePassthrough()})
        | conditional_prompt | llm | answer_parser
    ) 

    return chain

if __name__ == "__main__":

    chain = build_chain()

    while True:
        question = input("\nQuestion: ")
        
        print("Answer: ", end="", flush=True)
        for s in chain.stream(question):
            print(s, end="", flush=True)