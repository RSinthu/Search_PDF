import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64
import textwrap
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import chromadb
from accelerate import infer_auto_device_map

persist_directory = "db"

checkpoint = "LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map="auto",
    offload_folder="offload_weights",  # Specify folder for offloaded weights
)

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path=persist_directory)

    db = Chroma(
        persist_directory=persist_directory, embedding_function=embeddings, client=chroma_client
    )
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return qa

def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer,generated_text

def main():
    st.title('Search your pdf')
    with st.expander("About the app"):
        st.markdown(
            "This is a generative ai powered question and answering app that responds to questions about your pdf file"
        )

    question = st.text_area("Enter your question:")
    if  st.button("Get Answer"):
        st.info("Your question: "+question)
        st.info("Your answer")
        answer,metadata = process_answer(question)
        st.write(answer)
        st.write(metadata)

if __name__ == '__main__':
    main()