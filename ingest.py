from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb import PersistentClient
import os

persist_directory = "db"

def main():
    documents = []
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(f"Processing: {file}")
                loader = PDFMinerLoader(os.path.join(root, file))
                documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Use the updated Chroma client
    client = PersistentClient(path=persist_directory)
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client=client)
    
    
    print("Data ingestion completed successfully!")

if __name__ == "__main__":
    main()
