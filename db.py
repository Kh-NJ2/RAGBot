import openai
import os
from dotenv import load_dotenv    # reads .env files
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import shutil    # lets you move, copy, delete files & folders at a higher level than os.


PATH = "files"
CHROMA_PATH = "chroma"

load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']   # reads the api OPENAI_API_KEY value from .env file


def documents_loader():
    loader = DirectoryLoader(PATH, glob="*.md")
    documents = loader.load()

    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database if it already exists.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH)   # turns chunks into embeddings and saves the vectors to the chroma path
    db.persist()    # saves to disk so they stay stored for future use.
    
    
def load_store():
    documents = documents_loader()
    chunks = split_text(documents)
    save_to_chroma(chunks)
     

if __name__ == "__main__":
    load_store()


  