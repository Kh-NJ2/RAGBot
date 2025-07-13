# **RAGBot — Simple Local AI Search Bot**  

This project is a tiny Retrieval-Augmented Generation (RAG) pipeline.  
It uses OpenAI to embed your text chunks, stores them in a local Chroma vector DB, and lets you ask questions that get answered using only relevant context.

<br>

## How It Works :
1. Store Data — Your documents get split into chunks, embedded with OpenAI, and stored in Chroma.

2. Search — You run a query from the command line.

3. Retrieve — It searches Chroma for the most relevant chunks.

4. Generate — It feeds the chunks to an OpenAI chat model to generate an answer.

5. Respond — It prints the answer.

<br>

## Requirements
- Python 3.10+

- langchain

- langchain_community

- langchain_openai

- unstructured[md]

- chromadb

- openai

- python-dotenv

<br>

## **How to use :**
1. install the dependenies in the requirements.txt file.
```python
pip install -r requirements.txt
```
2. Setup .env file with your API key
```python
OPENAI_API_KEY=YOUR_KEY_HERE
```

3. Create a chroma database
```python
python db.py
```

4. Ask a Question
```python
python query.py "What secret power do pigeons wish they had?"
```

