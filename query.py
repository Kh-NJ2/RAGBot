from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import argparse      # Allows the script to read command-line arguments


PROMPT_TEMPLATE = """
You must answer **only** using this context:

{context}

---

Question: {question}
Answer:
"""


PATH = "chroma"


def main():
    parser = argparse.ArgumentParser()       # read command-line arguments

    # expects a "query_text" of type string
    parser.add_argument("query", type=str)

    args = parser.parse_args()
    query = args.query

    embedding_fun = OpenAIEmbeddings()

    db = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory=PATH)
    # persist_directory → where DB files are saved.
    # embedding_function → tells Chroma how to turn text into vectors.

    results = db._similarity_search_with_relevance_scores(query, k=3)
    # embeds query_text, finds the 3 closest matches in the DB.

    if len(results) == 0 or results[0][1] < 0.7:
        print("unable to find matching results")
        return
    

    context = "\n-------------\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=query)
    model = ChatOpenAI()
    response = model.predict(prompt)

    print( f"\nResponse: {response}\n")

if __name__ == "__main__":
    main()    

