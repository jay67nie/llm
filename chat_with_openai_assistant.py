import json
import os

import chromadb
from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.chroma import Chroma
from openai import OpenAI

#TODO: Add streaming support for the chat response. (Jerome)
#TODO: Create a new thread for each user, and delete it when the chat is done. (Samson)
#TODO: Modify the functionality to suit frontend, but maintain the backend logic. (Samson)
#TODO: Suggest a persona for the user to use when chatting with the assistant. (Jerome)
#TODO: Evaluate the responses from the assistant. (Jerome)
# TODO: Save chat history to chroma db with thread_id as the collection name when the user is done chatting.
# TODO: If the user deletes a thread, remember to delete the chat history as well.

chat_history = []

contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question as the user persona \
    which can be understood by the document retriever without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

chat_history.append(('system', contextualize_q_system_prompt))

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")

# Retrieve from an existing collection
db = None
thread_id = "thread_ylT9GI1bxvJSv0rmeNAEP5sQ"



def set_sector(sector):
    # sector = input("Enter the sector: ")
    global db

    if sector.lower().__contains__("agric"):
        sector = "agric"

    elif sector.lower().__contains__("const"):
        sector = "const"

    elif sector.lower().__contains__("education"):
        sector = "education"

    elif sector.lower().__contains__("health"):
        sector = "health"

    elif sector.lower().__contains__("manufacturing"):
        sector = "manufacturing"

    elif sector.lower().__contains__("hotel"):
        sector = "hotel_accommodation"

    elif sector.lower().__contains__("retail") or sector.lower().__contains__("wholesale"):
        sector = "wholesale_retail"

    db = Chroma(persist_directory="./chroma_db", collection_name=f"{sector}_guide", embedding_function=OpenAIEmbeddings())



# TODO: Print thread to file called thread.txt
# with open(f"{sector}_thread.txt", "w") as f:
#     f.write(thread.id)


def chat_with_assistant(query):
    # Make the retriever history aware
    # query = input('Enter your query: ')
    global db
    if db is None:
        raise ValueError("Database has not been set. Please set the sector first.")

    
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10})

    compressor = FlashrankRerank(top_n=10)
    compression_retriever = ContextualCompressionRetriever(base_retriever=retriever, base_compressor=compressor)

    openai_client = OpenAI()

    # thread = openai_client.beta.threads.create()

    if len(chat_history) > 1:
        contextualized_query = contextualize_query_for_retriever(query)
    else:
        contextualized_query = query

    reranked_results = compression_retriever.get_relevant_documents(contextualized_query)

    # for result in reranked_results:
    #     print(result.page_content, "\n")

    user_message = f"""
    <context>
    {reranked_results}
    </context>
    
    <question>
    {query}
    </question>
    """
    openai_client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_message
    )
    # print(messages)

    run = openai_client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=os.environ.get("ASSISTANT_ID"),
        temperature=0
    )

    if run.status == "completed":
        messages = openai_client.beta.threads.messages.list(thread_id=thread_id, limit=1, order="desc")

        msg_json = json.loads(messages.to_json())
        # print(msg_json)
        if msg_json["data"][0]["role"] == "assistant":
            response = response = msg_json["data"][0]["content"][0]["text"]["value"]
            build_chat_history(query, response)
            return response

    else:
        return run.status
        print(run.status)


def build_chat_history(query, response):
    print("GPT-35 Response: ", response)

    chat_history.append(('user', query))
    chat_history.append(('assistant', response))

    print("Chat History: ", chat_history)


def contextualize_query_for_retriever(query):
    local_chat_history = chat_history + [("user", "{question}")]
    chat_prompt = ChatPromptTemplate.from_messages(
        local_chat_history,
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    chain = chat_prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))

    llm_response = chain.invoke({"question": query})

    print("LLM Response: ", llm_response)

    return llm_response[0]


# while True:
#     chat_with_assistant()
#     user_input = input("Do you want to continue chatting? (yes/no): ")
#     if user_input.lower() == "no":
#         # client.delete_collection(f"{thread_id}")
#         # client.create_collection(f"{thread_id}", documents=chat_history)
#         break
#     else:
#         continue

