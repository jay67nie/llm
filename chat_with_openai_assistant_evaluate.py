import json
import os
from typing_extensions import override

import chromadb
from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.chroma import Chroma
from openai import OpenAI
from openai.lib.streaming import AssistantEventHandler

# TODO: Add streaming support for the chat response. (Jerome) ✔️
# TODO: Create a new thread for each user, and delete it when the chat is done. (Samson)
# TODO: Modify the functionality to suit frontend, but maintain the backend logic. (Samson)
# TODO: Suggest a persona for the user to use when chatting with the assistant. (Jerome) ✔️
# TODO: Evaluate the responses from the assistant. (Jerome)
chat_history = []

contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question as the user persona \
    which can be understood by the document retriever without the chat history. Do NOT answer the question! Strictly \
    just restructure it as a question for the retriever. I repeat, do not answer the user question!!!!
    Only restructure the question!!!"""

chat_history.append(('system', contextualize_q_system_prompt))

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")


class EventHandler(AssistantEventHandler):
    @override
    def on_text_created(self, text) -> None:
        print(f"\n", end="", flush=True)

    @override
    def on_text_delta(self, delta, snapshot):
        print(delta.value, end="", flush=True)

    def on_tool_call_created(self, tool_call):
        print(f"\n {tool_call.type}\n", flush=True)

    def on_tool_call_delta(self, delta, snapshot):
        if delta.type == 'code_interpreter':
            if delta.code_interpreter.input:
                print(delta.code_interpreter.input, end="", flush=True)
            if delta.code_interpreter.outputs:
                print(f"\n\n", flush=True)
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        print(f"\n{output.logs}", flush=True)

client = None
retriever = None
compression_retriever = None
openai_client = OpenAI()
thread_id = None
sector = None
# Retrieve from an existing collection

def set_sector(selected_sector):
# sector = input("Enter the sector: ")
    global db
    global client, retriever, compression_retriever, thread_id, sector
    sector_mapping = {
        "agriculture": "agric",
        "construction": "const",
        "education": "education",
        "health": "health",
        "manufacturing": "manufacturing",
        "hotel accommodation": "hotel_accommodation",
        "retail": "wholesale_retail",
        "wholesale": "wholesale_retail"
    }
    sector_key = next((key for key in sector_mapping if key in selected_sector.lower()), None)

    if sector_key:
        sector = sector_mapping[sector_key]

        print(f"Setting sector to {sector}")
        db = Chroma(persist_directory="./chroma_db", collection_name=f"{sector}_guide", embedding_function=OpenAIEmbeddings())
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3})
        compressor = FlashrankRerank(top_n=3)
        compression_retriever = ContextualCompressionRetriever(base_retriever=retriever, base_compressor=compressor)
        openai_client = OpenAI()
        thread_id = openai_client.beta.threads.create().id
    else:
        raise ValueError("Invalid sector provided")
    
    
     

    # client = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
    

    # thread_id = create_thread()  # Call create_thread() to create a new thread


    # TODO: Print thread to file called thread.txt
    # with open(f"{sector}_thread.txt", "w") as f:
    #     f.write(thread.id)


def chat_with_assistant(query):
    # Make the retriever history aware
    # query = input('Enter your query: ')
    global db, thread_id, sector, compression_retriever, retriever, openai_client
    
    print("##  Sector: ## ", sector)
    if db is None:
        raise ValueError("Database has not been set. Please set the sector first.")
    
   

    if len(chat_history) > 1:
        contextualized_query = contextualize_query_for_retriever(query)
    else:
        contextualized_query = query

    reranked_results = compression_retriever.get_relevant_documents(contextualized_query)
    
    print("COMPRESSOR 2 RESULTS ", compression_retriever)

    for result in reranked_results:
        print("Result", result, "\n")

    user_message = f"""
    <context>
    {reranked_results}
    </context>

    <question>
    {query}
    </question>
    """

    print("User message: ", user_message)

    openai_client.beta.threads.messages.create(
        thread_id=thread_id,
        role="assistant",
        content=create_system_prompt(sector),
    )

    openai_client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_message
    )
    # print(messages)

    # run = openai_client.beta.threads.runs.create_and_poll(
    #     thread_id=thread_id,
    #     assistant_id=os.environ.get("ASSISTANT_ID"),
    #     temperature=0
    # )

    with openai_client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=os.environ.get("ASSISTANT_ID"),
            temperature=0.2,
            event_handler=EventHandler()
    ) as stream:
        stream.until_done()

        # if run.status == "completed":
        messages = openai_client.beta.threads.messages.list(thread_id=thread_id, limit=1, order="desc")

        msg_json = json.loads(messages.to_json())
        # print(msg_json)
        if msg_json["data"][0]["role"] == "assistant":
            response = msg_json["data"][0]["content"][0]["text"]["value"]
            build_chat_history(query, response)
            return response
        else:
             return "Assistant response not found"


    # else:
    #     print(run.status)


def build_chat_history(query, response):
    # print("GPT-35 Response: ", response)

    chat_history.append(('user', query))
    chat_history.append(('assistant', response))

    print("\nChat History: ", chat_history)


def contextualize_query_for_retriever(query):
    local_chat_history = chat_history + [("user", "{question}")]
    chat_prompt = ChatPromptTemplate.from_messages(
        local_chat_history,
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    chain = chat_prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))

    llm_response = chain.invoke({"question": query})

    print("LLM Response: ", llm_response)

    return llm_response[0]


# Call when user is done chatting / leaves the website
def delete_thread(t_id):
    openai_client.beta.threads.delete(thread_id=t_id)


# Call to create a new thread
def create_thread():
    thread = openai_client.beta.threads.create()
    return thread.id


def create_system_prompt(sector):
    return f"""The user is posing as a person in the {sector} sector who is a taxpayer. So the person in the  {sector} 
    sector, in the user question
    should be substituted for taxpayer in the passed-in context """


# while True:
#     chat_with_assistant()
#     user_input = input("Do you want to continue chatting? (yes/no): ")
#     if user_input.lower() == "no":
#         delete_thread(thread_id)
#         break
#     else:
#         continue

