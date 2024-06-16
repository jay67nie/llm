import json
import os
from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from openai import OpenAI
from openai.lib.streaming import AssistantEventHandler
from typing_extensions import override

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


def set_sector(selected_sector):
    global db, client, retriever, compression_retriever, openai_client, thread_id, sector
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
        db = Chroma(persist_directory="./chroma_db", collection_name=f"{sector}_guide",
                    embedding_function=OpenAIEmbeddings())
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3})
        compressor = FlashrankRerank(top_n=3)
        compression_retriever = ContextualCompressionRetriever(base_retriever=retriever, base_compressor=compressor)
        openai_client = OpenAI()
        thread_id = "thread_AYA7B1cxHaA1Pbl08Gt5H62M"
    else:
        raise ValueError("Invalid sector provided")


def initialize_database(sector):
    db = Chroma(persist_directory="./chroma_db", collection_name=f"{sector}_guide",
                embedding_function=OpenAIEmbeddings())
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    compressor = FlashrankRerank(top_n=3)
    compression_retriever = ContextualCompressionRetriever(base_retriever=retriever, base_compressor=compressor)
    return compression_retriever


def create_thread():
    openai_client = OpenAI()
    thread = openai_client.beta.threads.create()

    thread_id = thread.id

    with open("thread.txt", "w") as file:
        file.write(thread_id)

    return thread_id


def delete_thread(t_id):
    openai_client.beta.threads.delete(thread_id=t_id)


def create_system_prompt(sector):
    return f"""The user is posing as a person in the {sector} sector who is a taxpayer. So the person in the {sector} 
    sector, in the user question should be substituted for taxpayer in the passed-in context """


def contextualize_query_for_retriever(query, chat_history):
    local_chat_history = chat_history + [("user", "{question}")]
    chat_prompt = ChatPromptTemplate.from_messages(local_chat_history)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
    chain = chat_prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
    llm_response = chain.invoke({"question": query})
    print("LLM Response: ", llm_response)
    return llm_response[0]


chat_history = []
contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question as the user persona \
    which can be understood by the document retriever without the chat history. Do NOT answer the question! Strictly \
    just restructure it as a question for the retriever. I repeat, do not answer the user question!!!! \
    Only restructure the question!!!"""
chat_history.append(('system', contextualize_q_system_prompt))


def build_chat_history(query, response):
    chat_history.append(('user', query))
    chat_history.append(('assistant', response))
    print("\nChat History: ", chat_history)


def chat_with_assistant(query):
    global db, thread_id, sector, compression_retriever, retriever, openai_client

    print("##  Sector: ## ", sector)
    if db is None:
        raise ValueError("Database has not been set. Please set the sector first.")

    if len(chat_history) > 1:
        contextualized_query = contextualize_query_for_retriever(query, chat_history)
    else:
        contextualized_query = query

    reranked_results = compression_retriever.get_relevant_documents(contextualized_query)
    # print("COMPRESSOR 2 RESULTS ", compression_retriever)

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

    with openai_client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=os.environ.get("ASSISTANT_ID"),
            temperature=0.2,
            event_handler=EventHandler()
    ) as stream:
        stream.until_done()
        messages = openai_client.beta.threads.messages.list(thread_id=thread_id, limit=1, order="desc")
        msg_json = json.loads(messages.to_json())
        if msg_json["data"][0]["role"] == "assistant":
            response = msg_json["data"][0]["content"][0]["text"]["value"]
            build_chat_history(query, response)
            return response
        else:
            return "Assistant response not found"
