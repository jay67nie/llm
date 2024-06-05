import json
import os
from typing_extensions import override

from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
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
    just restructure it as a question for the retriever"""

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

# Retrieve from an existing collection
client = Chroma(persist_directory="./chroma_db")

sector = input("Enter the sector: ")

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

db = Chroma(client=client, collection_name=f"{sector}_guide", embedding_function=OpenAIEmbeddings())

retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3})

compressor = FlashrankRerank(top_n=3)
compression_retriever = ContextualCompressionRetriever(base_retriever=retriever, base_compressor=compressor)

openai_client = OpenAI()

thread_id = openai_client.beta.threads.create().id
# thread_id = create_thread()  # Call create_thread() to create a new thread


# TODO: Print thread to file called thread.txt
# with open(f"{sector}_thread.txt", "w") as f:
#     f.write(thread.id)


def chat_with_assistant():
    # Make the retriever history aware
    query = input('Enter your query: ')

    if len(chat_history) > 1:
        contextualized_query = contextualize_query_for_retriever(query)
    else:
        contextualized_query = query

    reranked_results = compression_retriever.get_relevant_documents(contextualized_query)

    for result in reranked_results:
        print("Result", result.page_content, "\n")

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
        role="assistant",
        content=create_system_prompt(sector)
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
        temperature=0,
        event_handler=EventHandler()
    ) as stream:
        stream.until_done()

    # if run.status == "completed":
        messages = openai_client.beta.threads.messages.list(thread_id=thread_id, limit=1, order="desc")

        msg_json = json.loads(messages.to_json())
        # print(msg_json)
        if msg_json["data"][0]["role"] == "assistant":
            build_chat_history(query, msg_json["data"][0]["content"][0]["text"]["value"])

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

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

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


while True:
    chat_with_assistant()
    user_input = input("Do you want to continue chatting? (yes/no): ")
    if user_input.lower() == "no":
        delete_thread(thread_id)
        break
    else:
        continue

