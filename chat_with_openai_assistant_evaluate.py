# Import necessary libraries
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

# Load environment variables
load_dotenv()
# Set the OpenAI API key from the environment variable
# This enables us to connect to the OpenAI ChatGPT LLM via API calls
os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")

contextualize_q_system_prompt = """Given a chat history and the latest user query \
            which might reference context in the chat history, formulate a standalone query as the user persona \
            which can be understood by the document retriever without the chat history. Do NOT answer the query! Strictly \
            just restructure it as a query for the retriever. I repeat, do not answer the user query!!!! \
            Only restructure the query!!!"""


# Define an event handler for the OpenAI assistant
# This is used to handle the assistant's responses for streaming,
# as documented in the OpenAI documentation.
# However, we failed to stream on the frontend, but it does stream
# in the backend
# This overrides internal methods in the AssistantEventHandler as built by OpenAI
class EventHandler(AssistantEventHandler):
    # Override the on_text_created method to print a newline
    @override
    def on_text_created(self, text) -> None:
        print(f"\n", end="", flush=True)

    # Override the on_text_delta method to print the delta value
    @override
    def on_text_delta(self, delta, snapshot):
        print(delta.value, end="", flush=True)

    # Define what happens when a tool call is created
    def on_tool_call_created(self, tool_call):
        print(f"\n {tool_call.type}\n", flush=True)

    # Define what happens when a tool call delta is received
    def on_tool_call_delta(self, delta, snapshot):
        if delta.type == 'code_interpreter':
            if delta.code_interpreter.input:
                print(delta.code_interpreter.input, end="", flush=True)
            if delta.code_interpreter.outputs:
                print(f"\n\n", flush=True)
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        print(f"\n{output.logs}", flush=True)


# Function to set the sector
def set_sector(selected_sector):
    global sector, openai_client, thread_id, chat_history, contextualize_q_system_prompt
    # Define a mapping from sector names to database keys to identify the sector guide
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
    # Find the key for the selected sector
    sector_key = next((key for key in sector_mapping if key in selected_sector.lower()), None)

    if sector_key:
        # Set the sector to the corresponding key
        sector = sector_mapping[sector_key]
        print(f"Setting sector to {sector}")
        # Initialize the chat history
        chat_history = []
        # Add the system prompt to the chat history
        # This is used to restructure the user question for the retriever
        chat_history.append(('assistant', contextualize_q_system_prompt))

        # Set up the database for the selected sector
        # and the retrieval system
        initialize_database(sector)

        # Initialize the OpenAI client
        # This is used to interact with the OpenAI API
        openai_client = OpenAI()

        # Set the thread ID
        # This is used to identify the conversation thread / session
        # Right now, we are using a fixed thread ID for testing purposes
        thread_id = "thread_AYA7B1cxHaA1Pbl08Gt5H62M"
    else:
        # Raise an error if the sector is invalid
        raise ValueError("Invalid sector provided")


# Function to initialize the database
def initialize_database(sector):
    global db, client, retriever, compression_retriever

    # Initialize the Chroma database with the sector guide
    # The persist_directory is where the database is stored
    # The collection_name is the name of the collection (guide) in the database
    # The embedding_function is the function used to embed the documents in the database
    # It needs to be the same as the one used to embed the query
    db = Chroma(persist_directory="./chroma_db", collection_name=f"{sector}_guide",
                embedding_function=OpenAIEmbeddings())

    # Initialize the retriever with the database for vector search
    # mmr is the search type, and k is the number of results to return
    # According to medium, MMR gives us the flexibility on retrieving diverse documents
    # which helps in removing redundant information which is usually encountered in a similarity based search.
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3})

    # Initialize the compressor with the top 3 results
    # FlashrankRerank is a document compressor that uses FlashRank to rerank the documents
    # It aids in selecting the most relevant documents
    compressor = FlashrankRerank(top_n=3)

    # Initialize the compression retriever with the retriever and compressor
    compression_retriever = ContextualCompressionRetriever(base_retriever=retriever, base_compressor=compressor)


# Function to create a new thread
def create_thread():
    # Initialize the OpenAI client
    openai_client = OpenAI()
    # Create a new thread
    thread = openai_client.beta.threads.create()

    # Get the thread ID
    thread_id = thread.id

    # Write the thread ID to a file
    # This is for testing purposes
    with open("thread.txt", "w") as file:
        file.write(thread_id)

    return thread_id


# Function to delete a thread
def delete_thread(t_id):
    # Initialize the OpenAI client
    openai_client = OpenAI()
    # Delete the thread with the given ID
    openai_client.beta.threads.delete(thread_id=t_id)


# Function to create a system prompt
def create_system_prompt(sector):
    # Create a system prompt for the LLM to restructure the user question
    return f"""The user is posing as a person in the {sector} sector who is a taxpayer. So the person in the {sector} 
    sector, in the user question should be substituted for taxpayer in the passed-in context
    Reply using first person directly to the user. Remember the user doesn't know the context of the chat history."""


# Function to contextualize the query for the retriever
# This function enables the assistant to understand the context of the user query
# through query transformation, a technique which is essential for accurate responses
def contextualize_query_for_retriever(query, chat_history):
    # Add the user question to the chat history
    local_chat_history = chat_history + [("assistant", contextualize_q_system_prompt)] + [("user", query)]

    # Create a chat prompt from the chat history
    chat_prompt = ChatPromptTemplate.from_messages(local_chat_history)

    print("Chat Prompt: ", chat_prompt)

    # Initialize the language model
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    # Create a chain from the chat prompt, language model, and output parser
    # Chaining is a technique used to combine multiple functions into a single function
    # It uses the input of one function as the output of the previous function
    chain = chat_prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))

    # Invoke the chain with the query
    llm_response = chain.invoke({})

    print("LLM Response: ", llm_response)
    return llm_response[0]



# Function to build the chat history
def build_chat_history(query, response):
    # Add the user query and assistant response to the chat history
    chat_history.append(('user', query))
    chat_history.append(('assistant', response))
    print("\nChat History: ", chat_history)


# Function to chat with the assistant
def chat_with_assistant(query):
    global db, thread_id, sector, compression_retriever, retriever, openai_client

    print("##  Sector: ## ", sector)
    # Raise an error if the database has not been set
    if db is None:
        raise ValueError("Database has not been set. Please set the sector first.")

    # Contextualize the query if there is more than one message in the chat history
    # This is done to provide context to the assistant for accurate responses
    # For example, if the user asks a follow-up question, the context of the previous questions is important
    if len(chat_history) > 1:
        contextualized_query = contextualize_query_for_retriever(query, chat_history)
    else:
        contextualized_query = query

    # Get the relevant documents for the contextualized query
    reranked_results = compression_retriever.get_relevant_documents(contextualized_query)

    for result in reranked_results:
        print("Result", result, "\n")

    # Create the user message with the reranked results and query
    user_message = f"""
    <context>
    {reranked_results}
    </context>

    <question>
    {query}
    </question>
    """
    print("User message: ", user_message)

    # Create a message in the thread with the system prompt
    openai_client.beta.threads.messages.create(
        thread_id=thread_id,
        role="assistant",
        content=create_system_prompt(sector),
    )

    # Create a message in the thread with the user message
    openai_client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_message
    )

    # Stream the thread runs
    with openai_client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=os.environ.get("ASSISTANT_ID"),
            temperature=0.2,
            event_handler=EventHandler()
    ) as stream:
        # Wait until the stream is done
        stream.until_done()
        # Get the last message in the thread
        messages = openai_client.beta.threads.messages.list(thread_id=thread_id, limit=1, order="desc")
        # Parse the response JSON
        msg_json = json.loads(messages.to_json())
        # Check if the last message is from the assistant
        if msg_json["data"][0]["role"] == "assistant":
            # Get the assistant's response
            response = msg_json["data"][0]["content"][0]["text"]["value"]
            # Build the chat history with the query and response
            build_chat_history(query, response)
            return response
        else:
            return "Assistant response not found"
