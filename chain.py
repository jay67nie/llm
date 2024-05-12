import os

import chromadb
from langchain.chains.llm import LLMChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, TextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticToolsParser
from langchain_core.output_parsers import StrOutputParser

with open("text/construction.txt", "r", encoding='utf-8') as f:
    construction_txt = f.read()

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, separators=["\n\n", "\n\n\n",
#                                                                                               "\n\n\n\n"])
#
# text_chunks = text_splitter.create_documents([construction_txt])

# for text in text_chunks:
#     print(text)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

text_chunks = text_splitter.create_documents([construction_txt])

for text in text_chunks:
    print(text)

## Retrieve from an existing collection
client = chromadb.PersistentClient(path="./chroma_db")

client.delete_collection("const_guide")

# db = Chroma(client=client, collection_name="const_guide", embedding_function=OpenAIEmbeddings())

#####This can cause duplicates in the database. Run only once

## TODO: Separate creation of collection into a different py file
db = Chroma.from_documents(documents=text_chunks, embedding=OpenAIEmbeddings(),
                           persist_directory="./chroma_db", collection_name="const_guide")


retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10})

query = input("Enter a query: ")

# results = retriever.get_relevant_documents(query)
#
# print("Raw RAG --------------------------------")
# print(len(results))
# for result in results:
#     print(result)
print("RAG with Flashrank ---------------------")
compressor = FlashrankRerank(top_n=10)
compression_retriever = ContextualCompressionRetriever(base_retriever=retriever, base_compressor=compressor)

reranked_results = compression_retriever.get_relevant_documents(query)

print("Reranked RAG ---------------------------", reranked_results)

for result in reranked_results:
    print(result)
system = """
        Below is the context of some information about the agricultural sector taxes from the
        Uganda Revenue Authority documents. You are an expert in tax law and you are to give an
        answer to the question solely based on the information provided in the context.
        If you cannot find the answer in the context, you should return 'I don't have the answer to that, unfortunately'

        <context>
        {context}
        </context>

        """

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("user", "{question}")

    ]
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

chain = chat_prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))

llm_response = chain.invoke({"question": query, "context": reranked_results})

print("LLM Response: ", llm_response)
