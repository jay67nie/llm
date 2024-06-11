import os

from dotenv import load_dotenv
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50, separator="\n\n\n")

# with open("text/construction.txt", "r", encoding='utf-8') as f:
#     construction_txt = f.read()
#
# text_chunks = text_splitter.create_documents([construction_txt])
#
# Chroma.from_documents(documents=text_chunks, embedding=OpenAIEmbeddings(), collection_name="const_guide",
#                       persist_directory="./chroma_db")
#
# print("Done writing construction guide to database. Collection name: const_guide")

# with open("text/agric.txt", "r", encoding='utf-8') as f:
#     agric_text = f.read()
#
# text_chunks = text_splitter.create_documents([agric_text])
#
# Chroma.from_documents(documents=text_chunks, embedding=OpenAIEmbeddings(), collection_name="agric_guide",
#                       persist_directory="./chroma_db")
#
# print("Done writing agriculture guide to database. Collection name: agric_guide")
#
# with open("text/education.txt", "r", encoding='utf-8') as f:
#     education_txt = f.read()
#
# text_chunks = text_splitter.create_documents([education_txt])
#
# Chroma.from_documents(documents=text_chunks, embedding=OpenAIEmbeddings(), collection_name="education_guide",
#                       persist_directory="./chroma_db")
#
# print("Done writing education guide to database. Collection name: education_guide")
#
# with open("text/health.txt", "r", encoding='utf-8') as f:
#     health_txt = f.read()
#
# text_chunks = text_splitter.create_documents([health_txt])
#
# Chroma.from_documents(documents=text_chunks, embedding=OpenAIEmbeddings(), collection_name="health_guide",
#                       persist_directory="./chroma_db")
#
# print("Done writing health guide to database. Collection name: health_guide")
#
# with open("text/manufacturing.txt", "r", encoding='utf-8') as f:
#     manufacturing_txt = f.read()
#
# text_chunks = text_splitter.create_documents([manufacturing_txt])
#
# Chroma.from_documents(documents=text_chunks, embedding=OpenAIEmbeddings(), collection_name="manufacturing_guide",
#                       persist_directory="./chroma_db")
#
# print("Done writing manufacturing guide to database. Collection name: manufacturing_guide")
#
# with open("text/wholesale_and_retail.txt", "r", encoding='utf-8') as f:
#     wholesale_retail_txt = f.read()
#
# text_chunks = text_splitter.create_documents([wholesale_retail_txt])
#
# Chroma.from_documents(documents=text_chunks, embedding=OpenAIEmbeddings(), collection_name="wholesale_retail_guide",
#                       persist_directory="./chroma_db")
#
# print("Done writing wholesale and retail guide to database. Collection name: wholesale_retail_guide")
#
# with open("text/hotel_and_accommodation.txt", "r", encoding='utf-8') as f:
#     hotel_accommodation_txt = f.read()
#
# text_chunks = text_splitter.create_documents([hotel_accommodation_txt])
#
# Chroma.from_documents(documents=text_chunks, embedding=OpenAIEmbeddings(), collection_name="hotel_accommodation_guide",
#                       persist_directory="./chroma_db")
#
# print("Done writing hotel and accommodation guide to database. Collection name: hotel_accommodation_guide")
