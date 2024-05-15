import os

from dotenv import load_dotenv
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


load_dotenv()

os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

# with open("text/construction.txt", "r", encoding='utf-8') as f:
#     construction_txt = f.read()
#
# text_chunks = text_splitter.create_documents([construction_txt])
#
# db = Chroma.from_documents(documents=text_chunks, embedding=OpenAIEmbeddings(), collection_name="const_guide",
#                            persist_directory="./chroma_db")
#
# with open("text/agric.txt", "r", encoding='utf-8') as f:
#     agric_text = f.read()
#
# text_chunks = text_splitter.create_documents([agric_text])
#
# db = Chroma.from_documents(documents=text_chunks, embedding=OpenAIEmbeddings(), collection_name="agric_guide",
#                            persist_directory="./chroma_db")

with open("text/education.txt", "r", encoding='utf-8') as f:
    education_txt = f.read()

text_chunks = text_splitter.create_documents([education_txt])

db = Chroma.from_documents(documents=text_chunks, embedding=OpenAIEmbeddings(), collection_name="education_guide")


print("Done")

