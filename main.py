import os

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore

if __name__ == '__main__':
    load_dotenv()

    loader = TextLoader("./vector-blog.txt")
    document = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    texts = splitter.split_documents(document)
    embeddings = OpenAIEmbeddings()

    docsearch = PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("PINECONE_INDEX_NAME"))

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever()
    )

    query = "What is a vector DB? Give me a 15 word answer for a beginner."
    result = qa({ "query": query})

    print(result)
