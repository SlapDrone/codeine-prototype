
import pickle
import os
from pathlib import Path


import langchain.schema
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv("../.env")
from llama_index import download_loader, GPTVectorStoreIndex
from llama_index.readers import GithubRepositoryReader
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index import (
    download_loader,
    GPTVectorStoreIndex,
    ServiceContext, 
    StorageContext,
    load_index_from_storage
)
from llama_index.readers.github_readers.github_api_client import GithubClient
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.chat_engine import ReActChatEngine

model = "gpt-3.5-turbo"#"gpt-4"

docs = None
if os.path.exists("docs.pkl"):
    with open("docs.pkl", "rb") as f:
        docs = pickle.load(f)

if docs is None:
    loader = GithubRepositoryReader(
        owner =                  "SlapDrone",
        repo =                   "codeine-prototype",
        github_token=os.getenv("GITHUB_TOKEN"),
        ignore_directories =     ["notebooks"],
        ignore_file_extensions = [".js", ".css", ".ico"],
        verbose =                True,
        concurrent_requests =    10,
    )

    docs = loader.load_data(branch="main")
    for d in docs:
        print(d)
    with open("docs.pkl", "wb") as f:
        pickle.dump(docs, f)

embeddings_storage = Path(f"./storage/codeine-prototype")

if embeddings_storage.exists():
    # load document/index/vector store from storage
    storage_context = StorageContext.from_defaults(persist_dir=embeddings_storage)
    cur_index = load_index_from_storage(storage_context=storage_context)
    chunk_size_limit = 768
else:
    # generate document/index/vector store from documents, and store
    # we need an embedding model, hence we create a service context
    # the chunk_size_limit internally propagates to the textsplitter (which in turn goes to 
    # the node parser) and limits the number of tokens in each node
    chunk_size_limit, chunk_overlap = 768, 64
    text_splitter = TokenTextSplitter(chunk_size=chunk_size_limit, chunk_overlap=chunk_overlap)
    node_parser = SimpleNodeParser(text_splitter=text_splitter)
    service_context = ServiceContext.from_defaults(
        node_parser=node_parser,
        chunk_size_limit=chunk_size_limit
    ) 
    #index_set = {}

    # just a flat index for now, could loop and create multiple, hierarchically for example
    storage_context = StorageContext.from_defaults()
    # from_documents uses the service_context's nodeparser to translate documents into nodes

    cur_index = GPTVectorStoreIndex.from_documents(
        docs,#doc_set[key]
        service_context=service_context,
        storage_context=storage_context
    )
    #index_set[key] = cur_index
    storage_context.persist(persist_dir=embeddings_storage)
    
# since we're just doing one
index_set = cur_index

from llama_index import (
    GPTVectorStoreIndex,
    ResponseSynthesizer,
    LLMPredictor
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor

#index = GPTVectorStoreIndex.from_documents(docs, storage_context=...)

# configure retriever
retriever = VectorIndexRetriever(
    index=cur_index,
    similarity_top_k=5
)

# Response synthesis service context
service_context = ServiceContext.from_defaults(
    llm_predictor=LLMPredictor(ChatOpenAI(temperature=0, model_name=model)),
    chunk_size_limit=chunk_size_limit
)

# configure response synthesizer
response_synthesizer = ResponseSynthesizer.from_args(
    service_context=service_context,
    node_postprocessors=[
        SimilarityPostprocessor(similarity_cutoff=.7)
    ],
    response_mode="compact"
)

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

def build_chat_engine(
    name=None,
    description=None,
    verbose=None, 
    query_engine=query_engine,
    service_context=service_context
):
    return ReActChatEngine.from_query_engine(
        query_engine,
        service_context=service_context,
        name="Codeine Source Code Search",
        description=(
            "Useful for when you should answer questions about the codeine source code."
        ),
        verbose=True
    )
