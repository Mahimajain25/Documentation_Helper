from dotenv import load_dotenv
import os

from node_postprocessor.duplicate_postprocessing import (
    DuplicateRemoverNodePostprocessor,
)

load_dotenv()
from llama_index import VectorStoreIndex
from llama_index.vector_stores import PineconeVectorStore
import pinecone
from llama_index.callbacks import LlamaDebugHandler
from llama_index.callbacks.base import CallbackManager
from llama_index import download_loader, ServiceContext
import streamlit as st
from llama_index.postprocessor import SentenceEmbeddingOptimizer
from llama_index.chat_engine.types import ChatMode


llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager(handlers=[llama_debug])
service_context = ServiceContext.from_defaults(callback_manager=callback_manager)


# used for streamlit cache such that it will use only once the pinecone connection
@st.cache_resource(show_spinner=False)
def get_index() -> PineconeVectorStore:

    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        Environments=os.environ["PINECONE_ENVIRONMENT"],
    )

    # pinecone index name
    index_name = "documentation-helper"
    pinecone_index = pinecone.index(index_name=index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    # callback help us to
    # degub logs for before and after chunking, node parsing, LLm calls, retive query
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager(handlers=[llama_debug])
    service_context = ServiceContext.from_defaults(callback_manager=callback_manager)

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store, service_context=service_context
    )

    # query = "what is LlamaIndex query engine ?"
    # query_engine = index.as_query_engine()
    # response = query_engine.query(query)
    # print(response)


index = get_index()

# we don't want open new chat again so we using session here from streamlit
if "chat_engine" not in st.session_state.keys():

    # postprocessor will remove all the sentence which are not relevent
    postprocessor = SentenceEmbeddingOptimizer(
        embed_model=service_context.embed_model,
        percentile_cutoff=0.5,
        threshold_cutoff=0.7,
    )

    # remove irrelevent text for LLM and duplicate text
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,
        verbose=True,
        postprocessor=[postprocessor, DuplicateRemoverNodePostprocessor],
    )

st.set_page_config(
    page_title="Chat with LlamaIndex doc,powered by LlamaIndex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("Chat with LlamaIndex doc 🦙")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about LlamaIndex's Open source python library ?",
        }
    ]

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for messages in st.session_state.messages:
    with st.chat_message(["role"]):
        st.write(messages["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.Chat(messages=prompt)
            # display the response
            st.write(response.response)
            # nodes will provide symentic score 1 or 0, 1 symentic close (have same meaning)
            nodes = [node for node in response.source_nodes]
            # provide all the source nodes text with symentic score
            for col, node, i in zip(st.columns(len(nodes), nodes, range(len(nodes)))):
                with col:
                    st.header(f"Source Node {i+1}: score={node.score}")
                    st.write(node.text)
            messages = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(messages)
