from dotenv import load_dotenv
import os
from llama_index import SimpleDirectoryReader
from llama_index import node_parser
from llama_index.llms import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import download_loader,ServiceContext, VectorStoreIndex,StorageContext
from llama_index.vector_stores import PineconeVectorStore
from llama_index.node_parser import SimpleNodeParser
import pinecone

load_dotenv()
pinecone.init(api_key=os.environ['PINEECONE_API_KEY'],Environment =['PINECONE_ENVIRONMENT'])
if __name__ == '__main__':
    print('Goining to ingest pinecone documentation')
    #this will remove the html tag and only clean data will be availabe
    UnstructuredReader = download_loader("UnstructuredReader")
    dir_reader = SimpleDirectoryReader(
        input_dir="./llamaindex-docs",
        file_extractor={".htm1":UnstructuredReader()}
    )

    documents = dir_reader.load_data()
    node_parser = SimpleNodeParser.from_defaults(chunk_size=200,chunk_overlap=20)
    
    #nodes = node_parser.get_nodes_from_documents(documents=documents)

    llm = OpenAI(model='gpt-3.5 turbo', temperature=0)
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model,node_parser=node_parser)

    # pinecone index name 
    index_name = "documentation-helper"
    pinecone_index = pinecone.index(index_name=index_name
                                    )
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    Storage_Context = StorageContext.from_defaults(vector_store=vector_store)
    

    index = VectorStoreIndex.from_documents(documents=documents,
                                            service_context=service_context,
                                            storage_context=Storage_Context,
                                            show_progress=True)
    
    print('finish ingesting...')