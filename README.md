# Documentation_Helper using LlamaIndex
A LLM application that will allow user to chat with LlamaINdex official documents and get the relevent content with the accuracy score.

# Demo
![banner](https://github.com/Mahimajain25/Documentation_Helper/assets/96101074/5fb60428-5e34-4ad1-83b3-81406578410e)

# Project Overview
- **Dataset :** <br>Download the Llama Index official document using [code](https://github.com/Mahimajain25/Documentation_Helper/blob/main/download_docs.py)
- **Data Ingestion:** <br>
Ingest the data in pinecone vector database using Llama Index Framework [Data Ingestion](https://github.com/Mahimajain25/Documentation_Helper/blob/main/ingestion.py)
- **RAG & Query** <br>
Llamaindex internally call Retrieval Augmented Generation (RAG) by calling as_query_engine and user query call will also get converted into the vector and after embbeding will get relevent content and OpenAI will pass the relevent content and user query to get an answer.
- **Model Evaluation** <br>
Devloper can identfy For each node of relevant content which have the symentic score(node.score), when score is 1 or near to 1 means good accuracy. <br>
![Document-Helper](https://github.com/Mahimajain25/Documentation_Helper/assets/96101074/d466a414-6374-45b2-8890-ab2d7e522714)
- **How to run the project ?** write streamlit run main.py in terminal.
  
# **Technologies used**
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-blue?style=for-the-badge&logo=pinecone&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-purple?style=for-the-badge&logo=pinecone&logoColor=white)
![Open_AI](https://img.shields.io/badge/OpenAI-green?style=for-the-badge&logo=pinecone&logoColor=white)


# **Tools used**
![vscode](https://img.shields.io/badge/VSCode-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
## Contact

- linkedin - https://www.linkedin.com/in/mahima-jain-41b540191/
- gmail - mahimaj25@gmail.com
