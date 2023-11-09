import streamlit as st
import os

from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient

HF_TOKEN = os.getenv("HF_API_KEY")
QD_TOKEN = os.getenv("QD_API_KEY")
QD_URL = os.getenv("QD_URL")

with st.sidebar:
    llm_model = st.selectbox("Choose LLM from the list",options=["HuggingFaceH4/zephyr-7b-beta","HuggingFaceH4/zephyr-7b-alpha"])#,"meta-llama/Llama-2-7b-hf","meta-llama/Llama-2-13b-hf"])
    embedding_model = st.selectbox("Choose Embedder from the list",options=["BAAI/bge-large-en-v1.5"])#,"hkunlp/instructor-large"])
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, step=0.2, value=1.0)
    max_new_tokens = st.slider("Max New Tokens", min_value=50, max_value=1000, step=50, value=300)
    search_type = st.selectbox("Choose Search Type",options=["mmr","similarity","similarity_score_threshold"])
    fetch_k = st.slider("Fetch K", min_value=10, max_value=100, step=10, value=30)
    similarity_score_threshold = st.slider("Similarity Score Threshold", min_value=0.0, max_value=1.0, step=0.1, value=0.8)
    k = st.slider("Num Documents to retrieve", min_value=1, max_value=10, step=1, value=4)  

st.title("Paralegal - Chat with German Laws")

llm = HuggingFaceHub(
        repo_id=llm_model, huggingfacehub_api_token=HF_TOKEN, model_kwargs={"temperature": temperature, "max_new_tokens": max_new_tokens},
    )

@st.cache_resource(ttl="1h")
def configure_retriever(search_type,search_kwargs):

    embed_model = HuggingFaceInferenceAPIEmbeddings(api_key=HF_TOKEN,model_name=embedding_model)

    client = QdrantClient(url=QD_URL, api_key=QD_TOKEN, prefer_grpc=True)
    vectorstore = Qdrant(client=client,collection_name='paralegal_en',vector_name='bge-large-content',
                         embeddings=embed_model)

    retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    return retriever

search_kwargs = {'k': k}
if search_type == "similarity_score_threshold":
    search_kwargs.update({'similarity_score_threshold': similarity_score_threshold})
elif search_type == "mmr":
    search_kwargs.update({'fetch_k': fetch_k})

retriever = configure_retriever(search_type,search_kwargs)

@st.cache_resource(ttl="1h")
def configure_prompt():
    # Prepare Prompt
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't find the answer in the context, do not answer the question.

    Context: {context}

    Question: {question}
    Answer: """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    return PROMPT

PROMPT = configure_prompt()

chain_type_kwargs = {"prompt": PROMPT}
qa = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

query = st.text_input("Ask a question", value="what is german basic law?")
output = qa(query,return_only_outputs=True)

st.divider()
st.subheader("Answer")
st.write(output['answer'])

st.divider()
st.subheader("Source")

for x in output['sources'].split(','):
    href = f"https://www.gesetze-im-internet.de/englisch_{x.strip().split('-')[0]}/"
    st.write(f"[{x.strip()}]({href})")
