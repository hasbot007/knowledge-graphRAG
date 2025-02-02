import textwrap
from langchain_groq import ChatGroq
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from dotenv import load_dotenv
import streamlit as st
import os

# Load environment variables from .env file
load_dotenv()
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Connect to your Neo4j instance
graph = Neo4jGraph(
    url="neo4j+s://96646943.databases.neo4j.io",
    username="neo4j",
    password="broxNcCELLvtXM_A4CooYMorCh_Q-Q79dlER5Tax2gI"
)

# Initialize an LLM from Groq (e.g., DeepSeek)


chain=GraphCypherQAChain.from_llm(llm=llm,graph=graph,verbose=True,allow_dangerous_requests = True)

def get_llm_response(text):
  response = chain.invoke({"query": text})
  return response


def create_streamlit_app():
    st.set_page_config(layout="wide", page_title="Knowledge Graph")

    st.title("Movie GraphRAG")

    user_input = st.text_area("Enter your text prompt:")

    if st.button("Generate Response"):
        if user_input:
            with st.spinner("Generating response..."):
                response = get_llm_response(user_input)
            st.write("**LLM Output:**")
            st.write(response)
        else:
            st.warning("Please enter a prompt.")
        
if __name__ == "__main__":
    create_streamlit_app()