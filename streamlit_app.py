import streamlit as st
from local_and_global_search import local_search, global_retriever
from microsoft_to_neo4j import DB_CONFIG
from neo4j import GraphDatabase
from chroma_db import GenericVectorStore, reduce_chain
import os
import asyncio
import time

# Set Streamlit to wide mode
st.set_page_config(layout="wide")

# Load environment variables
def load_env_vars():
    with open(".env", "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            os.environ[key] = value

load_env_vars()

# Neo4j connection
url = DB_CONFIG["url"]
username = DB_CONFIG["username"]
password = DB_CONFIG["password"]

# Initialize GenericVectorStore
vector_store = GenericVectorStore()

def get_graph_data():
    driver = GraphDatabase.driver(url, auth=(username, password))
    with driver.session() as session:
        result = session.run("""
        MATCH (n)-[r]->(m)
        RETURN n.name AS source, m.name AS target, type(r) AS relationship
        LIMIT 100
        """)
        return [(record["source"], record["target"], record["relationship"]) for record in result]

st.title("Knowledge Graph and Vector DB Explorer")

# Query interface
st.header("Query the Knowledge Graph and Vector DB")
query = st.text_input("Enter your query:")
level = 1

async def timed_local_search(query):
    start_time = time.time()
    graph_results = local_search(query)
    graph_report_data = "\n\n".join([f"Result {i+1}: {result}" for i, result in enumerate(graph_results)])
    processed_graph_results = reduce_chain.invoke({
        "report_data": graph_report_data,
        "question": query,
    })
    end_time = time.time()
    return processed_graph_results, end_time - start_time

async def timed_vector_search(query):
    start_time = time.time()
    vector_results = vector_store.query(query)
    end_time = time.time()
    return vector_results, end_time - start_time

async def timed_global_search(query, level):
    start_time = time.time()
    graph_result = await global_retriever(query, level)
    end_time = time.time()
    return graph_result, end_time - start_time

if st.button("Search"):
    col1, col2, col3 = st.columns(3)
    
    async def run_searches():
        local_task = asyncio.create_task(timed_local_search(query))
        vector_task = asyncio.create_task(timed_vector_search(query))
        global_task = asyncio.create_task(timed_global_search(query, level))
        
        return await asyncio.gather(local_task, vector_task, global_task)
    
    results = asyncio.run(run_searches())
    
    with col1:
        st.subheader("Local Search Results")
        local_results, local_time = results[0]
        st.write(local_results)
        st.write(f"Time taken: {local_time:.2f} seconds")
    
    with col2:
        st.subheader("Vector DB Results")
        vector_results, vector_time = results[1]
        st.write(vector_results)
        st.write(f"Time taken: {vector_time:.2f} seconds")
    
    with col3:
        st.subheader("Global Search Results")
        global_results, global_time = results[2]
        st.write(global_results)
        st.write(f"Time taken: {global_time:.2f} seconds")

st.sidebar.header("About")
st.sidebar.info("This app allows you to explore a knowledge graph stored in a Neo4j database and a vector database, querying both using local and global search functions for comparison.")
