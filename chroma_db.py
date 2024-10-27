from chromadb import PersistentClient, EmbeddingFunction, Embeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

MODEL_NAME = 'dunzhang/stella_en_1.5B_v5'
DB_PATH = './.chroma_db'
TEXT_FILE_PATH = './book.txt'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

REDUCE_SYSTEM_PROMPT = """
You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.

Generate a response of the target length and format that responds to the user's question, summarizing all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided are ranked in descending order of importance.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should:
1. Remove all irrelevant information from the analysts' reports
2. Merge the cleaned information into a comprehensive answer
3. Provide explanations of all key points and implications appropriate for the response length and format
4. Add sections and commentary as appropriate for the length and format
5. Style the response in markdown

Preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Preserve all data references previously included in the analysts' reports, but do not mention the roles of multiple analysts in the analysis process.

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

Example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 34, 46, 64, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"
where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.
Do not include information where supporting evidence is not provided.

---Target response length and format---

multiple paragraphs

---Analyst Reports---

{report_data}
"""

reduce_prompt = ChatPromptTemplate.from_messages([
    ("system", REDUCE_SYSTEM_PROMPT),
    ("human", "{question}"),
])

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
reduce_chain = reduce_prompt | llm | StrOutputParser()

class CustomEmbeddingClass(EmbeddingFunction):
    def __init__(self, model_name):
        self.embedding_model = HuggingFaceEmbedding(model_name=MODEL_NAME)

    def __call__(self, input_texts: List[str]) -> Embeddings:
        return [self.embedding_model.get_text_embedding(text) for text in input_texts]

class GenericVectorStore:
    def __init__(self):
        db = PersistentClient(path=DB_PATH)
        custom_embedding_function = CustomEmbeddingClass(MODEL_NAME)
        self.collection = db.get_or_create_collection(name='TextChunks', embedding_function=custom_embedding_function)

        if self.collection.count() == 0:
            self._load_text_collection(TEXT_FILE_PATH)

    def _load_text_collection(self, text_file_path: str):
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)

        self.collection.add(
            documents=chunks,
            ids=[str(i) for i in range(len(chunks))],
            metadatas=[{"chunk_index": i} for i in range(len(chunks))]
        )

    def query(self, query: str, n_results: int = 8):
        results = self.collection.query(query_texts=[query], n_results=n_results)
        # Prepare the report data from the query results
        report_data = []
        for i in range(len(results['ids'][0])):  # Access the first element of 'ids'
            if results['documents'][0][i] and results['metadatas'][0][i]:
                chunk_index = results['metadatas'][0][i].get('chunk_index', i)
                report_data.append(f"Chunk {chunk_index}: {results['documents'][0][i]}")
        
        # Use the reduce chain to synthesize the final response
        final_response = reduce_chain.invoke({
            "report_data": "\n\n".join(report_data),
            "question": query,
        })
        
        return final_response