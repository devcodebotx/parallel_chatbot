from uuid import uuid4
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny, PointStruct, VectorParams, Distance
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Qdrant
from langchain_qdrant import Qdrant
from langchain_community.vectorstores.qdrant import Qdrant
from time import time

from fastapi import FastAPI, Request
from models.query_model import Query
from models.journal_model import Journal
from middlewares.cors import setup_cors
from datetime import datetime


load_dotenv()

app = FastAPI()

setup_cors(app)

COLLECTION_NAME = os.getenv("COLLECTION_NAME")

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    prefer_grpc=False,
    timeout=60
)

collection_exists = qdrant.collection_exists(COLLECTION_NAME)
if not COLLECTION_NAME:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE
        ),
    )

qdrant.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="type",
    field_schema="keyword",
)

qdrant.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="userId",
    field_schema="keyword",
)
qdrant.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="answer",
    field_schema="keyword",
)

# Embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api=os.getenv("GOOGLE_API_KEY")
)

# LangChain vector store
vectorstore = Qdrant(
    client=qdrant,
    collection_name=COLLECTION_NAME,
    embeddings=embedding_model,
    content_payload_key="text",
    metadata_payload_key="metadata",
)


prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are the parallel version of the user, like a clone. Don't tell the user you're them — just speak directly as if you are them.

Use the following metadata extracted from chat logs to answer like you're the user themself:
{context}

Now, respond to this question naturally in the user's voice:
{question}

Remember:
- Use metadata of this user and their different types of entries (chat, journal, initial)
- Use user's name from the metadata (e.g., username)
- Mention their location if relevant
- Talk about their questions and answers from past if it helps
- Never say "you", instead say "I" as if you're the user

Important Instructions:
- Only use the context above if it directly helps answer the question.
- If the question is factual or general(e.g., politics, news, science), do not include personal background.
- Speak naturally and directly, using "I" instead of "you".
- Mention user's personal details only if they are clearly relevant.
- Do not overexplain or go off-topic.
"""
)


llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

print(qdrant.get_collections())


@app.post("/ask")
def ask_parallel(query: Query):

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "filter": Filter(
                must=[
                    FieldCondition(
                        key="type",
                        match=MatchAny(any=["journal", "initial"])
                    ),

                    FieldCondition(
                        key="userId",
                        match=MatchValue(
                            value=query.user_id)
                    ),
                ]
            ),
            "k": 40
        }
    )

    # Create QA chain with prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        input_key="question"
    )

    retrieved_docs = retriever.get_relevant_documents(query.question)
    print("Retrieved Docs:", retrieved_docs)

    if not retrieved_docs:
        print("I'm you, I don't have anything to say about that")

    response = qa_chain.invoke(query.question)
    print("Response:", response["result"])

    question_embedding = embedding_model.embed_query(query.question)
    answer_embedding = embedding_model.embed_query(response["result"])

    question_point = PointStruct(
        id=str(uuid4()),
        vector=question_embedding,
        payload={
            "type": "chat",
            "question": "question",
            "userId": query.user_id,
            "text": query.question,
            "timestamp": int(time() * 1000),
            "name": query.name,
            "location": query.location
        }
    )

    answer_point = PointStruct(
        id=str(uuid4()),
        vector=answer_embedding,
        payload={
            "type": "chat",
            "answer": "answer",
            "userId": query.user_id,
            "text": response["result"],
            "timestamp": int(time() * 1000),
            "name": query.name,
            "location": query.location
        }
    )

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[question_point, answer_point]
    )
    count = qdrant.count(
        collection_name=COLLECTION_NAME,
        exact=True
    )
    print(f"Total points in Qdrant: {count.count}")

    return {"Response": response["result"]}


@app.post("/journal_entry")
def ask_parallel(journal: Journal):

    print("journal is:", journal.data)

    journal_embedding = embedding_model.embed_query(journal.data)

    journal_point = PointStruct(
        id=str(uuid4()),
        vector=journal_embedding,
        payload={
            "userId": journal.user_id,
            "type": "journal",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "text": journal.data,
            "timestamp": int(time() * 1000),
        }
    )

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[journal_point]
    )
    count = qdrant.count(
        collection_name=COLLECTION_NAME,
        exact=True
    )
    print(f"Total points in Qdrant: {count.count}")

    return {"journal": journal.data}
