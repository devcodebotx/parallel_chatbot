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

from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from .models.query_model import Query, user_id


load_dotenv()

app = FastAPI()

# CORS (update origin if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


COLLECTION_NAME = os.getenv("COLLECTION_NAME")
# USER_ID = os.getenv("USER_ID")

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
# userId field pe index create karo (UUID ya keyword type)
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
    # change to actual collection name
    collection_name=COLLECTION_NAME,
    embeddings=embedding_model,
    content_payload_key="text",
    metadata_payload_key="metadata",
)

print(qdrant.get_collections())

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
                        value="6b9f01ef-70df-4a83-b676-6e680ff312a8")
                ),
            ]
        ),
        "k": 20
    }
)

llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are the parallel version of the user, like a clone. Don't tell the user you're them — just speak directly as if you are them.

Use the following metadata extracted from chat logs to answer like you're the user themself:
{context}

Now, respond to this question naturally, using user's voice:
{question}

Remember:
- Use metadata of this user and their different types of entries (chat, journal, initial)
- Use user's name from the metadata (e.g., username)
- Mention their location if relevant
- Talk about their questions and answers from past if it helps
- Never say "you", instead say "I" as if you're the user
"""
)

# Create QA chain with prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    input_key="question"
)

while True:
    input_text = input("Ask To Your Parallel: ")

    retrieved_docs = retriever.get_relevant_documents(input_text)
    print("Retrieved Docs:", retrieved_docs)

    if not retrieved_docs:
        print("I'm you, I don't have anything to say about that")
        continue

    response = qa_chain.invoke(input_text)
    print(response["result"])


@app.post("/ask_parallel")
def ask_parallel(query: Query):

    # 1. Prepare the retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "filter": Filter(
                must=[
                    FieldCondition(key="type", match=MatchAny(
                        any=["journal", "initial", "chat"])),
                    FieldCondition(key="userId", match=MatchValue(
                        value=query.user_id)),
                ]
            ),
            "k": 40
        }
    )

    # 2. Get documents manually
    retrieved_docs = retriever.get_relevant_documents(query.question)
    print("Retrieved Docs:", retrieved_docs)

    # 3. Build context string
    context_parts = []
    print("Query Name is:", query.name)
    if query.name:
        context_parts.append(f"My name is {query.name}.")

    context_parts.extend([doc.page_content for doc in retrieved_docs])
    full_context = "\n".join(context_parts)

    if not retrieved_docs:
        print("I'm you, I don't have anything to say about that")
        return {"Response": "I'm you, I don't have anything to say about that"}

    # 4. Use the prompt manually via LLMChain or invoke method
    prompt_chain = chat_prompt | llm

    response = prompt_chain.invoke({
        "context": full_context,
        "question": query.question
    })

    print("Response:", response)

    # 5. Embed and store in Qdrant
    question_embedding = embedding_model.embed_query(query.question)
    answer_embedding = embedding_model.embed_query(response)

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
            "text": response,
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

    return {"Response": response}
