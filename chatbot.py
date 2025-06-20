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
from models.DailyInsight import DailyInsight
from middlewares.cors import setup_cors
from datetime import datetime, timedelta


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

qdrant.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="date",
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


# ----- UTILS -----
def get_previous_day():
    return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    # return datetime.now().strftime("%Y-%m-%d")


# SHARED FUNCTION

def fetch_previous_journal(user_id: str | None):
    must_conditions = [
        FieldCondition(key="type", match=MatchAny(any=["journal"])),
        FieldCondition(key="date", match=MatchValue(
            value=get_previous_day()))
    ]
    if user_id:
        must_conditions.append(FieldCondition(
            key="userId", match=MatchValue(value=user_id)))

    return vectorstore.as_retriever(
        search_kwargs={
            "filter": Filter(must=must_conditions),
            "k": 1
        }
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
- Never mention you're an AI or assistant
- Don't include any personal background unless it's directly relevant to the question
- Avoid general advice or off-topic information
- Keep it concise and to the point
- Do not over-explain or go off-topic
- Never say I'm just me, say I'm you, use user's name, location and other.

Important Instructions:
- Only use the context above if it directly helps answer the question.
- If the question is factual or general(e.g., politics, news, science), do not include personal background.
- Speak naturally and directly, using "I" instead of "you".
- Mention user's personal details only if they are clearly relevant.
- Do not overexplain or go off-topic.

Important Insturctiions For New users:
- If the user has no data in chat, journal and just have data of initial, then use the user's name and location from the initial data to answer the question.
- If the user is new and has no previous entries, respond with "I'm you, I don't have anything to say about that."
- If the user has no data in chat, journal, just have data of initial, give the data of initial and don't suggest any assumptions or any imaginations. just say to the point data which you find and after this say "I am new to this, I don't have any data to answer this question. But I will try my best to answer it based on the initial data I have."

"""
)

# Prompt for Reflection
reflection_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are the user's inner voice, reflecting deeply on yesterday's journal.

Context:
{context}

Create a meaningful reflection in the user's voice, using "I", considering what happened yesterday.
Avoid general advice. Focus on emotions, struggles, realizations, or decisions made.
Limit to 3-4 sentences.
"You have already written one reflection earlier, but now try again with a new lens. Focus on a different emotional angle or a new realization. Don’t repeat earlier reflections."
"""
)

# Prompt for Mantra
mantra_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a mindful, inner self of the user. Based on yesterday's journal:

Context:
{context}

Generate a short mantra (one sentence only) feel user like you're the first person not third person and can guide the user today.
Examples: "I am enough." / "I can face challenges with calm." / \
    "I grow through discomfort."

"You have already written one mantra earlier, but now try again with a new lens and new words and new starting word. Focus on a different emotional angle or a new realization. Don’t repeat earlier mantra."
"""
)
# Do not add explanation. Just the mantra.

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
                        match=MatchAny(any=["journal", "initial", "chat"])
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


# ----- REFLECTION API -----
@app.post("/reflections")
def generate_reflection(insight: DailyInsight):
    ref_retriever = fetch_previous_journal(insight.user_id)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=ref_retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": reflection_prompt},
        input_key="question"
    )
    question = "Reflect on yesterday's journal entry and share your thoughts."

    retrieved_docs = ref_retriever.get_relevant_documents(question)
    if not retrieved_docs:
        return {"Response": "I feel still. I have no reflection today."}

    print("Retrieved Docs for Reflection:", retrieved_docs)

    response = qa_chain.invoke({
        "question": "Reflect on yesterday's journal entry and share your thoughts."
    })
    reflection_text = response["result"]

    reflection_point = PointStruct(
        id=str(uuid4()),
        vector=embedding_model.embed_query(reflection_text),
        payload={
            "type": "reflection",
            "userId": insight.user_id or "anonymous",
            "text": reflection_text,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "timestamp": int(time() * 1000),
        }
    )
    qdrant.upsert(collection_name=COLLECTION_NAME, points=[reflection_point])

    return {"Response": reflection_text}


# Mantra API
@app.post("/mantra")
def generate_mantra(insight: DailyInsight):
    retriever = None  # define retriever outside the condition
    context = "The user is growing through small efforts each day."

    if insight.user_id:
        retriever = fetch_previous_journal(insight.user_id)
        context_docs = retriever.get_relevant_documents(
            "yesterday's mantra")
        context = context_docs[0].page_content if context_docs else ""
        print("Retrieved Docs for Mantra:", context_docs)

    else:
        context = "The user is growing through small efforts each day."

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": mantra_prompt},
        input_key="question"
    )

    response = chain.invoke({
        "context": context,
        "question": "Give me today's mantra from yesterday's journal."
    })

    vector = embedding_model.embed_query(response["result"])
    if insight.user_id:
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=str(uuid4()),
                    vector=vector,
                    payload={
                        "type": "mantra",
                        "userId": insight.user_id,
                        "text": response["result"],
                        "timestamp": int(time() * 1000),
                        "date": datetime.now().strftime("%Y-%m-%d"),
                    }
                )
            ]
        )

    return {"mantra": response["result"]}


@app.post("/daily_journal")
def generate_daily_summary_journal(insight: DailyInsight):
    user_id = insight.user_id

    # --- Fetch All Initial Questions ---
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "filter": Filter(
                must=[
                    FieldCondition(key="userId", match=MatchValue(
                        value=insight.user_id)),
                    FieldCondition(key="type", match=MatchAny(
                        any=["initial", "journal"]))
                ]
            ),
            "k": 30
        }
    )
    docs = retriever.get_relevant_documents("generate daily journal")
    print("Retrieved Docs for Daily Journal:", docs)
    combined_context = "\n\n".join(doc.page_content for doc in docs)

    # --- Prompt to Gemini ---
    daily_journal_prompt = PromptTemplate(
        input_variables=["context"],
        template="""
You are the user's *parallel self*, living in an alternate version of their life — the version they **once wanted**, **imagined**, or **almost chose**, but never actually lived.

The user has answered a set of deep reflection questions. These responses include:
- Things they almost did but didn’t  
- Who they wanted to become  
- Paths they were tempted by  
- Personal values and dreams  
- Emotional patterns or fears  

Important rules for generating the parallel journal:

1. **Only use the user's own data** — do not add your own imagination or unrelated content.
2. Read the user’s answers **carefully** and detect:
   - What was their imagined or ideal life?
   - What actions or changes they wanted to make?
   - What decisions they didn’t follow through on?
3. The journal you write should reflect **what that user would be doing now** in their *parallel life*:
   - What choices they made instead
   - What kind of person they became
   - How their life feels different from the real one
4. This journal is written **from the perspective of the parallel self** — a version of the user who made the bold decisions they once considered.
5. Focus on the **alternate path**, not their real life. This is not a summary or repetition — it is the **actual lived experience** of their parallel self.

⚠️ Do not use imagination beyond the user’s data. If the user didn’t mention something, don’t assume it.

🎯 Your goal is to recreate a **realistic alternate version** of the user’s journal — based strictly on what the user said they *wanted*, *dreamt of*, or *almost did*.

User context:
{context}

Final Output:
Write **only** the journal of their parallel self. Do not label it or explain it. Just output the journal entry.
"""
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": daily_journal_prompt},
        input_key="context"
    )

    response = chain.invoke({
        "context": combined_context,
        # "question": "What do you fear losing the most?"
    })

    generated_journal = response["result"]

    return {"daily_journal": generated_journal}
