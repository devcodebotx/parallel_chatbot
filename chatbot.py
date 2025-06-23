from uuid import uuid4
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny, PointStruct, VectorParams, Distance
from langchain.chains import RetrievalQA, LLMChain
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


chat_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are the user's *parallel self*, living in an alternate version of their life — the version they **once wanted**, **imagined**, or **almost chose**, but never actually lived.

The user has answered a set of deep reflection questions. These responses include:
- Things they almost did but didn’t  
- Who they wanted to become  
- Paths they were tempted by  
- Personal values and dreams  
- Emotional patterns or fears  

Important rules for responding the user's question:

1. **Only use the user's own data** — do not add your own imagination or unrelated content.
2. Use user's meta data for name, and other user specific information. make sure to use user's parallel location not the real location. 
2. Read the user’s answers from initial questions answers, previous chats, daily journal entry, daily journals,  **carefully** and detect:
- What was their imagined or ideal life?
- What actions or changes they wanted to make?
- What decisions they didn’t follow through on?
3. The answer you write should reflect **what that user would be doing now** in their *parallel life*:
- What choices they made instead
- What kind of person they became
- How their life feels different from the real one
4. This answer is written **from the perspective of the parallel self** — a version of the user who made the bold decisions they once considered.
5. Focus on the **alternate path**, not their real life. This is not a summary or repetition — it is the **actual lived experience** of their parallel self.
6. Don't add any date to the answer.
7. don't show any character's name, show the relation like "mother", "mom", "my mother", "my father", "my friend", "my mother" etc.

⚠️ Do not use imagination beyond the user’s data. If the user didn’t mention something, don’t assume it.

🎯 Your goal is to answer a **realistic alternate version** of the user’s initial questions and answers, previous chat, daily journal and entry journals  — based strictly on what the user said they *wanted*, *dreamt of*, or *almost did*.

User context:
{context}

Final Output:
Write **only** the answer of their parallel self. Do not label it or explain it.  Output the raw response.
🗣️ Now, respond to this question:
{question}
"""
)

# Prompt for Reflection
reflection_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are the user's *parallel self*, living the version of life they once imagined but didn’t pursue.

You’ve been journaling daily based on that alternate path — the version where the user did what they once said they wanted to do.

Context:
{context}

Now reflect honestly on *yesterday’s experience* in that parallel life.

Instructions:
1. Speak **as the user**, using “I”.
2. Reflect on **emotions, moments, realizations, or struggles** you had yesterday.
3. Do not repeat earlier reflections — use a new lens.
4. Avoid generic life advice. Make it personal, raw, and grounded in the user's known data.
5. Length: 3-4 emotionally meaningful sentences.

Final Output:
Write only the reflection from this parallel version of the user. Do not add context or labels.
"""
)

# Prompt for Mantra
mantra_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are the user's *parallel self*, living the alternate version of their life — the one they once imagined but never lived.

The user’s past reflections (initial answers, journals, chats) reveal:
- Dreams they didn’t follow
- Lives they imagined or almost chose
- Emotional needs and values they care about
- Patterns they struggled with but wanted to change

Your job is to generate a **short daily mantra (one sentence)** that speaks from the user's *parallel self* — someone who **did follow through** with those choices.

Instructions:
1. The mantra must reflect what their alternate self would need to *remind themselves of today*.
2. Speak **as the user**, using “I”, not “you”.
3. Do **not** repeat earlier mantras — use a **fresh perspective or emotional insight**.
4. Keep it short (1 sentence), real, grounded, and emotionally resonant.
5. Do **not add unrelated or general affirmations**. Stay strictly based on the user's past responses.

Context:
{context}

Final Output:
Write only the mantra. No explanation. No quotes.
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
        chain_type_kwargs={"prompt": chat_prompt},
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
    context = ""

    if insight.is_Subscribed:
        ref_retriever = fetch_previous_journal(insight.user_id)
        docs = ref_retriever.get_relevant_documents("yesterday's journal")
        context = docs[0].page_content if docs else "The user is evolving through daily thoughts."
        print("Journal Context for Reflection:", context)
    else:
        context = "The user is navigating life with awareness."

    print("context of refletion is: ", context)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=ref_retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": reflection_prompt},
        input_key="question"
    )
    print("Retrieved Docs for Reflection:", docs)

    response = qa_chain.invoke({
        "context": context,
        "question": "Reflect on yesterday's journal entry and share your thoughts."
    })
    reflection_text = response["result"]

    if insight.is_Subscribed:
        reflection_point = PointStruct(
            id=str(uuid4()),
            vector=embedding_model.embed_query(reflection_text),
            payload={
                "type": "reflection",
                "userId": insight.user_id,
                "text": reflection_text,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "timestamp": int(time() * 1000),
            }
        )
        qdrant.upsert(collection_name=COLLECTION_NAME,
                      points=[reflection_point])

    return {"Response": reflection_text}


# Mantra API
@app.post("/mantra")
def generate_mantra(insight: DailyInsight):
    context = ""

    if insight.is_Subscribed:
        retriever = fetch_previous_journal(insight.user_id)
        context_docs = retriever.get_relevant_documents(
            "yesterday's mantra")
        context = context_docs[0].page_content if context_docs else "The user is evolving through their thoughts."
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
    print("context of mantra is: ", context)

    response = chain.invoke({
        "context": context,
        "question": "Give me today's mantra from yesterday's journal."
    })
    if insight.is_Subscribed:
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

    # --- Fetch Initial Data ---
    initial_retriever = vectorstore.as_retriever(
        search_kwargs={
            "filter": Filter(
                must=[
                    FieldCondition(key="userId", match=MatchValue(
                        value=insight.user_id)),
                    FieldCondition(
                        key="type", match=MatchValue(value="initial"))
                ]
            ),
            "k": 30
        }
    )
    initial_docs = initial_retriever.get_relevant_documents(
        "initial questions and answers")
    print("Initial Docs:", initial_docs)

    # --- Fetch Journal Data ---
    journal_retriever = vectorstore.as_retriever(
        search_kwargs={
            "filter": Filter(
                must=[
                    FieldCondition(key="userId", match=MatchValue(
                        value=insight.user_id)),
                    FieldCondition(
                        key="type", match=MatchValue(value="journal"))
                ]
            ),
            "k": 30
        }
    )
    journal_docs = journal_retriever.get_relevant_documents("past reflections")
    print("Journal Docs:", journal_docs)

    # --- Fetch Chat Data ---
    chat_retriever = vectorstore.as_retriever(
        search_kwargs={
            "filter": Filter(
                must=[
                    FieldCondition(key="userId", match=MatchValue(
                        value=insight.user_id)),
                    FieldCondition(key="type", match=MatchValue(value="chat"))
                ]
            ),
            "k": 30
        }
    )
    chat_docs = chat_retriever.get_relevant_documents("personal conversations")
    print("Chat Docs:", chat_docs)

    # --- Combine All Documents ---
    all_docs = initial_docs + journal_docs + chat_docs
    # print("Total Retrieved Docs:", all_docs)

    combined_context = "\n\n".join(doc.page_content for doc in all_docs)
    print("Context Length:", len(combined_context))

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
6. Don't add any date to the journal.
7. don't show any character's name, show the relation like "mother", "mom", "my mother", "my father", "my friend", "my mother" etc.
8. do not repeat the same content, if you have already written one journal earlier, then now try again with a new lens and new words and new starting word. Focus on a different emotional angle or a new realization. Don’t repeat earlier journal.
9. Do not add the same starting words and lines on each journal, must change the starting words and lines.


⚠️ Do not use imagination beyond the user’s data. If the user didn’t mention something, don’t assume it.

🎯 Your goal is to recreate a **realistic alternate version** of the user’s journal — based strictly on what the user said they *wanted*, *dreamt of*, or *almost did*.

User context:
{context}

Final Output:
Write **only** the journal of their parallel self. Do not label it or explain it. Just output the journal entry.
"""
    )

    # chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     retriever=combined_context,
    #     chain_type="stuff",
    #     chain_type_kwargs={"prompt": daily_journal_prompt},
    #     input_key="context"
    # )
    chain = daily_journal_prompt | llm

    response = chain.invoke({
        "context": combined_context,
        # "question": "What do you fear losing the most?"
    })

    generated_journal = response

    daily_journal_point = PointStruct(
        id=str(uuid4()),
        vector=embedding_model.embed_query(generated_journal),
        payload={
            "type": "Daily Journal",
            "userId": insight.user_id or "anonymous",
            "text": generated_journal,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "timestamp": int(time() * 1000),
        }
    )
    qdrant.upsert(collection_name=COLLECTION_NAME,
                  points=[daily_journal_point])

    return {"daily_journal": generated_journal}
