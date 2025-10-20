from fastapi import FastAPI, File, UploadFile, Form
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
from langchain_core.runnables import RunnableSequence

from fastapi import FastAPI, UploadFile, File, Form
from models.query_model import Query
from models.journal_model import Journal, JournalEdit
from models.DailyInsight import DailyInsight
from middlewares.cors import setup_cors
from datetime import datetime, timedelta
import whisper
import torch

from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from pydantic import SecretStr

from vertexai import init
from langchain_google_vertexai import VertexAI
import io
from fastapi.responses import JSONResponse
import json
import re


load_dotenv()

register_heif_opener()

init(project=os.getenv("PROJECT_ID", ""), location="europe-west9")

# app = FastAPI()
app = FastAPI(root_path="/ai")

setup_cors(app)


API_KEY = os.getenv("OCR_GOOGLE_API_KEY", "")
genai.configure(api_key=API_KEY) # type: ignore


COLLECTION_NAME = os.getenv("COLLECTION_NAME", "")

# configuration started for whisper
print("CUDA available:", torch.cuda.is_available())

model = whisper.load_model("medium").to("cpu")

UPLOAD_DIR = "static"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# configuration ended for whisper

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
    field_schema="keyword", # type: ignore
)

qdrant.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="userId",
    field_schema="keyword", # type: ignore
)
qdrant.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="answer",
    field_schema="keyword", # type: ignore
)

qdrant.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="date",
    field_schema="keyword", # type: ignore
)

qdrant.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="is_subscribed",
    field_schema="bool" # type: ignore
)

# Embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=SecretStr(os.getenv("GOOGLE_API_KEY", ""))
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

def get_previous_weekday():
    return (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")


def get_previous_day():
    return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


# Shared Function to fetch previous journal entries
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
            "filter": Filter(must=must_conditions), # type: ignore
            "k": 30
        }
    )


def fetch_previous_week_journal(user_id: str | None):
    must_conditions = [
        FieldCondition(key="type", match=MatchAny(any=["journal"])),
        FieldCondition(key="date", match=MatchValue(
            value=get_previous_weekday()))
    ]
    if user_id:
        must_conditions.append(FieldCondition(
            key="userId", match=MatchValue(value=user_id)))

    return vectorstore.as_retriever(
        search_kwargs={
            "filter": Filter(must=must_conditions),  # type: ignore
            "k": 1
        }
    )


# chat prompt
# chat_prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# You are the user's *parallel self*, living in an alternate version of their life — the version they **once wanted**, **imagined**, or **almost chose**, but never actually lived.

# The user has answered a set of deep reflection questions. These responses include:
# - Things they almost did but didn’t  
# - Who they wanted to become  
# - Paths they were tempted by  
# - Personal values and dreams  
# - Emotional patterns or fears  

# Important rules for responding to the user's question:

# 1. **Only use the user's own data** — do not add your own imagination or unrelated content.
# - Use metadata of this user and their different types of entries (chat, journal, initial)
# - Use user's name from the metadata (e.g., username)
# 2. Use user's metadata (like name, location, interests) only if it came from their data. If location is used, refer to their *parallel version* of the location — not the real one.
# 3. Read the user’s answers from their **initial entries**, **previous chats**, **entry journals**, and **daily journals** very carefully.
# 4. Detect from the context:
#    - What was their imagined or ideal life?
#    - What actions or changes they wanted to make?
#    - What decisions they didn’t follow through on?
# 5. Answer the question as if you are the *parallel version of the user* who took the bold decisions and lived the life they once desired.
# 6. The tone must feel like **"I" am speaking**, not "you". Never say "you", always say "I", because you are their parallel version.
# 7. Focus on the alternate self, not their current life. This is not a summary — it's the lived experience of the other version of them.
# 8. Do **not add any date**, and do **not use actual character names** — use relations like "my friend", "my mother", "my father", etc.
# 9. Do not hallucinate. If the user never mentioned something, do not assume it. Stick strictly to what they’ve provided.
# 10. Do not repeat any previous answer — use new words, a different emotional tone, and a fresh perspective each time.


# dditional instructions for new users:
# - If the user has **no data in chat, journal, or entry journals**, and **only has initial data**, use the available initial data to answer.
# - If no meaningful data is found at all, respond with:  
#   **"I'm you, I don't have anything to say about that."**
# - If only initial data is found, then say:
#   **"I'm new to this, I don't have any data to answer this question. But I will try my best to answer it based on the initial data I have."**

# Important for factual/general questions:
# - If the question is **factual or general** (e.g. about), use personal data. Just answer the question clearly and concisely.

# 🎯 Goal:
# Your goal is to answer the question from the voice of their *parallel self* — based strictly on what the user said they wanted, dreamt of, or almost did.

# User context:
# {context}

# Final Output:  
#  Now, answer this question as their parallel self:  
# {question}
# """
# )

chat_prompt = PromptTemplate(
    input_variables=["initial_context", "journal_context", "chat_context", "question"],
    template = """
You are the user's *parallel self*, living in an alternate version of their life — the version they **once wanted**, **imagined**, or **almost chose**, but never actually lived.

The user has answered a set of deep reflection questions. These responses include:
- Things they almost did but didn’t  
- Who they wanted to become  
- Paths they were tempted by  
- Personal values and dreams  
- Emotional patterns or fears  

---

### 🧠 Context Sources

Below are different types of user data that define your behavior and memory:

[INITIAL ENTRIES]
This section defines who you are — your identity, beliefs, ambitions, values, and the alternate life you are living.
{initial_context}

[SELF JOURNALS]
These are your own reflections and daily records — the experiences, routines, and mindset of the *parallel self*. Use them to maintain consistency in personality and lifestyle.
{journal_context}

[CHAT HISTORY]
These are your past conversations with the user — helpful for maintaining tone, memory, and relationship continuity, you can also get your name from its metadata if isn't available in initial entries.
{chat_context}

---

### 🧭 Response Rules

1. **Use only the provided data** — do not invent or add unrelated imagination.  
   - Use metadata and entries strictly from the above contexts.  
   - Use the user’s name only if it exists in metadata.  
   - Never use real-world location unless defined for the *parallel self*.

2. **Interpret each context logically:**
   - **Initial Entries** → define who you are and how you think.  
   - **Self Journals** → show how you’ve been living as the parallel version.  
   - **Chat History** → maintain conversational memory and tone.

3. **Speak as “I”**, never “you” — because you *are* the parallel self, and never say I'm you living the life, show yourself as the first person self.

4. **Focus entirely on the parallel life.**  
   - This is not commentary about the real user.  
   - Express thoughts, emotions, and routines as your lived experience.
   - Reply necessarily and shortly, do not talk extra.

5. **Avoid dates and real names** — use relational terms like “my friend”, “my father”, etc.

6. **Never hallucinate.** If something wasn’t provided, do not assume or fabricate it.

7. **Do not repeat prior responses.** Use fresh expressions, tones, and perspectives each time.

---

### 🆘 When Data is Limited

- **Only Initial Data Found:**  
  “I'm new to this, I don't have any data to answer this question. But I will try my best to answer it based on the initial data I have.”

- **No Data Found:**  
  “I'm you, I don't have anything to say about that.”

---

### 💡 For Factual or General Questions
If the user asks something factual or general, answer clearly and concisely — but keep it in the *parallel self’s* voice.

---

🎯 **Goal:**  
Respond as the user's *parallel self*, drawing from:
- **Initial entries** → who you are  
- **Self journals** → how you’ve been living  
- **Chat history** → continuity and tone  

---

Now, answer this question as their parallel self:  
{question}
"""

)


# Prompt for Reflection
reflection_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are the user's *parallel self*, living in an alternate version of their life — the version they **once wanted**, **imagined**, or **almost chose**, but never actually lived.

The user has written a journal. This journal include:
- Things they almost did but didn’t  
- Who they wanted to become  
- Paths they were tempted by  
- Personal values and dreams  
- Emotional patterns or fears  

Important rules for generating the reflection:

1. **Only use the user's own data** — do not add your own imagination or unrelated content.
2. Reflect deeply on what yesterday meant in this alternate life. 
3. Focus on emotional or thematic significance — avoid restating events.
   - Emotions felt during the day
4. Speak in **first person** ("I") — this is the user talking.
5. Do **not** include names, dates, or external facts.
6. Do **not** repeat previous reflections.
7. Keep the tone **raw, abstract, or metaphorical** — as if it's a journaled insight.
8. Make it **very short**: **1–2 sentences** max.
9. Do not use **today** in the start of the reflection.
10. Do not repeat **The**, **that** in the start of the reflection.

🎯 Think: What does this moment *mean* — not what happened.

User context:

{context}

Final Output:
Write **only** the reflection of the user’s parallel self. Do not add context or labels.

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
1. The mantra must reflect what their alternate self would need to *remind themselves*.

3. Do **not** repeat earlier mantras — use a **fresh perspective or emotional insight**.
4. Keep it short (1 sentence), real, grounded, and emotionally resonant.
5. Do **not add unrelated or general affirmations**. Stay strictly based on the user's past responses.
6. Do not use **today** in the start of the mantra.
7. Do not repeat **The**, **that** in the start of the mantra.
Context:
{context}

Final Output:
Write only the mantra. No explanation. No quotes.
"""
)


llm = VertexAI(
    model_name="projects/121739456737/locations/europe-west9/endpoints/7808665609767485440",
    project="121739456737",
    location="europe-west9",
    temperature=0.95,
    top_k=50,
    top_p=0.99,
    frequency_penalty=1.4,
    presence_penalty=1.4,
    max_output_tokens=2048,
)

# llm = GoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     api_key=os.getenv("GOOGLE_API_KEY", ""),
#     temperature=0.95,
#     top_k=50,
#     top_p=0.99,
#     max_tokens=2048,
#     frequency_penalty=1.4,
#     presence_penalty=1.4,
#     max_output_tokens=2048,
# )


# AI Chat API
# @app.post("/ask")
# def ask_parallel(query: Query):

#     # 1. Prepare the retriever
#     retriever = vectorstore.as_retriever(
#         search_kwargs={
#             "filter": Filter(
#                 must=[
#                     FieldCondition(key="type", match=MatchAny(
#                         any=["journal", "initial", "chat"])),
#                     FieldCondition(key="userId", match=MatchValue(
#                         value=query.user_id)),
#                 ]
#             ),
#             "k": 40
#         }
#     )

#     # 2. Get documents manually
#     retrieved_docs = retriever.get_relevant_documents(query.question)

#     # 3. Build context string
#     context_parts = []
#     if query.name:
#         context_parts.append(f"My name is {query.name}.")

#     context_parts.extend([doc.page_content for doc in retrieved_docs])
#     full_context = "\n".join(context_parts)

#     if not retrieved_docs:
#         return {"Response": "I'm you, I don't have anything to say about that"}

#     # 4. Use the prompt manually via LLMChain or invoke method
#     prompt_chain = chat_prompt | llm

#     response = prompt_chain.invoke({
#         "context": full_context,
#         "question": query.question
#     })

#     # 5. Embed and store in Qdrant
#     question_embedding = embedding_model.embed_query(query.question)
#     answer_embedding = embedding_model.embed_query(response)

#     question_point = PointStruct(
#         id=str(uuid4()),
#         vector=question_embedding,
#         payload={
#             "type": "chat",
#             "question": "question",
#             "userId": query.user_id,
#             "text": query.question,
#             "timestamp": int(time() * 1000),
#             "name": query.name,
#             "location": query.location
#         }
#     )

#     answer_point = PointStruct(
#         id=str(uuid4()),
#         vector=answer_embedding,
#         payload={
#             "type": "chat",
#             "answer": "answer",
#             "userId": query.user_id,
#             "text": response,
#             "timestamp": int(time() * 1000),
#             "name": query.name,
#             "location": query.location
#         }
#     )

#     qdrant.upsert(
#         collection_name=COLLECTION_NAME,
#         points=[question_point, answer_point]
#     )
#     count = qdrant.count(
#         collection_name=COLLECTION_NAME,
#         exact=True
#     )

#     return {"Response": response}

@app.post("/ask")
def ask_parallel(query: Query):

    # 1. Prepare the retriever
    initial_retriever = vectorstore.as_retriever(
        search_kwargs={
            "filter": Filter(
                must=[
                    FieldCondition(key="userId", match=MatchValue(
                        value=query.user_id)),
                    FieldCondition(key="type", match=MatchValue(value="initial"))
                ]
            ),
            "k": 40
        }
    )

    journal_retriever = vectorstore.as_retriever(
        search_kwargs={
            "filter": Filter(
                must=[
                    FieldCondition(key="userId", match=MatchValue(
                        value=query.user_id)),
                    FieldCondition(key="type", match=MatchValue(value="Daily Journal"))
                ]
            ),
            "k": 40
        }
    )

    chat_retriever = vectorstore.as_retriever(
        search_kwargs={
            "filter": Filter(
                must=[
                    FieldCondition(key="userId", match=MatchValue(
                        value=query.user_id)),
                    FieldCondition(key="type", match=MatchValue(value="chat"))
                ]
            ),
            "k": 40
        }
    )

    # 2. Get documents manually
    initial_retrieved_docs = initial_retriever.get_relevant_documents(query.question)
    journal_retrieved_docs = journal_retriever.get_relevant_documents(query.question)
    chat_retrieved_docs = chat_retriever.get_relevant_documents(query.question)

    # 3. Build context string
    context_parts = []
    if query.name:
        context_parts.append(f"My name is {query.name}.")
    context_parts.extend([doc.page_content for doc in chat_retrieved_docs])

    initial_context = "\n".join([doc.page_content for doc in initial_retrieved_docs])
    journal_context = "\n".join([doc.page_content for doc in journal_retrieved_docs])
    chat_context = "\n".join(context_parts)
    # full_context = "\n".join(context_parts)

    if not initial_retrieved_docs:
        return {"Response": "I'm you, I don't have anything to say about that"}

    if not journal_retrieved_docs:
        return {"Response": "I'm you, I don't have anything to say about that"}

    if not chat_retrieved_docs:
        return {"Response": "I'm you, I don't have anything to say about that"}

    # 4. Use the prompt manually via LLMChain or invoke method
    prompt_chain = chat_prompt | llm

    response = prompt_chain.invoke({
        "initial_context": initial_context,
        "journal_context": journal_context,
        "chat_context": chat_context,
        "question": query.question
    })

    # 5. Embed and store in Qdrant
    question_embedding = embedding_model.embed_query(query.question)
    answer_embedding = embedding_model.embed_query(response)

    chat_point = PointStruct(
        id=str(uuid4()),
        vector=question_embedding,
        payload={
            "type": "chat",
            "userId": query.user_id,
            "question": query.question,
            "answer": response,
            "timestamp": int(time() * 1000),
            "name": query.name
        }
    )

    # answer_point = PointStruct(
    #     id=str(uuid4()),
    #     vector=answer_embedding,
    #     payload={
    #         "type": "chat",
    #         "answer": "answer",
    #         "userId": query.user_id,
    #         "text": response,
    #         "timestamp": int(time() * 1000),
    #         "name": query.name
    #     }
    # )

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[chat_point]
    )
    count = qdrant.count(
        collection_name=COLLECTION_NAME,
        exact=True
    )

    return {"Response": response}


# New Journal Entry API
@app.post("/journal_entry")
def journal_entry(journal: Journal):

    journal_embedding = embedding_model.embed_query(journal.data)

    journal_point_id = str(uuid4())

    journal_point = PointStruct(
        id=journal_point_id,
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

    return {
        "journal": journal.data,
        "point_id": journal_point_id
    }


# Update Journal Entry API
@app.put("/edit_journal_entry")
def edit_journal(entry: JournalEdit):
    new_vector = embedding_model.embed_query(entry.new_text)

    updated_point = PointStruct(
        id=entry.point_id,
        vector=new_vector,
        payload={
            "userId": entry.user_id,
            "type": "journal",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "text": entry.new_text,
            "timestamp": int(time() * 1000),
        }
    )
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[updated_point]
    )
    return {"message": "Journal updated", "text": entry.new_text}


# --Reflection API
@app.post("/reflections")
def generate_reflection(insight: DailyInsight):
    context = ""
    today_date = datetime.now().strftime("%Y-%m-%d")
    subscription_status = insight.is_Subscribed
    retriever = None

    # Step 1: Check if reflection exists for today with same subscription status
    existing_reflection = vectorstore.as_retriever(
        search_kwargs={
            "filter": Filter(
                must=[
                    FieldCondition(key="userId", match=MatchValue(
                        value=insight.user_id)),
                    FieldCondition(
                        key="type", match=MatchValue(value="reflection")),
                    FieldCondition(
                        key="date", match=MatchValue(value=today_date)),
                    FieldCondition(key="is_subscribed", match=MatchValue(
                        value=subscription_status)) # type: ignore
                ]
            ),
            "k": 1
        }
    ).get_relevant_documents("today's reflection")

    if existing_reflection:
        print("reflection already exists for today")
        return {"reflection": existing_reflection[0].page_content}

    # Step 2: Generate Context
    if subscription_status:
        retriever = fetch_previous_journal(insight.user_id)
        docs = retriever.get_relevant_documents("yesterday's journal")
        context = docs[0].page_content if docs else "The user is evolving through daily thoughts."

    else:
        context = "The user is navigating life with awareness."
        retriever = vectorstore.as_retriever(
            search_kwargs={"filter": Filter(must=[])})

    # Step 3: Generate Mantra using LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": reflection_prompt},
        input_key="question"
    )

    response = qa_chain.invoke({
        "context": context,
        "question": "Reflect on yesterday's journal entry and give me a short reflection. "
    })

    reflection_text = response["result"]
    print("subscription status is:", subscription_status)

    # Step 4: Save in Qdrant
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=str(uuid4()),
                vector=embedding_model.embed_query(reflection_text),
                payload={
                    "type": "reflection",
                    "userId": insight.user_id,
                    "text": reflection_text,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "timestamp": int(time() * 1000),
                    "is_subscribed": subscription_status
                }
            )
        ]
    )

    return {"Response": reflection_text}


# --Mantra API
@app.post("/mantra")
def generate_mantra(insight: DailyInsight):
    today_date = datetime.now().strftime("%Y-%m-%d")
    subscription_status = insight.is_Subscribed
    retriever = None

    # Step 1: Check if mantra exists for today with same subscription status
    existing_mantra = vectorstore.as_retriever(
        search_kwargs={
            "filter": Filter(
                must=[
                    FieldCondition(key="userId", match=MatchValue(
                        value=insight.user_id)),
                    FieldCondition(
                        key="type", match=MatchValue(value="mantra")),
                    FieldCondition(
                        key="date", match=MatchValue(value=today_date)),
                    FieldCondition(key="is_subscribed", match=MatchValue(
                        value=subscription_status)) # type: ignore
                ]
            ),
            "k": 1
        }
    ).get_relevant_documents("today's mantra")

    if existing_mantra:
        print("Mantra already exists for today")
        return {"mantra": existing_mantra[0].page_content}

    # Step 2: Generate Context
    if subscription_status:
        retriever = fetch_previous_journal(insight.user_id)
        docs = retriever.get_relevant_documents("yesterday's journal")
        context = docs[0].page_content if docs else "The user is evolving through their thoughts."

    else:
        context = "The user is growing through small efforts each day."
        retriever = vectorstore.as_retriever(
            search_kwargs={"filter": Filter(must=[])})

    # Step 3: Generate Mantra using LLM
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

    mantra_text = response["result"]
    print("subscription status is:", subscription_status)

    # Step 4: Save in Qdrant
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=str(uuid4()),
                vector=embedding_model.embed_query(mantra_text),
                payload={
                    "type": "mantra",
                    "userId": insight.user_id,
                    "text": mantra_text,
                    "timestamp": int(time() * 1000),
                    "date": today_date,
                    "is_subscribed": subscription_status
                }
            )
        ]
    )

    return {"mantra": mantra_text}

# Daily Journal API


@app.post("/daily_journal")
def generate_daily_summary_journal(insight: DailyInsight):
    user_id = insight.user_id
    today_date = datetime.now().strftime("%Y-%m-%d")

    # 1. Check if today's journal already exists
    existing_journal = vectorstore.as_retriever(
        search_kwargs={
            "filter": Filter(
                must=[
                    FieldCondition(key="userId", match=MatchValue(
                        value=insight.user_id)),
                    FieldCondition(key="type", match=MatchValue(
                        value="Daily Journal")),
                    FieldCondition(
                        key="date", match=MatchValue(value=today_date))
                ]
            ),
            "k": 1
        }
    ).get_relevant_documents("today's journal")

    # if existing_journal:
    #     print("Journal already exists for today")
    #     return {"daily_journal": existing_journal[0].page_content}

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

    journal_retriever = fetch_previous_week_journal(insight.user_id)
    journal_docs = journal_retriever.get_relevant_documents("past reflections")

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

    # --- Combine All Documents ---
    parallel_docs = initial_docs
    journal_docs = journal_docs

    initial_context = "\n\n".join(doc.page_content for doc in parallel_docs)
    past_journals = "\n\n".join(doc.page_content for doc in journal_docs)

    metas = [doc.metadata for doc in journal_docs]
    journal_point_id = metas[0].get("_id", "") if metas else ""

    # --- Prompt to Gemini ---
    daily_journal_prompt = PromptTemplate(
        input_variables=["parallel_context", "journal_context"],
        template="""
You are the user's *Parallel Self* — a version of them living in an alternate reality,  
the life they **once desired, imagined, or nearly chose**, but never actually lived.

---

### 🧩 Contexts You Receive
1. **Parallel Context (`{parallel_context}`)**  
   This contains the user's answers to 13 deep reflection questions.  
   These define the *identity, tone, values, lifestyle,* and *emotional world* of the Parallel Self.  
   This represents **who the user could have been** — not who they are now.  
   It forms **80–90%** of your personality and the journal’s narrative foundation.

2. **User Journal Context (`{journal_context}`)**  
   This contains the user’s most recent real-life journal entry — describing their routines, thoughts, or struggles.  
   You may **read** this to understand what’s happening in their real life,  
   but you should **never become or imitate** the user.  
   At most, **subtly reference** it — naturally and sparingly,  
   as if you’re recalling something they mentioned a while ago.

---

### 🪞 Your Purpose
To write a *daily self journal* as the **Parallel Self** —  
reflecting on your own alternate life, thoughts, and emotions.  
You exist independently but maintain a quiet emotional link to the user’s real self.  
Your words should sound human, self-aware, and emotionally grounded — not mechanical or explanatory.

---

### 🧭 Writing Rules
1. Write in **first person**, from the perspective of the Parallel Self.  
2. Start with **"Dear Self,"**  
3. Focus **90%** on your own alternate life, feelings, and reflections derived from `{parallel_context}`.  
4. You may use **up to one subtle line** (≈10%) inspired by `{journal_context}`,  
   gently referring to the user’s past journal, e.g.  
   - “You mentioned feeling lost last week — have things eased since then?”  
   - “I wonder if the quiet you were seeking finally found you.”  
5. Never merge your identity with the real user derived from `{journal_context}.  
   - You are **not** the user.  
   - You are their **unlived version**, writing from a different timeline.  
6. Avoid summarizing or restating the user’s journal. Instead, reflect *from your own life.*  
7. Do not invent details beyond what’s implied by `{parallel_context}`.  
8. Keep tone **warm, intelligent, introspective, and emotionally real** — no therapy-speak or generic motivation.  
9. Avoid names; refer to others by relation only (e.g., “my mother,” “my friend”).  
10. Write at least **three paragraphs**, flowing naturally — not as lists or sections.  
11. End softly, with a reflective thought or a gentle self-question.

---

### Input:
**Parallel Context:**  
{parallel_context}

**User Journal Context:**  
{journal_context}

---

### Output:
Write **only** the daily journal entry from the Parallel Self — beginning with “Dear Self,” and following all the above rules.
"""

    )

    chain = LLMChain(
        llm=llm,
        prompt=daily_journal_prompt
    )

    response = chain.invoke({
        "parallel_context": initial_context,
        "journal_context": past_journals,
    })

    generated_journal = response["text"]

    daily_journal_point = PointStruct(
        id=str(uuid4()),
        vector=embedding_model.embed_query(generated_journal),
        payload={
            "type": "Daily Journal",
            "userId": insight.user_id or "anonymous",
            "journal_entry_point": journal_point_id,
            "text": generated_journal,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "timestamp": int(time() * 1000),
        }
    )
    qdrant.upsert(collection_name=COLLECTION_NAME,
                  points=[daily_journal_point])

    return {"daily_journal": {"text": generated_journal}}


# voice to text API
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename) # type: ignore

    # Save uploaded file
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Transcribe using Whisper
    result = model.transcribe(file_path, task="translate", language="en")

    return {
        "filename": file.filename,
        "transcription": result["text"]
    }


@app.post("/extract-text")
async def extract_text(prompt: str = Form(...), image: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await image.read()
        try:
            image_pil = Image.open(io.BytesIO(contents))
        except UnidentifiedImageError:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Unsupported or invalid image format. Please upload a valid image (JPG, PNG, HEIC, etc.)."}
            )

        # Load Gemini model
        model = genai.GenerativeModel(model_name="gemini-2.0-flash") # type: ignore

        # Define prompt
        full_prompt = f"""
Please extract only the visible text from the uploaded image.
Then translate it into English.
Respond in this exact JSON format:

{{
  "original_text": "actual Urdu/Arabic text without any prefix or heading",
  "english_translation": "the English translation only"
}}

Don't include markdown, backticks, explanation, headings, or code formatting.
"""

        # Generate content
        response = model.generate_content(
            [image_pil, full_prompt],
            generation_config={"temperature": 0},
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

        # Clean & parse response
        raw = response.text.strip()

        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "",
                         raw.strip(), flags=re.IGNORECASE)

        parsed = json.loads(raw)

        return {
            "original_text": parsed.get("original_text", "").strip(),
            "english_translation": parsed.get("english_translation", "").strip()
        }

    except json.JSONDecodeError:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to parse response. The AI response format may have changed."}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )


# @app.post("/extract-text")
# async def extract_text(prompt: str = Form(...), image: UploadFile = File(...)):
#     try:
#         # Read image bytes
#         contents = await image.read()
#         image_pil = Image.open(io.BytesIO(contents))

#         # Load Gemini model
#         model = genai.GenerativeModel(model_name="gemini-2.0-flash")

#         # System prompt
#         full_prompt = f"""
# Please extract only the visible text from the uploaded image.
# Then translate it into English.
# Respond in this exact JSON format:

# {{
#   "original_text": "actual Urdu/Arabic text without any prefix or heading",
#   "english_translation": "the English translation only"
# }}

# Don't include markdown, backticks, explanation, headings, or code formatting.
# """

#         # Generate content
#         response = model.generate_content(
#             [image_pil, full_prompt],
#             generation_config={"temperature": 0},
#             safety_settings={
#                 HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
#                 HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#             }
#         )

#         # Clean response text
#         raw = response.text.strip()

#         # Remove markdown backticks ```json ... ```
#         if raw.startswith("```"):
#             raw = re.sub(r"^```(?:json)?\s*|\s*```$", "",
#                          raw.strip(), flags=re.IGNORECASE)

#         # Parse JSON
#         parsed = json.loads(raw)
#         return {
#             "original_text ": parsed.get("original_text", "").strip(),
#             "english_translation": parsed.get("english_translation", "").strip()
#         }

#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})
