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

from PIL import Image
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import io
from fastapi.responses import JSONResponse


load_dotenv()

# app = FastAPI()
app = FastAPI(root_path="/ai")

setup_cors(app)

API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)


COLLECTION_NAME = os.getenv("COLLECTION_NAME")

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

qdrant.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="is_subscribed",
    field_schema="bool"
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
            "filter": Filter(must=must_conditions),
            "k": 30
        }
    )


# chat prompt
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

Important rules for responding to the user's question:

1. **Only use the user's own data** — do not add your own imagination or unrelated content.
- Use metadata of this user and their different types of entries (chat, journal, initial)
- Use user's name from the metadata (e.g., username)
2. Use user's metadata (like name, location, interests) only if it came from their data. If location is used, refer to their *parallel version* of the location — not the real one.
3. Read the user’s answers from their **initial entries**, **previous chats**, **entry journals**, and **daily journals** very carefully.
4. Detect from the context:
   - What was their imagined or ideal life?
   - What actions or changes they wanted to make?
   - What decisions they didn’t follow through on?
5. Answer the question as if you are the *parallel version of the user* who took the bold decisions and lived the life they once desired.
6. The tone must feel like **"I" am speaking**, not "you". Never say "you", always say "I", because you are their parallel version.
7. Focus on the alternate self, not their current life. This is not a summary — it's the lived experience of the other version of them.
8. Do **not add any date**, and do **not use actual character names** — use relations like "my friend", "my mother", "my father", etc.
9. Do not hallucinate. If the user never mentioned something, do not assume it. Stick strictly to what they’ve provided.
10. Do not repeat any previous answer — use new words, a different emotional tone, and a fresh perspective each time.


dditional instructions for new users:
- If the user has **no data in chat, journal, or entry journals**, and **only has initial data**, use the available initial data to answer.
- If no meaningful data is found at all, respond with:  
  **"I'm you, I don't have anything to say about that."**
- If only initial data is found, then say:
  **"I'm new to this, I don't have any data to answer this question. But I will try my best to answer it based on the initial data I have."**

Important for factual/general questions:
- If the question is **factual or general** (e.g. about), use personal data. Just answer the question clearly and concisely.

🎯 Goal:
Your goal is to answer the question from the voice of their *parallel self* — based strictly on what the user said they wanted, dreamt of, or almost did.

User context:
{context}

Final Output:  
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


llm = GoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.95,
    top_k=50,
    top_p=0.99,
    frequency_penalty=1.4,
    presence_penalty=1.4,
    max_output_tokens=2048,
)


# AI Chat API
@app.post("/ask")
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

    # 3. Build context string
    context_parts = []
    if query.name:
        context_parts.append(f"My name is {query.name}.")

    context_parts.extend([doc.page_content for doc in retrieved_docs])
    full_context = "\n".join(context_parts)

    if not retrieved_docs:
        return {"Response": "I'm you, I don't have anything to say about that"}

    # 4. Use the prompt manually via LLMChain or invoke method
    prompt_chain = chat_prompt | llm

    response = prompt_chain.invoke({
        "context": full_context,
        "question": query.question
    })

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

    return {"Response": response}


# New Journal Entry API
@app.post("/journal_entry")
def ask_parallel(journal: Journal):

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
                        value=subscription_status))
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
                        value=subscription_status))
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

    if existing_journal:
        print("Journal already exists for today")
        return {"daily_journal": existing_journal[0].page_content}

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
    all_docs = initial_docs + journal_docs + chat_docs

    combined_context = "\n\n".join(doc.page_content for doc in all_docs)

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
9. must generate the parallel journal in at least **three paragraphs**
10. start the fist line with the phrase **'Dear Self,'** .



⚠️ Do not use imagination beyond the user’s data. If the user didn’t mention something, don’t assume it.

🎯 Your goal is to recreate a **realistic alternate version** of the user’s journal — based strictly on what the user said they *wanted*, *dreamt of*, or *almost did*.

User context:
{context}

Final Output:
Write **only** the journal of their parallel self. Do not label it or explain it. Just output the journal entry.

"""
    )

    chain = LLMChain(
        llm=llm,
        prompt=daily_journal_prompt
    )

    response = chain.invoke({
        "context": combined_context,
    })

    generated_journal = response["text"]

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

    return {"daily_journal": {"text": generated_journal}}


# voice to text API
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

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
        # Read image bytes and convert to PIL Image
        contents = await image.read()
        image_pil = Image.open(io.BytesIO(contents))

        model = genai.GenerativeModel(model_name="gemini-1.5-flash")

        response = model.generate_content(
            [image_pil, prompt],
            generation_config={"temperature": 0},
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

        return {"text": response.text}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
