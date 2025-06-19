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
    # if insight.is_Subscribed == True:
    if insight.user_id == True:
        retriever = fetch_previous_journal(insight.user_id)
        context_docs = retriever.get_relevant_documents("yesterday's mantra")
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
    initial_retriever = vectorstore.as_retriever(
        search_kwargs={
            "filter": Filter(
                must=[
                    FieldCondition(
                        key="type",
                        match=MatchValue(value="initial")
                    ),
                    FieldCondition(
                        key="userId",
                        match=MatchValue(value=user_id)
                    )
                ]
            ),
            "k": 13
        }
    )
    initial_docs = initial_retriever.get_relevant_documents("initial")
    print("Initial Docs for daily journal:", initial_docs)

    # --- Fetch All Past Journals ---
    journal_retriever = vectorstore.as_retriever(
        search_kwargs={
            "filter": Filter(
                must=[
                    FieldCondition(
                        key="type", match=MatchValue(value="journal")),
                    FieldCondition(
                        key="userId", match=MatchValue(value=user_id))
                ]
            ),
            "k": 100  # Increase if more journals exist
        }
    )
    journal_docs = journal_retriever.get_relevant_documents("journal")
    print("Journal Docs for daily journal:", journal_docs)

    # --- Combine All Content ---
    context = "\n\n".join(
        [doc.page_content for doc in initial_docs + journal_docs])

    # --- Prompt to Gemini ---
    daily_journal_prompt = PromptTemplate(
        input_variables=["context"],
        template="""
    You are the user's inner self — aware of their thoughts, goals, and past experiences.

    Below is a collection of their initial self-reflection answers and daily journal entries:
    {context}

    Now write **a completely new** journal entry from the given user's context, inital data and all journals that reflects their personal thoughts, growth, and feelings.  
    Use only the data from the above context — **don't include anything that's not aligned with their data** (e.g., no irrelevant mentions of rain, struggles, or events that were never described).  
    Keep the tone authentic and human — it should feel like the user wrote it in their own words.

    Instructions:
    - Don't repeat content. Write a fresh journal even if the questions or themes are reused.
    - Try a new emotional angle each time — explore different aspects of what they’ve shared.
    - Let the journal feel organic — like a snapshot of what the user might genuinely be thinking or feeling and make sure to use the data of the user from the initial and journals.
    - Keep it grounded. Use about **90% actual context**, and at most **10% subtle, fitting imagination** — but only if that imagination enhances the realism of the journal and aligns perfectly with the user’s profile and tone.

    Final Output:
    Write **only** the new journal entry. Do not explain it or add labels like "Journal".
    """
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=journal_retriever,  # optional, for structure
        chain_type="stuff",
        chain_type_kwargs={"prompt": daily_journal_prompt},
        input_key="question"
    )

    response = chain.invoke({
        "context": context,
        "question": "Write today's journal entry."
    })

    generated_journal = response["result"]

    # --- Save to Qdrant ---
    point = PointStruct(
        id=str(uuid4()),
        vector=embedding_model.embed_query(generated_journal),
        payload={
            "type": "daily_journal",
            "userId": user_id,
            "text": generated_journal,
            "timestamp": int(time() * 1000),
            "date": datetime.now().strftime("%Y-%m-%d")
        }
    )
    qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])

    return {"daily_journal": generated_journal}
