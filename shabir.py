from fastapi import FastAPI, File, UploadFile, Form
from dotenv import load_dotenv
import os
from middlewares.cors import setup_cors
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
genai.configure(api_key=API_KEY) # type: ignore


@app.post("/extract-text")
async def extract_text(prompt: str = Form(...), image: UploadFile = File(...)):
    try:
        # Read image bytes and convert to PIL Image
        contents = await image.read()
        image_pil = Image.open(io.BytesIO(contents))

        model = genai.GenerativeModel(model_name="gemini-1.5-flash") # type: ignore

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
