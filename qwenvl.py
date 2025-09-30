from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import torch
import os
import tempfile
from typing import Optional

app = FastAPI()

processor = None
model = None

class ImageRequest(BaseModel):
    system_prompt: Optional[str] = "You are a text extraction assistant. Extract and output only the document content as plain text or simple Markdown. Do not use LaTeX formatting, mathematical notation, or $ symbols. Do not wrap text in \\text{} commands. Output clean, readable text only."
    prompt: Optional[str] = "Extract content as plain text"

def load_model(model_path: str):
    global processor, model
    try:
        processor = AutoProcessor.from_pretrained(model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")

@app.post("/image")
async def process_image(
    file: UploadFile = File(...),
    system_prompt: str = Form("You are a text extraction assistant. Extract and output only the document content as plain text or simple Markdown. Do not use LaTeX formatting, mathematical notation, or $ symbols. Do not wrap text in \\text{} commands. Output clean, readable text only."),
    prompt: str = Form("Extract content as plain text")
):
    if processor is None or model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name

        image = Image.open(temp_path).convert("RGB")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}
        ]

        text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=image, text=text_in, return_tensors="pt").to(model.device)

        output_ids = model.generate(**inputs, max_new_tokens=1500)
        text_out = processor.decode(output_ids[0], skip_special_tokens=True)

        os.unlink(temp_path)

        return {"text": text_out}

    except Exception as e:
        if 'temp_path' in locals():
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    model_path = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")
    load_model(model_path)
    uvicorn.run(app, host="0.0.0.0", port=8000)

