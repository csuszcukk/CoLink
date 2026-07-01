from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI(title="Ollama API Wrapper")

OLLAMA_URL = "http://localhost:11434/api/generate"

class PromptRequest(BaseModel):
    model: str = "llama3"
    prompt: str