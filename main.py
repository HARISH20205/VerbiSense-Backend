from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
from source import main
import logging

load_dotenv()

# Setup logging
logging.basicConfig(filename="error.log", level=logging.ERROR)

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://verbisense.vercel.app"],  # Only allow frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted Hosts
app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["verbisense.vercel.app", "harish20205-verbisense.hf.space"]
)

# Middleware to Block Unauthorized Origins
@app.middleware("http")
async def block_unauthorized_origins(request: Request, call_next):
    origin = request.headers.get("origin")
    if origin and origin != "https://verbisense.vercel.app":
        raise HTTPException(status_code=403, detail="Unauthorized origin")
    response = await call_next(request)
    return response

# Request Model
class QueryChat(BaseModel):
    userId: str
    files: List
    query: str

@app.get("/")
def read_root():
    return {"message": "Welcome to Verbisense!"}

@app.post("/chat")
async def chat(data: QueryChat):
    try:
        print(f"userId: {data.userId}")
        print(f"files: {data.files}")
        print(f"query: {data.query}")

        response = main(data.files, data.query)

        print("\n" + "=" * 50)
        print(response)
        print("=" * 50)


        return {"query": data.query, "response": response}

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")
