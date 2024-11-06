from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from firebase import db
from firebase_admin import auth, storage
from pydantic import BaseModel
from typing import Dict, List
import os
from source import main

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['https://verbisense.vercel.app'],  # Allow only localhost:5173
    allow_credentials=True, 
    allow_methods=['*'],
    allow_headers=['*']
)

class QueryChat(BaseModel):
    userId: str
    files: List
    query: str
    
    
bucket = storage.bucket("verbisense.appspot.com") 
    

@app.post("/chat")
async def chat(data: QueryChat):
    try:
        print("userId : ",data.userId)
        print("files : ",data.files)
        print("query : ",data.query)
        
        response = main(data.files,data.query)
        
        print("\n" + "="*50)
        print(response)
        print("="*50)
        if not response:
            return False
        return {"query":data.query,"response":response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")