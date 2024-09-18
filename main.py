from fastapi import FastAPI, HTTPException
from firebase import db
from firebase_admin import auth
from pydantic import BaseModel
from typing import Dict,List


app = FastAPI()

class UserRegister(BaseModel):
    email: str
    password: str
    
    
@app.get('/test')
async def test()->Dict:
    return {"message": "Success!"}