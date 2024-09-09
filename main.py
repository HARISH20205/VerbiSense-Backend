from fastapi import FastAPI, HTTPException
from firebase import db
from firebase_admin import auth
from pydantic import BaseModel

app = FastAPI()

class UserRegister(BaseModel):
    email: str
    password: str

@app.get('/')
async def hello():
    return {"message": "Hello, World!"}

# Route to create a user in Firebase Authentication
@app.post("/register")
async def register_user(user: UserRegister):
    try:
        # Create the user with email and password
        user_record = auth.create_user(email=user.email, password=user.password)
    
        return {"message": "User created successfully", "uid": user_record.uid}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))