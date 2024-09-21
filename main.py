from fastapi import FastAPI, HTTPException
from firebase import db
from firebase_admin import auth, storage
from pydantic import BaseModel
from typing import Dict, List
import os


app = FastAPI()

class UserRegister(BaseModel):
    email: str
    password: str

# Define a request body model using Pydantic
class UserUpdateRequest(BaseModel):
    field: str
    value: str

bucket = storage.bucket("verbisense.appspot.com") 

file ="isample.jpg"
# Function to upload file to Firebase Storage
@app.post("/upload-file/")
async def upload_file():
    try:
        # Path to the local file you want to upload
        file_path = os.path.join('files', file)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail="File does not exist.")

        # Blob is the storage object in Firebase
        blob = bucket.blob(f'uploads/{file}')  
        
        # Upload file from the local path
        blob.upload_from_filename(file_path)

        # Optional: Make the file publicly accessible
        blob.make_public()

        return {
            "message": "File uploaded successfully",
            "public_url": blob.public_url  # Return the public URL for the uploaded file
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# Endpoint to update user details
@app.put("/update-user/{user_id}")
async def update_user(user_id: str, request: UserUpdateRequest):
    try:
        # Reference to the specific user's document
        user_ref = db.collection("users").document(user_id)
        
        # Update the document with the new field values
        user_ref.update({
            request.field: request.value
        })

        return {"message": "User updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.get("/get-file/")
async def get_file():
    try:
        # Define the path of the file in Firebase Storage
        file_path_in_storage = f'uploads/{file}'
        
        # Reference to the file (blob) in Firebase Storage
        blob = bucket.blob(file_path_in_storage)
        
        # Check if the file exists in Firebase Storage
        if not blob.exists():
            raise HTTPException(status_code=404, detail="File not found in Firebase Storage.")

        # Optional: You can return the public URL if the file is publicly accessible
        if not blob.public_url:
            # Optionally make the file public if it isn't already
            blob.make_public()

        return {
            "message": "File found",
            "public_url": blob.public_url  # Return the public URL of the file
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
