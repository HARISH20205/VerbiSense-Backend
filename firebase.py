import firebase_admin
from firebase_admin import credentials, firestore
import os

credPath = os.path.join("services","firebase_credentials.json")
# Path to your Firebase credentials JSON file
cred = credentials.Certificate(credPath)

# Initialize the Firebase app
firebase_admin.initialize_app(cred)

# Initialize Firestore DB
db = firestore.client()