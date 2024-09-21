import firebase_admin
from firebase_admin import credentials, firestore
import os

credPath = os.path.join("services","firebase_credentials.json")
# Path to your Firebase credentials JSON file
cred = credentials.Certificate(credPath)

# Initialize the Firebase app with storageBucket
firebase_admin.initialize_app(cred, {
    'storageBucket': 'verbisense.appspot.com'  # Replace with your bucket name
})

# Initialize Firestore DB
db = firestore.client()