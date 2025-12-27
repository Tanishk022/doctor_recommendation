import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
cred = credentials.Certificate("firebasekey.json")  # your service account key
firebase_admin.initialize_app(cred)

# Firestore client
db = firestore.client()

# Load your doctor CSV file
df = pd.read_csv("DOCTOR_data_updated.csv")
print("✅ Running!")

# Upload each doctor to the 'doctors' collection
i=1
for index, row in df.iterrows():
    doctor_data = {
        "Name": row.get("Name", ""),
        "specialization": row.get("specialization", ""),
        "Description": row.get("Description", ""),
        "Location": row.get("Location", ""),
        "City": row.get("City", ""),
        "Consult Fee": float(row.get("Consult Fee", 0)),
        "Years of Experience": row.get("Years of Experience", "")
    }
    print(i)
    i=i+1
    print("\n")
    
    # Use name or index as document ID (or let Firebase auto-generate)
    db.collection("doctors").add(doctor_data)

print("✅ Doctor data uploaded to Firestore!")

