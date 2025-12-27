import pandas as pd
import qrcode
from io import BytesIO
import joblib
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Load processed filescl
df = pd.read_csv("processed_doctor_data.csv")
vectorizer = joblib.load("vectorizer.pkl")
vectors = joblib.load("vectors.pkl")

# Preprocess function
def preprocess_text(text):
    if pd.isna(text):
        return ""
    return " ".join(text.lower().split())

# Recommendation function
def recommend_doctor(disease, city_filter=None):
    disease = preprocess_text(disease)
    disease_vector = vectorizer.transform([disease])

    filtered_df = df.copy()
    if city_filter and city_filter != "All":
        filtered_df = filtered_df[filtered_df['City'] == city_filter]
        filtered_vectors = vectorizer.transform(filtered_df['combined_text'])
    else:
        filtered_vectors = vectors

    if filtered_df.empty:
        return pd.DataFrame()

    similarity_scores = cosine_similarity(disease_vector, filtered_vectors)[0]
    threshold = 0.1
    filtered_indices = [i for i, score in enumerate(similarity_scores) if score >= threshold]

    if not filtered_indices:
        return pd.DataFrame()

    sorted_indices = sorted(filtered_indices, key=lambda i: similarity_scores[i], reverse=True)
    top_docs = filtered_df.iloc[sorted_indices[:10]]

    return top_docs[['Name', 'specialization', 'Location', 'City', 'Consult Fee', 'Years of Experience']]

# QR Code Generator
def create_qr_code(doctor_name, amount):
    upi_id = "tanishkmalviya@ibl"
    upi_url = f"upi://pay?pa={upi_id}&pn={doctor_name}&am={amount}&cu=INR"
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(upi_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    #convert to bytes
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

# Streamlit UI
st.title("🩺 Doctor Recommendation System")
st.write("Enter your symptoms or disease to get doctor recommendations.")

if 'selected_doctor' not in st.session_state:
    st.session_state.selected_doctor = None
if 'show_qr' not in st.session_state:
    st.session_state.show_qr = False
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'disease_query' not in st.session_state:
    st.session_state.disease_query = ""
if 'selected_city' not in st.session_state:
    st.session_state.selected_city = "All"

col1, col2 = st.columns([3, 1])
with col1:
    specializations = sorted(df["specialization"].dropna().unique())
    disease_query = st.selectbox("Select Specialization/Disease:", specializations, index=0)
    st.session_state.disease_query = disease_query

with col2:
    city_list = ["All"] + sorted(df["City"].dropna().unique())
    selected_city = st.selectbox("Select City:", city_list, index=city_list.index(st.session_state.selected_city))
    st.session_state.selected_city = selected_city

# Search trigger
if st.button("Find Doctors") and disease_query.strip():
    st.session_state.disease_query = disease_query
    results = recommend_doctor(disease_query, selected_city)
    if results.empty:
        st.warning("No matching doctors found. Try different input.")
    else:
        st.session_state.search_results = results
        st.rerun()
elif not disease_query.strip():
    st.warning("Please enter a valid disease or symptom.")


# Show QR on booking
if st.session_state.selected_doctor is not None and st.session_state.show_qr:
    doc = st.session_state.selected_doctor
    qr_img = create_qr_code(doc['Name'], doc['Consult Fee'])
    st.success(f"✅ Appointment Booking Initiated for Dr. {doc['Name']}")
    st.image(qr_img, caption=f"Scan to Pay ₹{doc['Consult Fee']} via UPI", width=300)
    st.info("After successful payment, you will receive a confirmation message.")
    if st.button("🔙 Back"):
        st.session_state.selected_doctor = None
        st.session_state.show_qr = False
        st.rerun()

# Show results
elif st.session_state.search_results is not None:
    st.subheader("Recommended Doctors:")
    for i, row in st.session_state.search_results.iterrows():
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"**👨‍⚕️ {row['Name']}**")
            st.write(f"📌 {row['specialization']}")
            st.write(f"📍 {row['Location']}, {row['City']}")
            st.write(f"💰 ₹{row['Consult Fee']} | 🏅 {row['Years of Experience']} yrs")
            if st.button(f"Book Appointment with {row['Name']}", key=f"book_{i}"):
                st.session_state.selected_doctor = row
                st.session_state.show_qr = True
                st.rerun()
        st.markdown("---")
