# #Doctor Recommendation System

## Project Overview
The **Doctor Recommendation System** is a web-based application designed to help users find the **most suitable doctor** based on disease or medical condition.  
The system uses structured medical data and real-time database integration to provide **accurate, quick, and reliable doctor recommendations**, improving accessibility to healthcare services.

---

## Problem Statement
Finding the right doctor for a specific disease is often difficult due to:
- Lack of proper medical guidance  
- Limited information about doctor specialization and availability  
- Time-consuming manual search  

This project addresses the problem by offering an **intelligent recommendation system** that connects patients with appropriate doctors efficiently.

---

## Solution Approach
The system allows users to:
- Select or enter a disease/medical condition  
- Retrieve a list of **recommended doctors** based on specialization  
- View doctor details such as:
  - Name  
  - Specialization  
  - Hospital/Clinic  
  - Address  
  - Consultation details  

Doctor data is stored and fetched in **real time** using a cloud-based database.

---

## Technologies Used
- **Programming Language:** Python  
- **Web Framework:** Streamlit  
- **Backend Services:** Firebase Firestore  
- **Database:** NoSQL (Cloud Firestore)  
- **Data Handling:** Pandas  
- **Payment Integration:** UPI QR Code Generation  
- **Development Tools:** VS Code  

---

## System Workflow
1. User selects a disease or medical condition  
2. System maps the disease to required medical specialization  
3. Doctor data is fetched from Firebase Firestore  
4. Relevant doctors are displayed to the user  
5. User can proceed with consultation or payment (if applicable)

---

## Database Description
- Doctor information is stored in **Firebase Firestore**
- Supports:
  - Add new doctors  
  - Update existing doctor details  
  - Remove doctor records  
- Enables **real-time data access** without using static CSV files

---

## How to Run the Project
1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/doctor-recommendation-system.git

## for visiting the webpage
  https://doctor-recommendation-system-vepa5f7fwbazkxivuv8qhm.streamlit.app/
