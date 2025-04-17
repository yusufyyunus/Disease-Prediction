import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from models import *
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import joblib, pickle
import numpy as np, os
from dotenv import load_dotenv
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from fastapi import status
from fastapi.responses import JSONResponse
import pytz

app = FastAPI(title="Health Assessment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True, 
    allow_methods=["*"],    
    allow_headers=["*"],   
)
class EmailRequest(BaseModel):
    assessmentId: int

# ------------------------------------------------
# UTILS
def initialize_database():
    conn = sqlite3.connect("health_assessments.db")
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS health_assessments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        
        -- Demographics
        name TEXT,
        email TEXT,           
        age TEXT,
        MF TEXT,
        married TEXT,
        residence TEXT,
        work_type TEXT,
        
        -- Lifestyle
        smoking TEXT,
        smoking_history TEXT,
        bmi TEXT,
        
        -- Medical History
        hyper TEXT,
        heart TEXT,
        
        -- Cardiovascular
        cp TEXT,
        trestbps TEXT,
        cholesterol TEXT,
        fbs TEXT,
        restecg TEXT,
        thalach TEXT,
        exang TEXT,
        oldpeak TEXT,
        slope TEXT,
        ca TEXT,
        thal TEXT,
        
        -- Glucose
        glucose TEXT,
        HbA1c_level TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS admin_users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        full_name TEXT,
        email TEXT
    )
    ''')
    
    
    cursor.execute("SELECT COUNT(*) FROM admin_users WHERE username = 'admin'")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO admin_users (username, password, full_name, email) VALUES (?, ?, ?, ?)",
            ("admin", "admin123", "System Administrator", "admin@gmail.com")
        )


    
    conn.commit()
    conn.close()

def preprocess_data(record):
    """Convert SQLite record into a structured DataFrame with cleaned input values."""
    
    def clean_value(value):
        """Extract only the part before the comma, if present."""
        return value.split(",")[0] if isinstance(value, str) and "," in value else value
    
    input_data = {
        "age": clean_value(record["age"]),
        "MF": clean_value(record["MF"]),
        "married": clean_value(record["married"]),
        "residence": clean_value(record["residence"]),
        "work_type": clean_value(record["work_type"]),
        "smoking": clean_value(record["smoking"]),
        "smoking_history": clean_value(record["smoking_history"]),
        "bmi": clean_value(record["bmi"]),
        "hyper": clean_value(record["hyper"]),
        "heart": clean_value(record["heart"]),
        "cp": clean_value(record["cp"]),
        "trestbps": clean_value(record["trestbps"]),
        "cholesterol": clean_value(record["cholesterol"]),
        "fbs": clean_value(record["fbs"]),
        "restecg": clean_value(record["restecg"]),
        "thalach": clean_value(record["thalach"]),
        "exang": clean_value(record["exang"]),
        "oldpeak": clean_value(record["oldpeak"]),
        "slope": clean_value(record["slope"]),
        "ca": clean_value(record["ca"]),
        "thal": clean_value(record["thal"]),
        "glucose": clean_value(record["glucose"]),
        "HbA1c_level": clean_value(record["HbA1c_level"])
    }

    return pd.DataFrame([input_data])

def initilize_models():
    # Load the model
    diabetes_model = joblib.load("model/diabetes_prediction_model.pkl")
    hypertension_model = joblib.load("model/hypertension_prediction_model.pkl")
    stroke_model = joblib.load("model/stroke_prediction_model.pkl")

    return diabetes_model, hypertension_model, stroke_model

load_dotenv()

EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = os.getenv("EMAIL_PORT")
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_FROM = os.getenv("EMAIL_FROM", EMAIL_USERNAME)

initialize_database()

diabetes_model, hypertension_model, stroke_model = initilize_models()

# ------------------------------------------------
# MAPPING

@app.get("/")
def read_root():
    return {"status": "active", "message": "Health Assessment API is running now"}

@app.post("/submit-form-assessment/")
async def submit_form_assessment(
    # Demographics
    name: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    age: Optional[str] = Form(None),
    MF: Optional[str] = Form(None),
    married: Optional[str] = Form(None),
    residence: Optional[str] = Form(None),
    work_type: Optional[str] = Form(None),
    
    # Lifestyle
    smoking: Optional[str] = Form(None),
    smoking_history: Optional[str] = Form(None),
    bmi: Optional[str] = Form(None),
    
    # Medical History
    hyper: Optional[str] = Form(None),
    heart: Optional[str] = Form(None),
    
    # Cardiovascular
    cp: Optional[str] = Form(None),
    trestbps: Optional[str] = Form(None),
    cholesterol: Optional[str] = Form(None),
    fbs: Optional[str] = Form(None),
    restecg: Optional[str] = Form(None),
    thalach: Optional[str] = Form(None),
    exang: Optional[str] = Form(None),
    oldpeak: Optional[str] = Form(None),
    slope: Optional[str] = Form(None),
    ca: Optional[str] = Form(None),
    thal: Optional[str] = Form(None),
    
    # Glucose
    glucose: Optional[str] = Form(None),
    HbA1c_level: Optional[str] = Form(None)
):
    try:

        #Set the timezone for Kuala Lumpur
        kuala_lumpur_tz = pytz.timezone('Asia/Kuala_Lumpur')

        #Get the current local time
        local_time = datetime.now(kuala_lumpur_tz)

        #Format the time as a string(YYYY-MM-DD HH:MM:SS)
        timestamp = local_time.strftime('%Y-%m-%d %H:%M:%S')

        # Connect to the SQLite database
        conn = sqlite3.connect("health_assessments.db")
        cursor = conn.cursor()
        
        # Insert the form data into the database
        cursor.execute('''
        INSERT INTO health_assessments (
            timestamp, name, email, age, MF, married, residence, work_type,
            smoking, smoking_history, bmi,
            hyper, heart,
            cp, trestbps, cholesterol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal,
            glucose, HbA1c_level
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp,name, email, age, MF, married, residence, work_type,
            smoking, smoking_history, bmi,
            hyper, heart,
            cp, trestbps, cholesterol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal,
            glucose, HbA1c_level
        ))
        
        conn.commit()
        record_id = cursor.lastrowid
        conn.close()
        
        return {"status": "success", "message": "Assessment data saved successfully", "record_id": record_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving assessment data: {str(e)}")

@app.get("/assessments/")
async def get_assessments():
    try:
        conn = sqlite3.connect("health_assessments.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM health_assessments ORDER BY timestamp DESC")
        records = cursor.fetchall()

        # Convert the records to a list of dictionaries
        result = [dict(record) for record in records]
        conn.close()
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving assessments: {str(e)}")

@app.get("/assessment/{assessment_id}")
async def get_assessment(assessment_id: int):
    try:
        conn = sqlite3.connect("health_assessments.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM health_assessments WHERE id = ?", (assessment_id,))
        record = cursor.fetchone()
        conn.close()

        if not record:
            raise HTTPException(status_code=404, detail="Assessment not found")

        if record:
            record_dict = dict(record)  
            predictions = predict(record_dict)  

            return {"record": record_dict, "predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving assessment: {str(e)}")
    
@app.delete("/admin/assessment/{assessment_id}")
async def delete_assessment(assessment_id: int):
    try:
        conn = sqlite3.connect("health_assessments.db")
        cursor = conn.cursor()

        # Check if the record exists first
        cursor.execute("SELECT * FROM health_assessments WHERE id = ?", (assessment_id,))
        record = cursor.fetchone()

        if not record:
            raise HTTPException(status_code=404, detail="Assessment not found")

        # Delete the assessment
        cursor.execute("DELETE FROM health_assessments WHERE id = ?", (assessment_id,))
        conn.commit()
        conn.close()

        return {"status": "success", "message": "Assessment deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting assessment: {str(e)}")


@app.post("/admin/login/")
async def admin_login(username: str = Form(...), password: str = Form(...)):
    try:
        conn = sqlite3.connect("health_assessments.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM admin_users WHERE username = ? AND password = ?", (username, password))
        user = cursor.fetchone()
        
        conn.close()
        
        if user:
            return {"status": "success", "message": "Login successful"}
        else:
            raise HTTPException(status_code=401, detail="Invalid username or password")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error logging in: {str(e)}")


@app.post("/predict/")
def predict(record):
    """Make predictions using the three trained models."""
    df = preprocess_data(record)

    diabetes_input = df[["MF", "age", "hyper", "heart", "smoking_history", "bmi", "HbA1c_level", "glucose"]].to_numpy()
    
    hypertension_input = df[["age", "MF", "cp", "trestbps", "cholesterol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]].to_numpy()

    stroke_input = df[["MF", "age", "hyper", "heart", "married", "work_type", "residence", "glucose", "bmi", "smoking"]].to_numpy()


    diabetes_pred = diabetes_model.predict(diabetes_input)
    hypertension_pred = hypertension_model.predict(hypertension_input)
    stroke_pred = stroke_model.predict(stroke_input)

    return {
        "diabetes_prediction": "Diabetic" if diabetes_pred == 1 else "Non-Diabetic",
        "hypertension_prediction": "Has Hypertension" if hypertension_pred == 1 else "No Hypertension",
        "stroke_prediction": "At Risk of Stroke" if stroke_pred == 1 else "Low Risk"
    }

def format_value(value):
    if not value:
        return "Not specified"
    
    parts = value.split(',')
    return parts[1] if len(parts) > 1 else value

def generate_email_html(assessment):
    """Generate HTML email content similar to frontend template"""
    record = assessment["record"]
    predictions = assessment["predictions"]
    
    name = format_value(record.get("name", ""))
    age = format_value(record.get("age", ""))
    sex = format_value(record.get("MF", ""))
    
    # Color coding based on prediction results
    diabetes_color = "#dc3545" if predictions["diabetes_prediction"] == "Diabetic" else "#198754"
    hypertension_color = "#dc3545" if predictions["hypertension_prediction"] == "Has Hypertension" else "#198754"
    stroke_color = "#ffc107" if predictions["stroke_prediction"] == "At Risk of Stroke" else "#198754"
    
    # Generate HTML content - matches the frontend template
    email_content = f"""
    <div style="font-family: Arial, sans-serif;">
        <h3>Your Health Assessment Report is Ready</h3>
        
        <p>Dear {name},</p>
        
        <p>We're pleased to inform you that your recent health assessment has been processed and your results are now available. Below is a summary of your assessment:</p>
        
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; font-family: Arial, sans-serif; color: #212529;">
            <h4 style="color: #0d6efd; margin-bottom: 10px;">Assessment Summary</h4>
            <p style="margin: 5px 0;"><strong>Assessment ID:</strong> #{record.get("id")}</p>
            <p style="margin: 5px 0;"><strong>Date:</strong> {datetime.fromisoformat(record.get("timestamp")).strftime('%Y-%m-%d')}</p>
            <p style="margin: 5px 0;"><strong>Age:</strong> {age}</p>
            <p style="margin: 5px 0;"><strong>Sex:</strong> {sex}</p>

            <h5 style="margin-top: 15px; color: #0d6efd;">Result</h5>
            <!-- Prediction Results -->
            <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                <div style="flex: 1; text-align: center; margin: 0 5px; padding: 10px; border: 1px solid #dee2e6; border-radius: 5px; background-color: #ffffff;">
                    <h6 style="margin-bottom: 10px; color: #495057;">Diabetes Prediction</h6>
                    <p style="font-size: 1.5rem; color: {diabetes_color};">
                        {predictions["diabetes_prediction"]}
                    </p>
                </div>
                <div style="flex: 1; text-align: center; margin: 0 5px; padding: 10px; border: 1px solid #dee2e6; border-radius: 5px; background-color: #ffffff;">
                    <h6 style="margin-bottom: 10px; color: #495057;">Hypertension Prediction</h6>
                    <p style="font-size: 1.5rem; color: {hypertension_color};">
                        {predictions["hypertension_prediction"]}
                    </p>
                </div>
                <div style="flex: 1; text-align: center; margin: 0 5px; padding: 10px; border: 1px solid #dee2e6; border-radius: 5px; background-color: #ffffff;">
                    <h6 style="margin-bottom: 10px; color: #495057;">Stroke Prediction</h6>
                    <p style="font-size: 1.5rem; color: {stroke_color};">
                        {predictions["stroke_prediction"]}
                    </p>
                </div>
            </div>
        </div>
        
        <div style="background-color: #e9f7fb; padding: 15px; border-radius: 5px; margin: 15px 0;">
            <h4>Next Steps</h4>
            <p>We recommend discussing these results with your healthcare provider during your next appointment. They can provide personalized guidance based on your complete health profile.</p>
            
            <p>You can view your complete assessment report by logging into your patient portal or by clicking the button below:</p>
            
            <div style="text-align: center; margin: 20px 0;">
                <a href="http://localhost:5500/result.html?id={record.get('id')}" target="_blank" style="background-color: #0d6efd; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold;">
                    View Full Report
                </a>
            </div>
        </div>
        
        <p>If you have any questions about your assessment results, please don't hesitate to contact our healthcare team.</p>
        
        <p>Best regards,<br>
        Health Assessment Team</p>
        
        <div style="font-size: 12px; color: #6c757d; margin-top: 30px; padding-top: 15px; border-top: 1px solid #dee2e6;">
            <p>This is an automated notification. Please do not reply to this email. If you need assistance, please contact our support team at support@healthassessment.com or call +60 123456789.</p>
            <p>CONFIDENTIALITY NOTICE: This email contains information that may be confidential or privileged and is intended solely for the individual or entity named above. If you are not the intended recipient, please notify the sender immediately and delete this message.</p>
        </div>
    </div>
    """
    
    return email_content


@app.post("/send-email/")
async def send_email(request: EmailRequest):
    """
    Send assessment results email to the specified email address
    """
    try:
        assessment = await get_assessment(request.assessmentId)
        email_html = generate_email_html(assessment)
        
        # Send email
        subject = f"Your Health Assessment Results - ID #{request.assessmentId}"
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = assessment["record"]["email"]
        msg['Subject'] = subject
        msg.attach(MIMEText(email_html, 'html'))

        # Connect to the SMTP server and send the email
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.send_message(msg)
        print(f"Email sent successfully to {assessment['record']['email']}")

        return {"success": True, "message": "Email sent successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending email: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)