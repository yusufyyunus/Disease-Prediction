import sqlite3
import pandas as pd
import joblib
import numpy as np

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
    diabetes_model = joblib.load("../model/diabetes_prediction_model.pkl")
    hypertension_model = joblib.load("../model/hypertension_prediction_model.pkl")
    stroke_model = joblib.load("../model/stroke_prediction_model.pkl")

    return diabetes_model, hypertension_model, stroke_model