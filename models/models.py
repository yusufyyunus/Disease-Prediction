from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from pydantic.networks import EmailStr

class EmailRequest(BaseModel):
    assessmentId: int
    email: EmailStr

class PredictionResult(BaseModel):
    diabetes_prediction: str
    hypertension_prediction: str
    stroke_prediction: str

class AssessmentRecord(BaseModel):
    id: int
    name: Optional[str] = None
    email: Optional[str] = None
    age: Optional[str] = None
    MF: Optional[str] = None  # Sex
    timestamp: Optional[datetime] = None

class EmailResponse(BaseModel):
    success: bool
    message: str

class HealthAssessmentForm(BaseModel):
    name: str = None
    email: str = None
    age: str = None
    MF: str = None
    married: str = None
    residence: str = None
    work_type: str = None
    
    # Lifestyle
    smoking: str = None
    smoking_history: str = None
    bmi: str = None
    
    # Medical History
    hyper: str = None
    heart: str = None
    
    # Cardiovascular
    cp: str = None
    trestbps: str = None
    cholesterol: str = None
    fbs: str = None
    restecg: str = None
    thalach: str = None
    exang: str = None
    oldpeak: str = None
    slope: str = None
    ca: str = None
    thal: str = None
    
    # Glucose
    glucose: str = None
    HbA1c_level: str = None