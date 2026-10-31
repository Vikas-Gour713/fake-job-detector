from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from fastapi.responses import FileResponse

app = FastAPI()
model = joblib.load('model.pkl')

class Job(BaseModel):
    title: str
    company_profile: str = ""
    description: str
    requirements: str = ""
    employment_type: str = "Unknown"
    required_experience: str = "Unknown"
    required_education: str = "Unknown"
    telecommuting: int
    has_company_logo: int
    has_questions: int
    
@app.get("/")
def home():
    return FileResponse('index.html')

@app.post("/predict")
def predict(job : Job):
    text = f"{job.title} {job.company_profile} {job.description} {job.requirements}"
    row = pd.DataFrame([{
        'text' : text,
        'employment_type' : job.employment_type,
        'required_experience' : job.required_experience,
        'required_education' : job.required_education,
        'telecommuting' : job.telecommuting,    
        'has_company_logo' : job.has_company_logo,
        'has_questions' : job.has_questions
    }])
    prob = float(model.predict_proba(row)[0][1])
    return {'fraudulent' : int(prob >= 0.5), 'probability' : round(prob, 3)}
