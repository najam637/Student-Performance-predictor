import json
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(title="Student Performance Predictor")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

reg_model = joblib.load("models/regression_model.joblib")
clf_model = joblib.load("models/classification_model.joblib")
info = json.load(open("models/model_info.json"))
test_students = pd.read_csv("models/test_students.csv")
FEATURES = list(info["features"])


class Student(BaseModel):
    StudyHours: int = Field(ge=5, le=44)
    Attendance: int = Field(ge=60, le=100)
    Age: int = Field(ge=18, le=29)
    Gender: int = Field(ge=0, le=1)
    LearningStyle: int = Field(ge=0, le=3)
    OnlineCourses: int = Field(ge=0, le=20)
    Discussions: int = Field(ge=0, le=1)
    AssignmentCompletion: int = Field(ge=50, le=100)
    EduTech: int = Field(ge=0, le=1)
    StressLevel: int = Field(ge=0, le=2)


def to_frame(student: Student):
    return pd.DataFrame([student.model_dump()])[FEATURES]


@app.post("/predict/regression")
def predict_regression(student: Student):
    score = reg_model.predict(to_frame(student))[0]
    return {"predicted_exam_score": round(float(score), 1)}


@app.post("/predict/classification")
def predict_classification(student: Student):
    chances = clf_model.predict_proba(to_frame(student))[0]
    best = int(clf_model.classes_[chances.argmax()])
    return {
        "predicted_grade": best,
        "score_range": info["grade_bands"][str(best)],
        "probabilities": {str(int(c)): round(float(p), 3) for c, p in zip(clf_model.classes_, chances)},
    }


@app.get("/dataset/info")
def dataset_info():
    return info


@app.get("/dataset/sample")
def dataset_sample():
    return test_students.sample(1).iloc[0].astype(int).to_dict()


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("static/index.html")
