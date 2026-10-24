# Student Performance Predictor

A small full-stack machine learning app. Enter a student's study habits and get a predicted **exam score** (regression) or **final grade** (classification).

## What's inside

| File | Purpose |
|---|---|
| `train_models.py` | Trains both models with scikit-learn, prints the scores, saves them with joblib |
| `main.py` | FastAPI backend with the two prediction endpoints |
| `static/index.html` | The web page (one form per model) |
| `data/student_performance.csv` | The dataset |
| `models/` | Saved models and the info the website reads |

## How to run

```bash
pip install -r requirements.txt
python train_models.py
python -m uvicorn main:app
```

Then open http://127.0.0.1:8000 in your browser. API docs are at http://127.0.0.1:8000/docs.

## API

- `POST /predict/regression` returns `{"predicted_exam_score": 73.8}`
- `POST /predict/classification` returns the predicted grade, its score range and the chance of each grade
- `GET /dataset/info` returns feature ranges and model scores (the page builds its sliders from this)
- `GET /dataset/sample` returns a random real student for the "Try a real student" button

Both POST endpoints take the same JSON body:

```json
{"StudyHours": 20, "Attendance": 80, "Age": 24, "Gender": 1, "LearningStyle": 2,
 "OnlineCourses": 10, "Discussions": 1, "AssignmentCompletion": 74, "EduTech": 1, "StressLevel": 2}
```

## Models and results

Both models are Extra Trees (scikit-learn), trained on 80% of the data and tested on the other 20%.

| Model | Predicts | Result |
|---|---|---|
| Regression | `ExamScore` | MAE 1.91 marks, R² 0.872 |
| Classification | `FinalGrade` | Accuracy 90.8% |

## Notes on the data

- 1,534 duplicate rows were removed before training.
- `Resources`, `Motivation`, `Extracurricular` and `Internet` were left out. The exam score never changes when only these columns change, so they only add noise.
- The dataset holds about 4,200 unique student profiles, each repeated several times. The scores above are measured on held-out rows, but similar profiles also appear in the training data. For a profile unlike any in the dataset, the typical error is about 16 marks, close to guessing the average. Real-world accuracy would be lower than the table above.
- The CSV does not explain what codes like `Gender` or `LearningStyle` mean, so the page shows neutral labels. They can be renamed at the top of the script in `static/index.html`.