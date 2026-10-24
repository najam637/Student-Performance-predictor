import json
import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

FEATURES = ["StudyHours", "Attendance", "Age", "Gender", "LearningStyle",
            "OnlineCourses", "Discussions", "AssignmentCompletion",
            "EduTech", "StressLevel"]

# Load the dataset and remove repeated rows
data = pd.read_csv("data/student_performance.csv").drop_duplicates()
X = data[FEATURES]

# Resources, Motivation, Extracurricular and Internet are left out:
# the exam score never changes when only those columns change

# 80% to train, 20% to test
X_train, X_test, y_train, y_test = train_test_split(
    X, data[["ExamScore", "FinalGrade"]], test_size=0.2, random_state=42)

# Model 1: predicts the exam score (a number)
reg = ExtraTreesRegressor(n_estimators=100, random_state=0, n_jobs=-1)
reg.fit(X_train, y_train["ExamScore"])
pred = reg.predict(X_test)
mae = mean_absolute_error(y_test["ExamScore"], pred)
r2 = r2_score(y_test["ExamScore"], pred)

# Model 2: predicts the final grade (a category)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0, n_jobs=-1)
clf.fit(X_train, y_train["FinalGrade"])
acc = accuracy_score(y_test["FinalGrade"], clf.predict(X_test))

print(f"Regression      -> MAE: {mae:.2f} marks   R2: {r2:.3f}")
print(f"Classification  -> Accuracy: {acc:.1%}")

# Honest check: test on student profiles the model has never seen
groups = data.groupby(FEATURES).ngroup()
train_idx, test_idx = next(GroupShuffleSplit(test_size=0.2, random_state=0).split(data, groups=groups))
check = ExtraTreesRegressor(n_estimators=100, random_state=0, n_jobs=-1)
check.fit(X.iloc[train_idx], data["ExamScore"].iloc[train_idx])
new_mae = mean_absolute_error(data["ExamScore"].iloc[test_idx], check.predict(X.iloc[test_idx]))
print(f"On brand-new profiles the MAE is {new_mae:.2f} marks")

# Save the models and the info the website needs
joblib.dump(reg, "models/regression_model.joblib", compress=3)
joblib.dump(clf, "models/classification_model.joblib", compress=3)

bands = data.groupby("FinalGrade")["ExamScore"].agg(["min", "max"])
info = {
    "regression": {"mae": round(mae, 2), "r2": round(r2, 3)},
    "classification": {"accuracy": round(acc, 3)},
    "new_profile_mae": round(new_mae, 2),
    "grade_bands": {str(g): [int(r["min"]), int(r["max"])] for g, r in bands.iterrows()},
    "features": {f: {"min": int(data[f].min()), "max": int(data[f].max()),
                     "default": int(data[f].median())} for f in FEATURES},
}
json.dump(info, open("models/model_info.json", "w"), indent=2)

# Test students are used by the "Try a real student" button
X_test.join(y_test).to_csv("models/test_students.csv", index=False)
print("Models saved in the models/ folder")
