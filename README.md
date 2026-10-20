# Internship Eligibility Predictor

An interactive machine-learning project that predicts a **YES/NO outcome** from labelled student data and provides practical areas to improve.

> **Important:** The supplied placementdata.csv labels students as Placed or NotPlaced. The model therefore predicts a placement outcome, not actual eligibility for a specific internship. Use a properly labelled internship eligibility dataset and the target company's real requirements for a genuine eligibility model. This demo must not be used to make real hiring decisions.

## Features

- Upload a labelled CSV dataset through the app.
- Select the target column and choose which label means YES.
- Exclude identifiers such as StudentID to avoid data leakage.
- Train Random Forest or Logistic Regression.
- Evaluate with accuracy, precision, recall, and F1 on held-out test data.
- Enter a student's profile and receive a YES/NO prediction.
- Get practical suggestions for common gaps: projects, internships, aptitude, communication, certifications, and preparation.
- View target distribution and dataset preview.

## Dataset format

The example dataset has 10,000 rows and includes CGPA, Internships, Projects, Workshops/Certifications, AptitudeTestScore, SoftSkillsRating, ExtracurricularActivities, PlacementTraining, SSC_Marks, HSC_Marks, and PlacementStatus (target).

Upload your own CSV in the app. Keep one row per student, include a labelled target column, and remove personal identifiers. Do not include names, email addresses, phone numbers, or other sensitive data.

## Run locally

Python 3.10+ recommended.

```bash
git clone https://github.com/Shahal17/Internship-eligibility-predicter.git
cd Internship-eligibility-predicter
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\\Scripts\\Activate.ps1
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies and start the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload placementdata.csv when the app opens.

## Deploy free with Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open https://share.streamlit.io/.
3. Connect GitHub and choose this repository.
4. Set the main file path to app.py.
5. Deploy, then upload the CSV in the app.

## Model and limitations

- The app uses a scikit-learn pipeline with imputation, scaling for numeric columns, and one-hot encoding for categorical columns.
- A random 80/20 train/test split is used. Results may not generalize to other institutions or hiring cycles.
- Accuracy alone is not enough; review precision, recall, F1, class balance, and false-positive/false-negative costs.
- Improvement suggestions are transparent heuristics, not causal explanations of predictions.
- Prediction confidence should not be interpreted as proof that changing a feature will cause a better outcome.
- Historical student datasets can contain bias. Do not use protected attributes or the prediction as an automated hiring decision.

## Recommended next improvements

1. Add cross-validation and hyperparameter tuning.
2. Compare models and show a confusion matrix.
3. Add permutation importance or SHAP for more careful explanations.
4. Validate data quality and check class imbalance.
5. Build a dedicated dataset labelled with real internship eligibility outcomes.
6. Add automated tests and continuous integration.
7. Add a sample anonymized CSV only after confirming its provenance and redistribution licence.

## Tech stack

Python · Pandas · NumPy · scikit-learn · Streamlit
