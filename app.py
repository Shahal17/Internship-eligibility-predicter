import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title='Internship Eligibility Predictor', page_icon='🎯', layout='wide')
st.title('🎯 Internship Eligibility Predictor')
st.write('Estimate a YES/NO outcome from historical student data and get practical areas to improve.')
st.info('Learning/demo model only. The example dataset predicts PlacementStatus, which is a proxy—not ground-truth internship eligibility.')
uploaded = st.file_uploader('Upload your labelled student CSV dataset', type=['csv'])
if uploaded is None:
    st.markdown('### Getting started')
    st.write('Upload placementdata.csv to train the model. The CSV must include student feature columns and a labelled outcome column such as PlacementStatus.')
    st.stop()
try:
    df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f'Could not read this CSV: {exc}')
    st.stop()
if len(df) < 20 or df.shape[1] < 2:
    st.error('Please upload a dataset with at least 20 rows and two columns.')
    st.stop()
st.caption(f'Dataset loaded: {len(df):,} rows × {len(df.columns)} columns')
target = st.selectbox('Choose the outcome/target column', df.columns, index=(list(df.columns).index('PlacementStatus') if 'PlacementStatus' in df.columns else len(df.columns)-1))
id_candidates = [c for c in df.columns if c.lower() in {'studentid', 'student_id', 'id', 'rollno', 'roll_no'}]
drop_cols = st.multiselect('Columns to exclude (IDs or irrelevant fields)', [c for c in df.columns if c != target], default=id_candidates)
data = df.dropna(subset=[target]).copy()
feature_cols = [c for c in data.columns if c != target and c not in drop_cols]
if not feature_cols:
    st.error('Keep at least one feature column.')
    st.stop()
X = data[feature_cols].copy()
y = data[target].astype(str).str.strip()
classes = sorted(y.unique().tolist())
if len(classes) < 2 or len(classes) > 20:
    st.error('Target must contain between 2 and 20 distinct outcome classes.')
    st.stop()
def_yes = next((c for c in classes if c.lower() in {'placed','yes','eligible','selected','true','1'}), classes[-1])
yes_class = st.selectbox('Which target value should mean YES?', classes, index=classes.index(def_yes))
valid_features = [c for c in feature_cols if not X[c].isna().all()]
X = X[valid_features]
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]
preprocessor = ColumnTransformer([('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric_cols), ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)])
model_choice = st.selectbox('Model', ['Random Forest', 'Logistic Regression'])
classifier = RandomForestClassifier(n_estimators=250, class_weight='balanced', random_state=42) if model_choice == 'Random Forest' else LogisticRegression(max_iter=2000, class_weight='balanced')
pipeline = Pipeline([('preprocess', preprocessor), ('model', classifier)])
try:
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
except Exception as exc:
    st.error(f'Training failed. Check target/features. Details: {exc}')
    st.stop()
st.subheader('Model evaluation on held-out test data')
a,b,c,d = st.columns(4)
a.metric('Accuracy', f'{accuracy_score(y_test,y_pred):.1%}')
b.metric('Precision (weighted)', f'{precision_score(y_test,y_pred,average="weighted",zero_division=0):.1%}')
c.metric('Recall (weighted)', f'{recall_score(y_test,y_pred,average="weighted",zero_division=0):.1%}')
d.metric('F1 score (weighted)', f'{f1_score(y_test,y_pred,average="weighted",zero_division=0):.1%}')
st.caption('A random hold-out split does not guarantee performance on other colleges, employers, or future applicants.')
st.subheader('Check a student profile')
with st.form('profile_form'):
    inputs = {}
    left,right = st.columns(2)
    for i,col in enumerate(valid_features):
        series = X[col]
        container = left if i%2 == 0 else right
        with container:
            if col in numeric_cols:
                vals = pd.to_numeric(series, errors='coerce').dropna()
                default = float(vals.median()) if not vals.empty else 0.0
                min_v = float(vals.min()) if not vals.empty else 0.0
                max_v = float(vals.max()) if not vals.empty else max(1.0,default)
                if min_v == max_v: max_v = min_v+1.0
                inputs[col] = st.number_input(col, min_value=min_v, max_value=max_v, value=min(max(default,min_v),max_v))
            else:
                choices = sorted(series.dropna().astype(str).unique().tolist())
                inputs[col] = st.selectbox(col, choices if choices else ['Unknown'])
    submitted = st.form_submit_button('Predict eligibility')
if submitted:
    row = pd.DataFrame([{c:inputs[c] for c in valid_features}])
    prediction = str(pipeline.predict(row)[0])
    probabilities = pipeline.predict_proba(row)[0] if hasattr(pipeline,'predict_proba') else None
    if prediction == yes_class: st.success(f'YES — model predicts: {prediction}')
    else: st.error(f'NO — model predicts: {prediction}')
    if probabilities is not None:
        fitted_classes = pipeline.named_steps['model'].classes_
        confidence = float(probabilities[list(fitted_classes).index(prediction)])
        st.write(f'Model confidence for predicted class: **{confidence:.1%}** (not a calibrated probability of getting an internship).')
    suggestions=[]
    cgpa = next((c for c in valid_features if c.lower() in {'cgpa','gpa'}),None)
    internships = next((c for c in valid_features if 'internship' in c.lower()),None)
    projects = next((c for c in valid_features if 'project' in c.lower()),None)
    aptitude = next((c for c in valid_features if 'aptitude' in c.lower() or 'testscore' in c.lower()),None)
    soft = next((c for c in valid_features if 'softskill' in c.lower()),None)
    cert = next((c for c in valid_features if 'workshop' in c.lower() or 'certification' in c.lower()),None)
    training = next((c for c in valid_features if 'training' in c.lower()),None)
    if cgpa and float(inputs[cgpa]) < 7.5: suggestions.append('Strengthen core subjects and aim to improve CGPA; use the target company’s actual cutoff as the benchmark.')
    if projects and float(inputs[projects]) < 2: suggestions.append('Build 2–3 complete projects, publish clear READMEs, and include a demo plus measurable results.')
    if internships and float(inputs[internships]) < 1: suggestions.append('Gain practical experience through an internship, open-source contribution, research project, or verified client project.')
    if aptitude and float(inputs[aptitude]) < 75: suggestions.append('Practice aptitude, coding, and role-specific assessments weekly; track accuracy and time taken.')
    if soft and float(inputs[soft]) < 3.5: suggestions.append('Practice communication, teamwork, interview storytelling, and explaining technical decisions.')
    if cert and float(inputs[cert]) < 1: suggestions.append('Complete a relevant course and demonstrate the skill in a project.')
    if training and str(inputs[training]).lower() in {'no','false','0'}: suggestions.append('Try mock interviews, resume reviews, and structured preparation.')
    if prediction != yes_class: suggestions.append('Compare your profile with the target role’s published requirements and close the biggest skill gap first.')
    if not suggestions: suggestions=['Keep building role-relevant projects, document impact, tailor your resume, and practice interviews.']
    st.subheader('Areas to improve')
    for suggestion in dict.fromkeys(suggestions): st.write('- '+suggestion)
    st.warning('Suggestions are simple heuristics, not causal explanations of the model. Do not use this result as a final hiring decision.')
with st.expander('Dataset overview'):
    st.write('Target distribution')
    st.bar_chart(y.value_counts())
    st.dataframe(df.head(10), use_container_width=True)
