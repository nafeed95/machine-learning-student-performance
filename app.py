import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓")

st.title("🎓 Student Performance Prediction")
st.write("Predict whether a student will pass or fail using ML models.")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("student_data.csv")
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    df['pass'] = (df['G3'] >= 10).astype(int)
    df.drop(columns=['G1', 'G2', 'G3'], inplace=True)
    return df

df = load_data()
X = df.drop(columns=['pass'])
y = df['pass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(X_test))

# Sidebar
st.sidebar.header("📊 Model Accuracy")
st.sidebar.success(f"Logistic Regression: {lr_acc*100:.2f}%")
st.sidebar.info(f"Decision Tree: {dt_acc*100:.2f}%")

st.sidebar.header("🔧 Select Model")
model_choice = st.sidebar.radio("Choose Model", ["Logistic Regression", "Decision Tree"])

st.header("📝 Enter Student Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 15, 22, 17)
    studytime = st.selectbox("Study Time (hrs/week)", [1, 2, 3, 4], index=1)
    failures = st.selectbox("Past Failures", [0, 1, 2, 3])
    absences = st.slider("Absences", 0, 93, 5)

with col2:
    Medu = st.selectbox("Mother Education (0-4)", [0, 1, 2, 3, 4], index=2)
    Fedu = st.selectbox("Father Education (0-4)", [0, 1, 2, 3, 4], index=2)
    famrel = st.slider("Family Relationship (1-5)", 1, 5, 4)
    freetime = st.slider("Free Time (1-5)", 1, 5, 3)

with col3:
    goout = st.slider("Going Out (1-5)", 1, 5, 3)
    Dalc = st.slider("Workday Alcohol (1-5)", 1, 5, 1)
    Walc = st.slider("Weekend Alcohol (1-5)", 1, 5, 2)
    health = st.slider("Health (1-5)", 1, 5, 3)

# Fill remaining columns with median
input_data = pd.DataFrame([X_train.median()], columns=X.columns)
input_data['age'] = age
input_data['studytime'] = studytime
input_data['failures'] = failures
input_data['absences'] = absences
input_data['Medu'] = Medu
input_data['Fedu'] = Fedu
input_data['famrel'] = famrel
input_data['freetime'] = freetime
input_data['goout'] = goout
input_data['Dalc'] = Dalc
input_data['Walc'] = Walc
input_data['health'] = health

if st.button("🔮 Predict"):
    model = lr if model_choice == "Logistic Regression" else dt
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    st.header("📈 Prediction Result")
    if prediction == 1:
        st.success(f"✅ Student Will **PASS** with {proba[1]*100:.1f}% confidence!")
    else:
        st.error(f"❌ Student Will **FAIL** with {proba[0]*100:.1f}% confidence!")

    st.progress(int(proba[1]*100))
    st.write(f"Pass Probability: **{proba[1]*100:.1f}%** | Fail Probability: **{proba[0]*100:.1f}%**")
