import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from datetime import datetime
import os
import requests

from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,mean_squared_error, mean_absolute_error, r2_score

#logger

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

#session state initialization

if "cleaned_saved" not in st.session_state:
    st.session_state.cleaned_saved = False

#folder setup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

#appilaction starting point

log("Application Started")
log(f" raw dir = {RAW_DIR}")
log(f"cleaned dir = {CLEAN_DIR}")

#page config

st.set_page_config("End-to-End SVM", layout = "wide")
st.title("End-to-End SVM Platform")

#sidebar: Model Settings

st.sidebar.header("SVM Settings")
kernel = st.sidebar.selectbox("kernel", ["linear", "rbf", "poly", "sigmoid"])

C = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])

log(f"SVM settings ---> Kernel = {kernel}, c = {C}, Gamma = {gamma}")

#Step 1 : Data Ingestion

st.header("Step 1 : Data Ingestion")
log("step 1 started : Data Ingestion")

option = st.radio("choose Data source ", ["Download Dataset", "Upload CSV"])
df =None 
raw_path = None

if option == "Download Dataset":
    if st.button("Download Iris Dataset"):
        log("Downloading Iris Dataset")
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        response = requests.get(url)

        raw_path = os.path.join(RAW_DIR, "iris.csv")
        with open(raw_path, "wb") as f:
            f.write(response.content)

        df = pd.read_csv(raw_path)
        st.success("Dataset downloaded")
        log(f"Iris Dataset Saved at {raw_path}")

if option == "Upload CSV":
    uploaded_file = st.file_uploader("upload CSV File", type = ["CSV"])
    if uploaded_file:
        raw_path = os.path.join(RAW_DIR, uploaded_file.name)
        with open(raw_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        df = pd.read_csv(raw_path)
        st.sucess("File uploaded succesfully")
        log(f"uploaded data saved at {raw_path}")


# Step 2: EDA

if df is not None : 
    st.header("Step 2 : Exploratory Data Analysis")
    log("Step 2 started : EDA")

    st.dataframe(df.head())
    st.write("Shape", df.shape)
    st.write("Missing Values : ", df.isnull().sum())

    fig,ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    log("EDA COMPLETED")

# Step 3 : Data Cleaning 

if df is not None : 
    st.header("Step 3 : Data Cleaning")
    strategy = st.selectbox(
        "Missing value strategy",
        ["Mean","Median","Drop Rows"]
    )
    df_clean = df.copy()

    if strategy == "Drop Rows":
        df_clean = df_clean.dropna()

    else:
        for col in df_clean.select_dtypes(include = np.number) :
            if strategy == "Mean" :
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            else :
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    st.session_state.df_clean = df_clean
    st.success("Data Cleaning Completed")

else:
    st.info("Please complete Step 1 (Data Ingestion) first...")


#step 4: save cleaned Data

if st.button("Save cleaned Datadet"):
    if st.session_state.df_clean is None:
        st.error("No Cleaned data found. Please Complete step 3 first...")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_filename = f"cleaned_dataset_{timestamp}.csv"
        clean_path = os.path.join(CLEAN_DIR, clean_filename)

        st.session_state.df_clean.to_csv(clean_path,index = False)

        st.success("Cleaned Dataset Saved")
        st.info(f"Saved at : {clean_path}")
        log(f"Cleaned dataset saved at {clean_path}")

#step 5 : Load Cleaned Dataset

st.header("Step 5 : load Cleaned Dataset")

clean_files = os.listdir(CLEAN_DIR)

if not clean_files:

    st.warning("No cleaned Datasets found. please save one in step4...")
    log("no Cleaned datasets available")
else:
    selected = st.selectbox("select Cleaned Dataset", clean_files)
    df_model = pd.read_csv(os.path.join(CLEAN_DIR, selected))

    st.success(f"Loaded datasets : {selected}")
    log(f"Loaded cleaned datasets: {selected}")

    st.dataframe(df_model.head())


# Step 6 : Train SVM

st.header("Step 6 : Train SVM")
log("Step 6 Started : SVM TRaining")

target = st.selectbox("select Target Column", df_model.columns)

y = df_model[target]

if y.dtype == "object":
    y = LabelEncoder().fit_transform(y)
    log("Target columns encoded")

#select numeric features only

x = df_model.drop(columns = [target])
x = x.select_dtypes(include = np.number)

if x.empty:
    st.error("No Numeric features available for the training..")
    st.stop()

#scale features

scaler = StandardScaler()
X = scaler.fit_transform(x)

#train - test split

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)

# training 
model_type = st.selectbox("Select Model", ["Classifer", "Regression"])

if model_type == "Classifer":
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(x_train,y_train)
else:
    model = SVR(kernel=kernel, C=C, gamma=gamma)
    model.fit(x_train,y_train)

#evaluate

if model_type == "Classifer":
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"Accuracy : {acc:.2f}")
    log(f"SVM training successfully | Accuracy = {acc : .2f}")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)
else:
    y_pred = model.predict(x_test)

    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.success(f"R² Score : {r2:.2f}")
    st.write(f"MAE : {mae:.2f}")
    st.write(f"MSE : {mse:.2f}")
    st.write(f"RMSE : {rmse:.2f}")

    log(f"SVR training successfully | R² = {r2:.2f}, RMSE = {rmse:.2f}")

    # Plot: Actual vs Predicted
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7, color="blue")
    ax.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--", lw=2)  # diagonal line
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("SVR: Actual vs Predicted")
    st.pyplot(fig)