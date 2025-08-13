import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

# Page config
st.set_page_config(page_title="DataDuel", layout="wide")

# Custom CSS for theming
st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
    }
    .title-bar {
        background-color: #1f4e79;
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .best-score {
        background-color: #d4edda;
        color: #155724;
        font-weight: bold;
        border-radius: 5px;
        padding: 2px 5px;
    }
    .footer {
        margin-top: 40px;
        text-align: center;
        color: #777;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# Title bar
st.markdown('<div class="title-bar">⚔️ DataDuel - Model Performance Comparison</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📂 Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Preview of Dataset")
    st.dataframe(df.head())

    target = st.sidebar.selectbox("🎯 Select Target Column", df.columns)
    exclude_cols = st.sidebar.multiselect("🚫 Columns to Exclude", [col for col in df.columns if col != target])

    if st.sidebar.button("⚡ Compare Models"):
        try:
            X = df.drop(columns=[target] + exclude_cols)
            y = df[target]

            # Encode categorical variables
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = LabelEncoder().fit_transform(X[col])
            if y.dtype == 'object':
                y = LabelEncoder().fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Detect task type
            is_classification = len(pd.Series(y).unique()) <= 20 and y.dtype != 'float64'

            results = []

            if is_classification:
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Random Forest Classifier": RandomForestClassifier()
                }
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    results.append({"Model": name, "Score": acc})
            else:
                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest Regressor": RandomForestRegressor()
                }
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    score = r2_score(y_test, y_pred)
                    results.append({"Model": name, "Score": score})

            results_df = pd.DataFrame(results)
            best_score = results_df["Score"].max()

            # Highlight best score
            def highlight_best(val):
                return 'background-color: #d4edda; color: #155724; font-weight: bold;' if val == best_score else ''

            st.subheader("🏆 Model Comparison Results")
            st.write("Here’s how the models performed on your dataset:")
            st.dataframe(results_df.style.applymap(highlight_best, subset=['Score']))

        except Exception as e:
            st.error(f"Error: {e}")

# Footer
st.markdown('<div class="footer">Powered by DataDuel</div>', unsafe_allow_html=True)
