import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Set page config
st.set_page_config(page_title="Model Comparison App", layout="wide")

# Custom CSS for better appearance
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    h1 {
        color: #2a3f5f;
    }
    .st-bq {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


def preprocess_data(df, target_column, test_size=0.2, random_state=42):
  """Preprocess the data for modeling"""
  X = df.drop(columns=[target_column])
  y = df[target_column]

  num_cols = X.select_dtypes(include=['int64', 'float64']).columns
  X[num_cols] = X[num_cols].fillna(X[num_cols].mean())

  cat_cols = X.select_dtypes(include=['object', 'category']).columns
  for col in cat_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

  X = pd.get_dummies(X, drop_first=True)

  scaler = StandardScaler()
  X[num_cols] = scaler.fit_transform(X[num_cols])

  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
  )

  return X_train, X_test, y_train, y_test, scaler


def compare_models(X_train, X_test, y_train, y_test):
  """Compare multiple classification models"""
  models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
  }

  results = {}
  for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc * 100  # Convert to percentage

  return results


def main():
  """Main app function"""
  st.title("🚀 Model Comparison App")
  st.markdown("""
    ### Your one-stop solution for comparing machine learning models
    Upload your dataset and we'll compare multiple classification models for you!
    """)

  # File upload section
  uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

  if uploaded_file is not None:
    try:
      df = pd.read_csv(uploaded_file)

      # Show dataset preview
      st.subheader("Dataset Preview")
      st.write(df.head())

      # Get target column
      target_column = st.selectbox(
        "Select the target column",
        options=df.columns,
        index=len(df.columns) - 1
      )

      # Additional options
      st.sidebar.header("Advanced Options")
      test_size = st.sidebar.slider(
        "Test set size",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05
      )
      random_state = st.sidebar.number_input(
        "Random state",
        min_value=0,
        max_value=100,
        value=42
      )

      if st.button("Compare Models"):
        with st.spinner("Preprocessing data and training models..."):
          # Preprocess data
          X_train, X_test, y_train, y_test, _ = preprocess_data(
            df, target_column, test_size, random_state
          )

          # Compare models
          results = compare_models(X_train, X_test, y_train, y_test)

          # Display results
          st.subheader("Model Comparison Results")

          # Create a nice results table
          results_df = pd.DataFrame.from_dict(
            results,
            orient='index',
            columns=['Accuracy (%)']
          ).sort_values('Accuracy (%)', ascending=False)

          st.dataframe(results_df.style.format({'Accuracy (%)': '{:.2f}%'}))

          # Visualize results
          st.subheader("Accuracy Comparison")
          st.bar_chart(results_df)

          # Show best model
          best_model = max(results, key=results.get)
          st.success(f"🎉 Best performing model: {best_model} with {results[best_model]:.2f}% accuracy")

    except Exception as e:
      st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
  main()