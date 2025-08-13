from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def preprocess_data(df, target_column, test_size=0.2, random_state=42):

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

def using_diff_models(X_train, X_test, y_train, y_test, scaler):

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
    results[name] = acc

  print(results)

def main():

  print("Welcome to DataDual:\n")
  print("Your one stop for choosing a machine learning model:\n")

  problem_statement = input("What would you like to solve?\n")
  dataset = input("Give ur dataset\n")
  target_column = input("What is the target column?\n")

  X_train, X_test, y_train, y_test, scaler = preprocess_data(dataset, target_column)
  using_diff_models(X_train, X_test, y_train, y_test, scaler)



if __name__ == "__main__":
  main()