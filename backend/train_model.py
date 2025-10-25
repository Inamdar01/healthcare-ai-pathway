import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'diabetes.csv')
    model_path = os.path.join(base_dir, 'model.pkl')

    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        sys.exit(1)

    # Load dataset
    data = pd.read_csv(data_path)

    # Features and target
    if 'Outcome' not in data.columns:
        print("ERROR: 'Outcome' column not found in dataset")
        sys.exit(1)

    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save the model to an explicit path so it's easy to find
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"✅ Model trained and saved as: {model_path}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('An error occurred while training/saving the model:')
        import traceback
        traceback.print_exc()
        sys.exit(1)
