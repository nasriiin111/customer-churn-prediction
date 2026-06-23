import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, recall_score

df=pd.read_csv('telecom_customer_churn_dataset.csv') 
print(df.head())

#checking for null values
df.isnull().sum()

#Removing unnecessary columns
df.drop(['customerID'], axis=1, inplace=True)

#Target variable
y = df["Churn"].map({"No": 0, "Yes": 1})

#Feature variables
X = df.drop("Churn", axis=1)

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#Training the model and preditions
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

#Evaluation
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test, predictions))

