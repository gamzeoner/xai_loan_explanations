import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import shap


data = pd.read_csv("credit_data.csv")

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #split data

model = LogisticRegression() # train model
model.fit(X_train, y_train)

explainer = shap.Explainer(model, X_train) #SHAP
shap_values = explainer(X_test)

shap.plots.waterfall(shap_values[0])
sample = X_test.iloc[0].copy()


print(model.predict([sample]))# prediction bfore change


sample["income"] += 1000 #income change


print(model.predict([sample])) #prediction after change