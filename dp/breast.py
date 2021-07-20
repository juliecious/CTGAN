from sklearn.datasets import load_breast_cancer
import pandas as pd


data = load_breast_cancer()
data = pd.DataFrame(data.data, columns=data.feature_names)

