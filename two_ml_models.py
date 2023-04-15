from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from keras.metrics import AUC
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import pandas as pd


class CoverTypeClassifierRFLR:
    def __init__(self, data_file_path):
        self.data = pd.read_csv(data_file_path, header=None)
        self.data = self.data.iloc[:1000, :]

        columns = [
            'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
            'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
            'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
            'Horizontal_Distance_To_Fire_Points'
        ]
        wilderness_areas = [f'Wilderness_Area_{i}' for i in range(1, 5)]
        soil_types = [f'Soil_Type_{i}' for i in range(1, 41)]
        columns.extend(wilderness_areas)
        columns.extend(soil_types)
        columns.append('Cover_Type')
        self.data.columns = columns

        self.X = self.data.drop('Cover_Type', axis=1)
        self.y = self.data['Cover_Type']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)

        self.scaler = StandardScaler()
        self.scaler.fit(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        self.random_forest = RandomForestClassifier()
        self.random_forest.fit(self.X_train, self.y_train)

        self.logistic_reg = LogisticRegression()
        self.logistic_reg.fit(self.X_train, self.y_train)

    def get_random_forest_accuracy(self):
        forest_pred = self.random_forest.predict(self.X_test)
        forest_acc = accuracy_score(self.y_test, forest_pred)
        # forest_f1 = f1_score(self.y_test, forest_pred)
        return forest_acc

    def get_logistic_regression_accuracy(self):
        logistic_reg_pred = self.logistic_reg.predict(self.X_test)
        logistic_reg_acc = accuracy_score(self.y_test, logistic_reg_pred)
        # logistic_reg_f1 = f1_score(self.y_test, logistic_reg_pred)
        return logistic_reg_acc

    def predict_cover_type(self, sample, pick):
        sample = np.asarray(sample)
        sample = sample.reshape(1, -1)
        sample = self.scaler.transform(sample)
        if pick == "RF":
            predicted_cover_type = self.random_forest.predict(sample)
            return predicted_cover_type[0]
        elif pick == "LR":
            predicted_cover_type = self.logistic_reg.predict(sample)
            return predicted_cover_type[0]

# idx = random.randint(0, len(X_test) - 1)
# sample = X_test[idx]
# sample = sample.reshape(1, -1)
# sample = scaler.transform(sample)
# predicted_cover_type = random_forest.predict(sample)
# print(f"Value: {y_test.iloc[idx]}, Predicted: {predicted_cover_type[0]}")

# sample_pred = [
#     2596, 51, 3, 258, 0, 510, 221, 232, 148, 6279,
#     1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#     0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
# ]
# print("INSIDE LOGISTIC REGRESSION 1")
# classifier = CoverTypeClassifierRFLR(data_file_path='covtype.data')
# print("INSIDE LOGISTIC REGRESSION 2")
# logistic_reg_acc = classifier.get_logistic_regression_accuracy()
# print("INSIDE LOGISTIC REGRESSION 3")
# predicted_cover_type = classifier.predict_cover_type(sample_pred, "LR")
# print("INSIDE LOGISTIC REGRESSION 4")
# predicted_cover_type = np.int64(predicted_cover_type).tolist()
# print("INSIDE LOGISTIC REGRESSION 5")
# output_json = {"Logistic Regression Accuracy:": logistic_reg_acc,
#                "Predict 'Cover_type' value for sample - Logistic Regression": predicted_cover_type}
# print(output_json)