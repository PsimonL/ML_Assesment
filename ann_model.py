import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler


class CoverTypeClassifierNN:

    def __init__(self, data_file_path):
        self.data = pd.read_csv(data_file_path, header=None)
        self.columns = [
            'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
            'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
            'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
            'Horizontal_Distance_To_Fire_Points'
        ]
        self.wilderness_areas = [f'Wilderness_Area_{i}' for i in range(1, 5)]
        self.soil_types = [f'Soil_Type_{i}' for i in range(1, 41)]
        self.columns.extend(self.wilderness_areas)
        self.columns.extend(self.soil_types)
        self.columns.append('Cover_Type')
        self.data.columns = self.columns
        self.cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                     'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                     'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                     'Horizontal_Distance_To_Fire_Points', 'Cover_Type']
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()

    def outliers(self):
        # # Interquartile Range (IQR)
        # Q1 = self.data.quantile(0.25)
        # Q3 = self.data.quantile(0.75)
        # IQR = Q3 - Q1
        # data = self.data[~((self.data < (Q1 - 1.5 * IQR)) | (self.data > (Q3 + 1.5 * IQR))).any(axis=1)]

        # Calculate the z-scores of each column - measure that represents the number of standard deviations a data point is from the mean of the dataset
        z_scores = (self.data - self.data.mean()) / self.data.std()
        # Set the threshold for the z-score
        threshold = 3
        # Remove any rows where the z-score is greater than the threshold
        self.data = self.data[(np.abs(z_scores) < threshold).all(axis=1)]

    def plot_boxplots(self):
        for i, column in enumerate(self.cols):
            fig, ax = plt.subplots()
            sns.boxplot(data=self.data[column], ax=ax).set(title=f"{column.upper()} boxplot", xlabel=f"{column}",
                                                           ylabel=f"Value of {column}")
            ax.set_title(column)
            sns.despine()
            plt.show()

    def split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data.drop("Cover_Type", axis=1),
                                                                                self.data["Cover_Type"],
                                                                                test_size=0.2)

    def scaling(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def correlation_matrix_heatmap(self):
        data_subset = self.data[self.cols]
        correlation_matrix = np.corrcoef(data_subset.values.T)

        fig, ax = plt.subplots(figsize=(7, 7))
        sns.set(font_scale=1.1)
        sns.heatmap(data=correlation_matrix, square=True, cbar=True, annot=True,
                    annot_kws={'size': 10}, xticklabels=self.cols, yticklabels=self.cols,
                    fmt=".2f", linewidth=.5, cmap=sns.cubehelix_palette(as_cmap=True))
        plt.show()

    def create_model(self, hidden_layer_size=128, activation="relu", optimizer="adam", dropout_rate=0.0):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=hidden_layer_size, activation=activation, input_shape=(self.X_train.shape[1],)),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(7, activation="softmax")
        ])
        self.model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return self.model

    def train(self):
        self.model.fit(self.X_train, self.y_train, epochs=1, batch_size=32, validation_split=0.2)
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        return accuracy

    def predict_cover_type(self, sample):
        # Sample row dataframe
        test_row = pd.DataFrame([sample], columns=self.data.columns[:-1])
        test_row = self.scaler.fit_transform(test_row)
        predicted_cover_type = self.model.predict(test_row)
        # Pick class with the highest value
        predicted_class = np.argmax(predicted_cover_type)
        predicted_value = self.data["Cover_Type"].unique()[predicted_class]
        return predicted_value

    def get_hyperparameters(self):
        self.model = KerasClassifier(build_fn=self.create_model, verbose=0)

        param_grid = {
            'hidden_layer_size': [64, 128, 256],
            'activation': ['relu', 'sigmoid'],
            'optimizer': ['adam', 'sgd'],
            'dropout_rate': [0.1, 0.2, 0.3],
            'batch_size': [32, 64],
            'epochs': [5, 10]
        }

        random_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_grid,
            n_iter=10,
            cv=3,
            n_jobs=-1
        )

        random_search.fit(self.X_train, self.y_train)

        print(f"Best hyperparameters = {random_search.best_params_}")
        print("Best accuracy score = {:.2f}%".format(random_search.best_score_ * 100))


ann_model = CoverTypeClassifierNN(data_file_path='covtype.data')
# ann_model.plot_boxplots()
ann_model.outliers()
ann_model.split()
ann_model.scaling()
# ann_model.correlation_matrix_heatmap()
ann_model.create_model()
acc = ann_model.train()
print(acc)
predict = [
    2596, 51, 3, 258, 0, 510, 221, 232, 148, 6279,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]  # 5
val = ann_model.predict_cover_type(predict)
print(val)
print("+++++++++++++++++++++++++++++++HYPERPARAMETERS+++++++++++++++++++++++++++++++")
ann_model.get_hyperparameters()
print("+++++++++++++++++++++++++++++++++++++DONE+++++++++++++++++++++++++++++++++++++")
