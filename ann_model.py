import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




# class CoverTypeClassifierANN:
    # def __init__(self, data_file_path, header=None):
    #     self.data = pd.read_csv(data_file_path, header=None)
    #
    #     columns = [
    #         'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    #         'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    #         'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    #         'Horizontal_Distance_To_Fire_Points'
    #     ]
    #     wilderness_areas = [f'Wilderness_Area_{i}' for i in range(1, 5)]
    #     soil_types = [f'Soil_Type_{i}' for i in range(1, 41)]
    #     columns.extend(wilderness_areas)
    #     columns.extend(soil_types)
    #     columns.append('Cover_Type')
    #     self.data.columns = columns
    #
    #     self.X = self.data.drop('Cover_Type', axis=1)
    #     self.y = self.data['Cover_Type']
    #     self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
    #         self.X, self.y, test_size=0.2, random_state=42)
    #
    #     self.scaler = StandardScaler()
    #     self.scaler.fit(self.X_train)
    #     self.X_train = self.scaler.transform(self.X_train)
    #     self.X_test = self.scaler.transform(self.X_test)
    #
    #     self.cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    #                  'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    #                  'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    #                  'Horizontal_Distance_To_Fire_Points', 'Cover_Type']
    #
    # def plot_boxplots(self):
    #     for i, column in enumerate(self.cols):
    #         fig, ax = plt.subplots()
    #         sns.boxplot(data=self.data[column], ax=ax).set(title=f"{column.upper()} boxplot", xlabel=f"{column}",
    #                                                   ylabel=f"Value of {column}")
    #         ax.set_title(column)
    #         sns.despine()
    #         plt.show()
    #
    # def remove_outliers(self):
    #     # # Calculate the z-scores of each column - measure that represents the number of standard deviations a data point is from the mean of the dataset
    #     # z_scores = (self.data - self.data.mean()) / self.data.std()
    #     # # Set the threshold for the z-score
    #     # threshold = 3
    #     # # Remove any rows where the z-score is greater than the threshold
    #     # self.data = self.data[(np.abs(z_scores) < threshold).all(axis=1)]
    #     # Interquartile Range (IQR)
    #     Q1 = self.data.quantile(0.25)
    #     Q3 = self.data.quantile(0.75)
    #     IQR = Q3 - Q1
    #     self.data = self.data[~((self.data < (Q1 - 1.5 * IQR)) | (self.data > (Q3 + 1.5 * IQR))).any(axis=1)]
    #
    # def correlation_matrix_heatmap(self):
    #     data_subset = self.data[self.cols]
    #     correlation_matrix = np.corrcoef(data_subset.values.T)
    #
    #     fig, ax = plt.subplots(figsize=(7, 7))
    #     sns.set(font_scale=1.1)
    #     sns.heatmap(data=correlation_matrix, square=True, cbar=True, annot=True,
    #                 annot_kws={'size': 10}, xticklabels=self.cols, yticklabels=self.cols,
    #                 fmt=".2f", linewidth=.5, cmap=sns.cubehelix_palette(as_cmap=True))
    #     plt.show()
    #
    # def build_model(self):
    #     self.model = tf.keras.models.Sequential([
    #         tf.keras.layers.Dense(128, activation="relu", input_shape=(self.X_train.shape[1],)),
    #         tf.keras.layers.Dense(7, activation="softmax")
    #     ])
    #     self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    #
    # def train_model(self, epochs=10, batch_size=32, validation_split=0.2):
    #     self.model.fit(self.X_train, self.y_train, epochs=10, batch_size=32, validation_split=0.2)


# model = CoverTypeClassifierANN("covtype.data")
# print("Model loaded")
# # model.plot_boxplots()
# # print("Boxplots finished")
# # model.remove_outliers()
# # print("Removing outliers finished")
# # model.correlation_matrix_heatmap()
# # print("Heat map finished")
#
# model.build_model()
# print("Building model finished")
# model.train_model()
# print("Training model finished")