import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class CoverTypeClassifierHeuristic:
    def __init__(self, data_file_path):
        # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
        self.data = pd.read_csv(data_file_path, header=None)

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
        self.cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                     'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                     'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                     'Horizontal_Distance_To_Fire_Points', 'Cover_Type']
        self.X = self.data.drop('Cover_Type', axis=1)
        self.y = self.data['Cover_Type']



    def correlation_matrix_heatmap(self):
        data_subset = self.data[self.cols]
        correlation_matrix = np.corrcoef(data_subset.values.T)

        fig, ax = plt.subplots(figsize=(7, 7))
        sns.set(font_scale=1.1)
        sns.heatmap(data=correlation_matrix, square=True, cbar=True, annot=True,
                    annot_kws={'size': 10}, xticklabels=self.cols, yticklabels=self.cols,
                    fmt=".2f", linewidth=.5, cmap=sns.cubehelix_palette(as_cmap=True))
        plt.show()

    def make_histogram(self, col_name, bins_val):
        min_val = self.data[col_name].min()
        max_val = self.data[col_name].max()
        print(f"Lowest {col_name} value: {min_val}. Highest {col_name} value: {max_val}.")
        plot_hist = self.data[col_name].plot.hist(bins=bins_val, grid=True)
        plot_hist.set_title(f"Represents number of cars for each production {col_name.upper()} category")
        plot_hist.set_xlabel(f"{col_name}")
        plot_hist.set_ylabel("Number of observations")
        plt.show()

    def min_max_mean_values(self, col_name):
        max_val = self.data[col_name].max()
        min_val = self.data[col_name].min()
        print(f"Maximum value of '{col_name}' - row {self.data.columns.get_loc}: {max_val}")
        print(f"Minimum value of '{col_name}' - row {self.data.columns.get_loc}: {min_val}")

        mean_val = self.data[col_name].mean()
        print(f"Mean value of '{col_name}' - row {self.data.columns.get_loc}:{mean_val}")

    def get_accu_simple_heuristic(self):
        self.data['predicted_cover_type'] = self.data.apply(self.get_pred_simple_heuristic, axis=1)
        accuracy = (self.data['predicted_cover_type'] == self.data['Cover_Type']).mean() * 100
        return accuracy

    def get_pred_simple_heuristic(self, row):
        if row['Elevation'] > 3000 and row['Slope'] < 20:
            return 1
        elif row['Elevation'] < 3000 and row['Slope'] < 15:
            return 2
        elif row['Elevation'] < 2500 and row['Slope'] > 10:
            return 3
        elif 180 < row['Aspect'] < 150 and row['Hillshade_Noon'] > 200 and row[
            'Horizontal_Distance_To_Roadways'] > 2000:
            return 4
        elif 270 < row['Aspect'] < 300 and row['Hillshade_3pm'] > 150 and row['Horizontal_Distance_To_Roadways'] < 2000:
            return 5
        elif row['Wilderness_Area_1'] == 0 and row['Soil_Type_10'] == 1:
            return 6
        else:
            return 7


# heuristic = CoverTypeClassifierHeuristic(data_file_path='dataset_and_info/covtype.data')
#
# heuristic.correlation_matrix_heatmap()
#
# heuristic.make_histogram('Elevation', 57)
# heuristic.make_histogram('Aspect', 57)
# heuristic.make_histogram('Slope', 57)
# heuristic.make_histogram('Horizontal_Distance_To_Hydrology', 57)
# heuristic.make_histogram('Horizontal_Distance_To_Roadways', 57)
# heuristic.make_histogram('Horizontal_Distance_To_Fire_Points', 57)
#
# heuristic.min_max_mean_values('Elevation')
# heuristic.min_max_mean_values('Aspect')
# heuristic.min_max_mean_values('Horizontal_Distance_To_Hydrology')
# heuristic.min_max_mean_values('Horizontal_Distance_To_Roadways')
# heuristic.min_max_mean_values('Horizontal_Distance_To_Fire_Points')
#
# accuracy = heuristic.get_accu_simple_heuristic()
# print("heuristic accuracy = ", accuracy)

# sample_row = pd.Series({
    # 'Elevation': 2800,
    # 'Slope': 12,
    # 'Aspect': 220,
    # 'Hillshade_Noon': 220,
    # 'Wilderness_Area_3': 1,
    # 'Horizontal_Distance_To_Hydrology': 120,
    # 'Hillshade_9am': 160
# })
# pred_val = heuristic.get_pred_simple_heuristic(sample_row)
# print("heuristic pred_val = ", pred_val)

# 'Elevation': 2596,
# 'Slope': 3,
# 'Aspect': 51,
# 'Hillshade_Noon': 232,
# 'Wilderness_Area_3': 0,
# 'Horizontal_Distance_To_Hydrology': 258,
# 'Hillshade_9am': 221