from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from request_driver import options
from two_ml_models import CoverTypeClassifierRFLR
from ann_model import CoverTypeClassifierNN
from simple_heuristic_algorithm import CoverTypeClassifierHeuristic

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_world():
    return {"Flask Status": "loaded"}


@app.route('/predict', methods=['POST'])
def predict():
    picked_option = request.json['option']
    sample_pred = np.array(request.json['pred_input'])
    print("picked_option = ", picked_option)
    print("sample_pred = ", sample_pred)

    output_json = {}
    output_json.clear()

    if picked_option == options[0]:
        classifier = CoverTypeClassifierHeuristic(data_file_path='covtype.data')
        heuristic_acc = classifier.get_accu_simple_heuristic()

        sample_row = pd.Series({
            'Elevation': int(sample_pred[0]),
            'Slope': int(sample_pred[2]),
            'Aspect': int(sample_pred[1]),
            'Hillshade_Noon': int(sample_pred[7]),
            'Wilderness_Area_3': int(sample_pred[12]),
            'Horizontal_Distance_To_Hydrology': int(sample_pred[3]),
            'Hillshade_9am': int(sample_pred[6])
        })

        predicted_cover_type = classifier.get_pred_simple_heuristic(sample_row)
        predicted_cover_type = np.int64(predicted_cover_type).tolist()
        output_json = {"Heuristic Algorithm Accuracy": heuristic_acc,
                       "Predict 'Cover_type' value for sample - Heuristic Algorithm": predicted_cover_type}
    elif picked_option == options[1]:
        classifier = CoverTypeClassifierRFLR(data_file_path='covtype.data')
        random_forest_acc = classifier.get_random_forest_accuracy()
        predicted_cover_type = classifier.predict_cover_type(sample_pred, "RF")
        predicted_cover_type = np.int64(predicted_cover_type).tolist()
        output_json = {"Random Forest Accuracy": random_forest_acc,
                       "Predict 'Cover_type' value for sample - Random Forest": predicted_cover_type}
    elif picked_option == options[2]:
        classifier = CoverTypeClassifierRFLR(data_file_path='covtype.data')
        logistic_reg_acc = classifier.get_logistic_regression_accuracy()
        predicted_cover_type = classifier.predict_cover_type(sample_pred, "LR")
        predicted_cover_type = np.int64(predicted_cover_type).tolist()
        output_json = {"Logistic Regression Accuracy:": logistic_reg_acc,
                       "Predict 'Cover_type' value for sample - Logistic Regression": predicted_cover_type}

    elif picked_option == options[3]:
        classifier = CoverTypeClassifierNN(data_file_path='covtype.data')
        classifier.outliers()
        classifier.split()
        classifier.scaling()
        classifier.create_model(optimizer="adam", hidden_layer_size=128, epochs=1, dropout_rate=0.0, batch_size=32, activation="relu")
        ann_acc = classifier.train(epochs=1, batch_size=32)
        predicted_cover_type = classifier.predict_cover_type(sample_pred)
        predicted_cover_type = np.int64(predicted_cover_type).tolist()
        output_json = {"ANN Accuracy": ann_acc,
                       "Predict 'Cover_type' value for sample - NN": predicted_cover_type}
    else:
        return jsonify({'error': 'Invalid option choice.'}), 400

    return jsonify(output_json)


if __name__ == '__main__':
    app.run(port=8080)
