from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_world():
    return {"Flask Status": "loaded"}


@app.route('/predict', methods=['POST'])
def predict():
    # Get the model choice from the request
    model_choice = request.json['model']

    # Load the selected model
    if model_choice == 'one':
        print("picked 1")
    elif model_choice == 'two':
        print("picked 2")
    else:
        return jsonify({'error': 'Invalid model choice.'}), 400

    return jsonify({'picked': model_choice})


if __name__ == '__main__':
    app.run(port=8080)

    # while True:
    #     print("Functionalities of REST API: ")
    #     print("1 - heuristic algorithm")
    #     print("2 - machine learning models")
    #     print("3 - neural network")
    #     print("e - exit")
    #
    #     insert = input(str("Choice: "))
    #
    #     print(f"You picked: {insert}")
    #
    #     counter = 0
    #     if insert == "e":
    #         break
    #     else:
    #         counter += 1
    #         print(f"All good #{counter}")
