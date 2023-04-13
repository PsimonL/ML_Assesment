import requests

options = ["heuristic algorithm", "machine learning models", "neural network", "exit"]
predict = [
    2596, 51, 3, 258, 0, 510, 221, 232, 148, 6279,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]

url = 'http://localhost:8080/predict'


# data = {'option': options[0], 'pred_input': predict}
def json_request(option, pred_input):
    return {'option': option, 'pred_input': pred_input}


if __name__ == "__main__":
    print(*predict)
    while True:
        print("Functionalities of REST API: ")
        print(f"1 - {options[0]}")
        print(f"2 - {options[1]}")
        print(f"3 - {options[2]}")
        print(f"e - {options[3]}")

        insert = input(str("Choice: "))

        print("insert = ", insert)

        if insert == "e":
            print("Exited!!!")
            break

        mylist = [
            options[0] if insert == "1" else options[1] if insert == "2" else options[2] if insert == "3" else ""
        ]
        print("options[int(insert)] = ", options[int(insert)-1])
        print("predict = ", predict)
        response = requests.post(url, json=json_request(options[int(insert)-1], predict))
        print("response = ", response)
        mylist.clear()

        print("response.json() = ", response.json())

