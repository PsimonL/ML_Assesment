from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def check():
    return {"Status": "FastAPI worked", "What next?": "Pick option"}


if __name__ == "__main__":
    while True:
        print("Functionalities of REST API: (pick)")
        print("1 - heuristic algorithm")
        print("2 - machine learning model I")
        print("3 - machine learning model II")
        print("4 - neural network")
        print("e - exit")

        insert = input(str("Choice: "))

        print(f"You picked: {insert}")

        counter = 0
        if insert == "e":
            break
        else:
            counter += 1
            print(f"All good #{counter}")
