import fastapi
import joblib
import numpy as np
import cv2

app = fastapi.FastAPI()


@app.get('/helloworld')
def helloworld():
    return {"msg": "Hello World!"}


# Define the endpoint and it's response format
@app.post("/predict/")
async def predict(file: bytes = fastapi.File(...)):
    dumpedPkl = joblib.load('dumps_pkl.pkl')
    return {"prediction": pred(file, dumpedPkl)}

def pre_process_img(file):
    csv_file_string = ""
    try:
        img = np.frombuffer(file, dtype='uint8')
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        csv_file_string = str(img)
    except Exception as e:
        print("Exception in pre_process_img(): %s" % (e))
        csv_file_string = None
    return csv_file_string


def pred(file, dumpedPkl):
    x = pre_process_img(file)
    print(x)
    if x is None or x == "":
        return "Invalid image"
    return x
