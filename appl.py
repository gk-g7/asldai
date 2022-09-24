import fastapi
import joblib
import numpy as np
import cv2
import mediapipe as mp
import pandas as pd

app = fastapi.FastAPI()


@app.get('/helloworld')
def helloworld():
    return {"msg": "Hello World!"}


# Define the endpoint and it's response format
@app.post("/predict/")
async def predict(file: bytes = fastapi.File(...)):
    dumpedPkl = joblib.load('dumps_pkl.pkl')
    return {"prediction": pred(file, dumpedPkl)}


def mediapipe_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Flip image around y-axis for correct handedness
    image = cv2.flip(image, 1)
    # using mediapipe hands to get co-ordinates
    mediapipe_hands_model = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                                     min_detection_confidence=0.7)
    mediapipe_output = mediapipe_hands_model.process(image)
    mediapipe_hands_model.close()
    try:
        mediapipe_output = str(mediapipe_output.multi_hand_landmarks[0])
        mediapipe_output = mediapipe_output.strip().split('\n')
        # removing unwanted details from the mediapipe output
        output_temp = []
        for i in mediapipe_output:
            if not (i == "landmark {" or i == "}"):
                output_temp.append(i)
        mediapipe_output = output_temp
        # scrape the coordinate values as list from the mediapipe output string
        coordinates = []
        for i in mediapipe_output:
            i = i.strip()
            coordinates.append(i[2:])
        return coordinates
    except Exception as e:
        print("Exception in mediapipe_image(): %s" % (e))
        raise Exception("Cannot process img")


def pre_process_img(file):
    img = np.frombuffer(file, dtype='uint8')
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    csv_file_string = ""
    try:
        for k in mediapipe_image(img):
            csv_file_string += str(k)
            csv_file_string += ','
    except Exception as e:
        print("Exception in pre_process_img(): %s" % (e))
        csv_file_string = None
    return csv_file_string


def pred(file, dumpedPkl):
    x = pre_process_img(file)
    if x is None or x == "":
        return "Invalid image"

    colNames = dumpedPkl["df_col_names"]
    colNames.remove('Label')

    vals = [i.strip() for i in x[:-1].split(',')]
    df = pd.DataFrame(data=[vals], columns=list(colNames)).astype(float)
    x = np.array(df.iloc[0])
    if "correlated_features" in dumpedPkl:
        cf = dumpedPkl["correlated_features"]
        df = df.drop(cf, axis=1)
        x = x.reshape(-1, len(cf))
    else:
        x = x.reshape(1, 63)
    if "mean" in dumpedPkl and "std" in dumpedPkl:
        mean = dumpedPkl["mean"]
        std = dumpedPkl["std"]
        x = x - mean
        x = x / std

    svmSvc = dumpedPkl["svmSvc"]
    return int(svmSvc.predict(x)[0])
