import fastapi
import joblib

app = fastapi.FastAPI()


@app.get('/helloworld')
def helloworld():
    return {"msg": "Hello World!"}


@app.post("/predict/")
async def predict(file: bytes = fastapi.File(...)):
    dumpedPkl = joblib.load('dumps_pkl.pkl')
    print(dumpedPkl)
    return {"prediction": dumpedPkl["df_col_names"]}
