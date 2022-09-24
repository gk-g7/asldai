import fastapi

app = fastapi.FastAPI()

@app.get('/helloworld')
def helloworld():
    return {"msg":"Hello World!"}