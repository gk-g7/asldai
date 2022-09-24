import uvicorn
import fastapi

app = fastapi.FastAPI()


@app.get('/helloworld')
def helloworld():
    return {"msg":"Hello World!"}


if __name__ == '__main__':
    uvicorn.run(app)