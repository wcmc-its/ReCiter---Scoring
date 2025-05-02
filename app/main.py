from fastapi import FastAPI

app = FastAPI()

@app.post("/identityfeedbackscore")
def identityfeedbackscore(data: dict):
    # Call your logic here
    return {"result": "processed", "input": data}
