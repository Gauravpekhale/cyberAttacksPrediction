
from fastapi import FastAPI, File, HTTPException, Body, UploadFile

app = FastAPI()

@app.post("/Upload_DataSets")
async def Upload_DataSets(trainingFile:UploadFile = File(...) , testingFile:UploadFile = File(...)):
    with open("train.csv","wb") as f:
        content = await trainingFile.read()
        f.write(content) 
    with open("test.csv","wb") as f:
        content = await trainingFile.read()
        f.write(content)
    return {'message' : "Files Stored Successfully"};
    

@app.post("/Train_CNN_Model")
async def Train_CNN_Model():
    
    print("Files Stored Successfully")
    return  "Hello "




if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)


