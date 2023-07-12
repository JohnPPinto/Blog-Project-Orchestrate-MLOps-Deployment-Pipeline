import uvicorn
from io import BytesIO
from PIL import Image
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from predict import predict

# Initiating FastAPI
app = FastAPI()

# A demo get method
@app.get("/")
async def read_root():
    return {'message': "Welcome, You can type docs in the URL to check the API documentation."}

# A post method to return the prediction
@app.post('/predict')
async def predict_image(file: UploadFile = File(...),
                        type: str = 'caption',
                        decode_method: str = 'beam',
                        question = None):
    # Reading the image file
    content = await file.read()
    image = Image.open(BytesIO(content))

    # Getting the prediction
    result = predict(image=image,
                     type=type,
                     decode_method=decode_method,
                     question=question)
    
    return result

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8080)