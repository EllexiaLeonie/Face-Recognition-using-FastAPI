import uvicorn
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from register import FaceRegistration
from login import FaceRecognizer
from typing import List
from tempfile import NamedTemporaryFile
from fastapi.responses import HTMLResponse
app = FastAPI(title='Face Recognition')
    
@app.post("/v1/register_faces/")
async def registration_face(person_name: str = Form(...), files: List[UploadFile] = File(...)):
    try:
        registration = FaceRegistration()
        registration.registration_face(person_name, files)
        return {"message": f"Face registration for {person_name} was successful!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/recognize_face/")
async def recognize_faces(upload_image: UploadFile = File(...)):
    try:
        with NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image:
            temp_image.write(upload_image.file.read())
            temp_image_path = temp_image.name
        face_recognizer = FaceRecognizer()
        matches = face_recognizer.recognize_faces(temp_image_path)
        # Cek apakah matches adalah list kosong
        if not matches:
            return {"Result": "No face found! Please register your Face."}
        return {"Result": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
