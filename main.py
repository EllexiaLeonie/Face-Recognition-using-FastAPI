import uvicorn
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from models import FaceRecognition
from typing import List
from tempfile import NamedTemporaryFile
app = FastAPI(title='Face Recognition')

# Menginisialisasi objek FaceRecognition di luar scope handler
face_recognizer = FaceRecognition()
    
@app.post("/v1/register_faces/")
async def registration_face(person_name: str = Form(...), files: List[UploadFile] = File(...)):
    try:
        face_recognizer.registration_faces(person_name, files)
        return {"message": f"Face registration for {person_name} was successful!"}
    except FileNotFoundError as e:
        # Tangkap kesalahan jika file tidak ditemukan
        raise HTTPException(status_code=404, detail="File tidak ditemukan")
    except Exception as e:
        # Tangkap kesalahan umum lainnya
        raise HTTPException(status_code=500, detail="Terjadi kesalahan internal server")

@app.post("/v1/recognize_face/")
async def recognize_faces(upload_image: UploadFile = File(...)):
    try:
        with NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image:
            temp_image.write(upload_image.file.read())
            temp_image_path = temp_image.name
        matches = face_recognizer.recognize_faces(temp_image_path)
        # Cek apakah matches adalah list kosong
        if not matches:
            return {"Result": "No face found! Please register your Face."}
        return {"Result": matches}
    except FileNotFoundError as e:
        # Tangkap kesalahan jika file tidak ditemukan
        raise HTTPException(status_code=404, detail="File tidak ditemukan")
    except Exception as e:
        # Tangkap kesalahan umum lainnya
        raise HTTPException(status_code=500, detail="Terjadi kesalahan internal server")
