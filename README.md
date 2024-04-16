# Face Recognition using FastAPI

This project implements a face recognition system using FastAPI. It allows users to register their faces and recognize with their faces.

## Features

- User registration: Users can register their faces by providing their names and uploading images of their faces.
- User recognizer/ login: Users can log in by uploading image of their faces. The system will recognize whether the face has been registered or not. 

## Installation

1. Clone the repository:
   git clone https://github.com/EllexiaLeonie/Face-Recognition-using-FastAPI.git
   
2. Install the dependencies:
   pip install -r requirements.txt
   

## Usage

1. Start the FastAPI server:
   uvicorn main:app --reload
 
2. Open your web browser and go to `http://localhost:8000` to access the API documentation (Swagger UI).
   add "/docs" to the last url (http://localhost:8000/docs)

3. Click 'Try it out' to try the system

## Endpoints

- `/v1/register`: Endpoint for user registration.
- `/v1/recognize`: Endpoint for login/ face recognize.





