# Face Recognition using FastAPI

This project implements a face recognition system using FastAPI. It allows users to register their faces and recognize with their faces.

## Features

- **User Registration**: Users can register their faces by providing their names and uploading images of their faces.
- **User Recognition/Login**: Users can log in by uploading an image of their faces. The system will recognize whether the face has been registered or not.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/EllexiaLeonie/Face-Recognition-using-FastAPI.git

2. **Go to the directory**:
   ```bash
   cd Face-Recognition-using-FastAPI
   
3. **Install the dependencies**:
   ```bash
   pip install -r requirements
   

## Usage

1. **Start the FastAPI server**:
   ```bash
   uvicorn main:app --reload
   
2. **Access API Documentation**: <br>
Open your web browser and go to http://127.0.0.1:8000/docs to access the API documentation (Swagger UI).

3. **Try the System**: <br>
Click 'Try it out' to interact with the endpoints and test the system.

## Endpoints

- `/v1/register`: Endpoint for user registration.
- `/v1/recognize`: Endpoint for login/ face recognize.





