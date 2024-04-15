from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
import cv2
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
import numpy as np
import os

class FaceRegistration:
    def __init__(self):
        self.model_face = YOLO('yolov8n-face.pt')
        self.facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
        face_pt = torch.load('face_data.pt')
        self.embedding_list = face_pt[0]
        self.name_list = face_pt[1]
        
    def get_face_embedding(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)  # Convert numpy array to PIL image
        image = transforms.functional.resize(image, (160, 160))
        image = transforms.functional.to_tensor(image).float()
        image = (image - 0.5) / 0.5  # Normalizing image to range [-1, 1]
        image = image.unsqueeze(0)
        with torch.no_grad():
            embedding = self.facenet_model(image)
        return embedding.squeeze().numpy()

    def face_crop_extractor(self, image):
        # Detect face and crop
        faces = self.model_face.predict(image)
    
        cropped_faces = []  # List to store cropped faces
        for r in faces:
            annotator = Annotator(image)
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                face = image[int(b[1]):int(b[3]), int(b[0]):int(b[2])]  # Crop the face from the original image
                cropped_faces.append(face)  # Append cropped face to list
                #draw a bounding box
                c = box.cls
                annotator.box_label(b, self.model_face.names[int(c)]) 
        return cropped_faces
    
    def registration_face(self, person_name, photos):
        # Loop melalui setiap nama file dan membaca gambar
        images = []
        for file in photos:
            contents = file.file.read()
            nparr = np.fromstring(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is not None:
                images.append(image)
            else:
                print(f"Failed to read image: {file.filename}")
                
        count = 0
        
        # Loop melalui gambar-gambar
        for image in images:
            # Proses pengenalan wajah pada gambar yang diberikan
            cropped_faces = self.face_crop_extractor(image)
            for face in cropped_faces:
                # Menentukan path file menggunakan nama orang dan nomor gambar
                # Increment count
                count += 1
                # Sesuaikan tempat penyimpanan
                file_name_path = r"C:\Users\Acer\Downloads\Magang\Face recognition\detect_faces\predict66\crops\Human Face\{}_{}.jpg".format(person_name, count)
                # Menyimpan gambar dengan nama yang telah ditentukan
                cv2.imwrite(file_name_path, face)
                # Mendapatkan embedding wajah
                embed = self.get_face_embedding(face)
                if embed is not None:
                    # Menambahkan embedding baru ke dalam list
                    self.embedding_list = np.concatenate((self.embedding_list, embed.reshape(1, -1)), axis=0)
                    self.name_list.append(file_name_path.split('\\')[-1].split('.')[0])  # Menyimpan nama orang ke dalam list
                else:
                    print("Face not found")
                    
        # Simpan data yang telah diperbarui ke dalam file
        torch.save((self.embedding_list, self.name_list), 'face_data.pt')

# Usage
if __name__ == "__main__":
    face_registration = FaceRegistration()
    face_registration.registration_face(person_name, photo)



