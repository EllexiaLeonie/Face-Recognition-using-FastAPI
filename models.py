from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import torch
from facenet.inception_resnet_v1 import InceptionResnetV1
from facenet.utils.download import download_url_to_file
from torchvision import transforms
from PIL import Image
import numpy as np

class FaceRecognition:
    def __init__(self):
        self.model_face = YOLO("weights/yolov8n-face.pt")
        self.facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
        face_pt = torch.load("weights/face_data.pt")
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
    
    def registration_faces(self, person_name, photos):
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
                file_name_path = "cropped_faces\{}_{}.jpg".format(person_name, count)
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
        torch.save((self.embedding_list, self.name_list), "weights/face_data.pt")

    def recognize_faces(self, image):
        image_to_test = cv2.imread(image)
        # Deteksi wajah pada gambar
        faces_to_test = self.face_crop_extractor(image_to_test)
        result = []

        # Loop untuk setiap wajah yang terdeteksi
        for face in faces_to_test:
            # Mendapatkan embedding wajah
            emb = self.get_face_embedding(face)
    
            # Konversi embedding_list ke tensor
            embedding_tensors = [torch.tensor(emb_db) for emb_db in self.embedding_list]
    
            # Hitung jarak antara wajah yang terdeteksi dengan embedding yang tersimpan
            dist_list = [torch.dist(torch.tensor(emb), emb_db).item() for emb_db in embedding_tensors]
    
            # Temukan jarak minimum
            min_dist = min(dist_list)
    
            # Jika jarak minimum kurang dari 0.8
            if min_dist < 0.8:
                idx_min = dist_list.index(min_dist)
#                 print(f"Matched! {self.name_list[idx_min][:self.name_list[idx_min].rfind('_')]}")
                print(f"Matched! {self.name_list[idx_min]}")
                result.append(self.name_list[idx_min])
            else:
                print("No face found! Please register your Face.")
        return result

        