import face_recognition
import os
import pickle

known_faces = []
known_names = []

for person in os.listdir("Face_Database"):
    for image_name in os.listdir(f"Face_Database/{person}"):
        img_path = f"Face_Database/{person}/{image_name}"
        img = face_recognition.load_image_file(img_path)
        encoding = face_recognition.face_encodings(img)

        if encoding:
            known_faces.append(encoding[0])
            known_names.append(person)

data = {"encodings": known_faces, "names": known_names}

with open("face_encodings.pkl", "wb") as f:
    pickle.dump(data, f)

print("Face encoding completed!")
