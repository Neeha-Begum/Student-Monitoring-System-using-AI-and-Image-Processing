import cv2
import os

# Get ID and Name from user
student_id = input("Enter ID: ")
name = input("Enter Name: ")

# Create directory using both ID and Name
folder_name = f"{student_id}_{name}"
path = os.path.join("Face_Database", folder_name)
os.makedirs(path, exist_ok=True)

# Open camera
cam = cv2.VideoCapture(0)
count = 0

while count < 10:
    ret, frame = cam.read()
    if not ret:
        break

    # Save image to path
    filename = os.path.join(path, f"{count}.jpg")
    cv2.imwrite(filename, frame)
    
    count += 1
    cv2.imshow("Capture", frame)
    cv2.waitKey(500)  # wait 500ms between captures

# Release resources
cam.release()
cv2.destroyAllWindows()
print(f"Captured 10 images for {name} (ID: {student_id}).")
