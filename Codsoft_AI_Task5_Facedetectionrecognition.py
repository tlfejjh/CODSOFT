import cv2
from google.colab.patches import cv2_imshow

def detect_faces(img_path):
    # Load the cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("Failed to load cascade classifier!")
        return

    # Read the input image
    img = cv2.imread(img_path)

    if img is None:
        print("Failed to read image!")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the output in Google Colab
    cv2_imshow(img)

# Test
detect_faces('/content/drive/MyDrive/Datasets/keerthan.jpg.jpeg')
