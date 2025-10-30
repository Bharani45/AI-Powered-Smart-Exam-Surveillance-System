import face_recognition
import os

# Path to student images folder
STUDENTS_FOLDER = "students"  # put your student images here
SAMPLE_IMAGE = "students/bhar2.jpg"   # the image you want to check

# Load student images and create encodings
student_encodings = {}
for filename in os.listdir(STUDENTS_FOLDER):
    if filename.lower().endswith((".jpg", ".png")):
        path = os.path.join(STUDENTS_FOLDER, filename)
        name = os.path.splitext(filename)[0]  # filename without extension
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:  # only add if a face is found
            student_encodings[name] = encodings[0]

# Load the sample image
sample_image = face_recognition.load_image_file(SAMPLE_IMAGE)
sample_encodings = face_recognition.face_encodings(sample_image)

if not sample_encodings:
    print("No faces found in the sample image.")
else:
    sample_encoding = sample_encodings[0]
    matches = face_recognition.compare_faces(
        list(student_encodings.values()), sample_encoding, tolerance=0.6
    )

    matched_names = [name for match, name in zip(matches, student_encodings.keys()) if match]

    if matched_names:
        print("Matched with student(s):", matched_names)
    else:
        print("No matches found.")
