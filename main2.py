import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime 

video_capture = cv2.VideoCapture(0)

# Load known faces
yash_image = face_recognition.load_image_file("faces/yash.jpg")
yash_encoding = face_recognition.face_encodings(yash_image)[0]

rohan_image = face_recognition.load_image_file("faces/rohan.jpg")
rohan_encoding = face_recognition.face_encodings(rohan_image)[0]

known_face_encodings = [yash_encoding, rohan_encoding]
known_face_names = ["Yash", "Rohan"]

# List of expected students
students = known_face_names.copy()

face_locations= []
face_encodings = []

# Get the current date and time
now = datetime.now()
current_date = datetime.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()  # first argument is video capture susseccful or not and second is frame(so we want frame)
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB) 

    # Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if(matches[best_match_index]):
            name = known_face_names[best_match_index]

        # Add the text if person is present
        if name in known_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottom_left_corner_oftext = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            LineType = 2
            cv2.putText(frame, name + "Present", bottom_left_corner_oftext,font, fontScale, fontColor, thickness, LineType )

            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([name, current_time])

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()