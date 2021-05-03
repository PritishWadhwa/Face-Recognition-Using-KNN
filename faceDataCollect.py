import cv2
import numpy as np

# initialize the camera
cap = cv2.VideoCapture(0)

# load haarcascade file for face detection
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
faceData = []
dataSet_Path = './data/'

fileName = input("Enter the name of person: ")

while True:

    ret, frame = cap.read()

    if ret == False:
        continue

    # can also store gray frame as it takes less memory(One-third to be precise)
    greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(frame, 1.3, 5)
    if len(faces) == 0:
        continue

    # sorting faces by the area(w*h) and picking the largest face
    faces = sorted(faces, key=lambda f: f[2]*f[3])

    # faceSection = None
    for face in faces[-1:]:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Extract the region of interest (Crop out the area of interest with some offset)
        offset = 10
        # by convention, first axis is y in frame
        faceSection = frame[y - offset: y + h +
                            offset, x - offset: x + w + offset]
        faceSection = cv2.resize(faceSection, (100, 100))

        skip += 1
        if skip % 10 == 0:
            faceData.append(faceSection)
            print(len(faceData))

    cv2.imshow("Frame", frame)
    cv2.imshow("Face Section", faceSection)

    keyPressed = cv2.waitKey(1) & 0xFF
    if keyPressed == ord('q'):
        break

# convert face list in nparray
faceData = np.asarray(faceData)
faceData = faceData.reshape((faceData.shape[0], -1))

# save the data in file system
np.save(dataSet_Path + fileName + '.npy', faceData)
print("Data saved at " + dataSet_Path + fileName + '.npy')


cap.release()
cv2.destroyAllWindows()
