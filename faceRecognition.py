import numpy as np
import cv2
import os

####KNN####


def distance(x1, x2):
    # Returns euclidian distance
    return np.sqrt(sum((x1-x2)**2))


def knn(train, test, k=5):
    dist = []
    for i in range(train.shape[0]):
        # get the vector and corresponding label
        ix = train[i, :-1]
        iy = train[i, -1]
        # compute distance from test point
        d = distance(test, ix)
        dist.append([d, iy])
    # sort dist based shortest distance and get the top k vals
    vals = sorted(dist, key=lambda x: x[0])
    vals = vals[:k]
    # retrieving the labels and getting their frequencies
    labels = np.array(vals)
    output = np.unique(labels[:, 1], return_counts=True)
    # getting the max frequency
    index = output[1].argmax()
    pred = output[0][index]
    return pred

####KNN End####


cap = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
dataSet_Path = './data/'

# x val of data
faceData = []
# y val or labels of data
labels = []
# labels for given file
classId = 0
names = {}

# Data Preparation
for fx in os.listdir(dataSet_Path):
    if fx.endswith('.npy'):
        # load the file
        dataItem = np.load(dataSet_Path + fx)
        faceData.append(dataItem)
        # Mapping between name and class id
        names[classId] = fx[:-4]
        print("Loaded " + fx)
        # Create Labels for class
        # multiply class id with the number of images we have for each person and append those in labels
        tgt = classId * np.ones((dataItem.shape[0], ))
        classId += 1
        labels.append(tgt)

faceDataSet = np.concatenate(faceData, axis=0)
faceLabels = np.concatenate(labels, axis=0).reshape((-1, 1))


trainSet = np.concatenate((faceDataSet, faceLabels), axis=1)

# Testing
while True:
    ret, frame = cap.read()
    if ret == False:
        continue
    faces = faceCascade.detectMultiScale(frame, 1.3, 5)
    for face in faces:
        x, y, w, h = face

        # Get the face region of interest
        offset = 10
        faceSection = frame[y - offset: y + h +
                            offset, x - offset: x + w + offset]
        faceSection = cv2.resize(faceSection, (100, 100))

        # predicted output
        out = knn(trainSet, faceSection.flatten())
        predictedName = names[int(out)]
        cv2.putText(frame, predictedName, (x, y - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Faces", frame)

    keyPressed = cv2.waitKey(1) & 0xFF
    if keyPressed == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
