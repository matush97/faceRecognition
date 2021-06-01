# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
import pandas as pd

from function.function_dp import *

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
                help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
                help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# array
different_persons = []

printDifferentPersons()
maxPair = 1000  # maximalny pocet dvojic

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
df = pd.read_csv('../identity_CelebA.txt', sep=" ")

for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] separation images {}/{}".format(i + 1,
                                                 len(imagePaths)))

    if (len(different_persons) == maxPair):
        break

    namePhoto = imagePath.split(os.path.sep)[-1]

    valueId = df.loc[df["image"] == namePhoto]
    nameID = valueId.identity.item()

    for (j, imagePath2) in enumerate(imagePaths):
        namePhoto_2 = imagePath2.split(os.path.sep)[-1]

        if (namePhoto == namePhoto_2):
            continue

        valueId_2 = df.loc[df["image"] == namePhoto_2]
        nameID_2 = valueId_2.identity.item()

        if (nameID != nameID_2):
            if (len(different_persons) == maxPair):
                break
            if ([namePhoto_2,namePhoto] not in different_persons):
                different_persons.append([namePhoto, namePhoto_2])
                appendToDifferentPersons(namePhoto,namePhoto_2)

print("different_persons", different_persons)

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
# imagePaths = list(paths.list_images(args["dataset"]))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, person) in enumerate(different_persons):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1,
                                                 len(different_persons)))
    print(person[0])

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        # print("[INFO] processing image {}/{}".format(i + 1,
        #                                              len(imagePaths)))

        personByImagePath = imagePath.split(os.path.sep)[-1]
        if (personByImagePath != person[0]):
            continue

        # load the input image and convert it from BGR (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb,
                                                model=args["detection_method"])
        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            knownEncodings.append(encoding)
            knownNames.append(personByImagePath)
            print("one")
        break

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()


# python encode_faces.py -i C:\Users\Lenovo\PycharmProjects\faceRecognition3\Celeb -e C:\Users\Lenovo\PycharmProjects\faceRecognition3\differentPersons\embeddings.pickle
