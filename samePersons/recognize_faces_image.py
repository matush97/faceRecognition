# import the necessary packages
import face_recognition
import argparse
import pickle
import cv2
import pandas
import os
import numpy as np

from imutils import paths

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized db of facial encodings")
ap.add_argument("-i", "--dataset", required=True,
                help="path to input directory of faces + images")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
                help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

from samePersons.function.function_sp import *

# array
same_persons= []
distanceArray = []

true_positive = 0
false_negative = 0
tolerance = 0.6

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
print(data)

# read from samePersons.txt the second columm
df = pandas.read_csv('samePersons.txt',sep=" ")
image_x = list(df['image_x'])
image_y = list(df['image_y'])
same_persons = np.column_stack((image_x,image_y))

#create new .text file
printSamePersonsDistance()

# loop over the image paths
for (i, person) in enumerate(same_persons):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1,
                                                 len(same_persons)))
    print(person[1])

    # loop over the image paths
    for (j, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        # print("[INFO] processing image {}/{}".format(j + 1,
        #                                              len(imagePaths)))

        personByImagePath = imagePath.split(os.path.sep)[-1]
        if (personByImagePath != person[1]):
            continue

        print(personByImagePath,person[1])
        # load the input image and convert it from BGR to RGB
        image = cv2.imread(imagePath)

        #edit size of photo
        h, w, c = image.shape
        if (h > 1000 or w > 1000):
            width = 1000
            height = 1000
            dim = (width, height)
            image = cv2.resize(image,dim)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes corresponding
        # to each face in the input image, then compute the facial embeddings
        # for each face
        print("[INFO] recognizing faces...")
        boxes = face_recognition.face_locations(rgb,
        	model=args["detection_method"])
        encodings = face_recognition.face_encodings(rgb, boxes)
        print(encodings)

        if (len(encodings) == 0):
            break

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            distances = face_recognition.face_distance(data["encodings"], encoding)
            # result = list(distances <= tolerance)

            distance = distances[i]

            print(distance)

            if (distance <= tolerance):
                name = "True " + person[0]
                true_positive += 1
            else:
                name = "False"
                false_negative += 1

            print(name)

            # # check to see if we have found a match
            # if True in result:
            #     # find the indexes of all matched faces then initialize a
            #     # dictionary to count the total number of times each face
            #     # was matched
            #     matchedIdxs = [i for (i, b) in enumerate(result) if b]
            #     counts = {}
            #
            #     # loop over the matched indexes and maintain a count for
            #     # each recognized face face
            #     for i in matchedIdxs:
            #         name = data["names"][i]
            #         counts[name] = counts.get(name, 0) + 1
            #
            #     # determine the recognized face with the largest number of
            #     # votes (note: in the event of an unlikely tie Python will
            #     # select first entry in the dictionary)
            #     # name = max(counts, key=counts.get)
            #
            #     # ak je v counts nasa person, kt. hladame je true
            #     if (person[0] in counts):
            #         name = "True " + person[0]
            #         true_positive += 1
            #     else:
            #         name = "False"
            #         false_negative += 1
            #
            #     print(name)

            appendToSamePersonsDistance(person[0],person[1],str(distance))

            # # loop over the recognized faces
            # for top, right, bottom, left in boxes:
            #     # draw the predicted face name on the image
            #     cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            #     y = top - 15 if top - 15 > 15 else top + 15
            #     cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            #                 0.75, (0, 255, 0), 2)
            #
            #     # show the output image
            #     cv2.imshow("Image", image)
            #     cv2.waitKey(0)
            break

# df2 = pandas.DataFrame(df)
# print(df2)
# df2['distances'] = distanceArray
# print(df2)
# print(distanceArray)

# df['distances'] = distanceArray
# df.to_csv('samePersons.txt',sep=" ", index=None)

print("true_positive " + str(true_positive))
print("false_negative " + str(false_negative))


# python recognize_faces_image.py -e C:\Users\Lenovo\PycharmProjects\faceRecognition2\embeddings.pickle -i C:\Users\Lenovo\Pychar
# mProjects\faceRecognition2\019532.jpg
