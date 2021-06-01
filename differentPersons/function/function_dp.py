def printDifferentPersons():
    file = open("differentPersons.txt", "w")
    file.write("image_x image_y distances\n")
    file.close()

def appendToDifferentPersons(image_x,image_y):
    file = open("differentPersons.txt", "a")
    file.write(image_x + " " +image_y)
    file.write("\n")
    file.close()

def printDifferentPersonsDistance():
    file = open("differentPersonsDistance.txt", "w")
    file.write("image_x image_y distances\n")
    file.close()

def appendToDifferentPersonsDistance(image_x,image_y,distance):
    file = open("differentPersonsDistance.txt", "a")
    file.write(image_x + " " + image_y + " " + distance)
    file.write("\n")
    file.close()