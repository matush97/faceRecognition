def printSamePersons():
    file = open("samePersons.txt", "w")
    file.write("image_x image_y distances\n")
    file.close()

def appendToSamePersons(image_x,image_y):
    file = open("samePersons.txt", "a")
    file.write(image_x + " " +image_y)
    file.write("\n")
    file.close()

def printSamePersonsDistance():
    file = open("samePersonsDistance.txt", "w")
    file.write("image_x image_y distances\n")
    file.close()

def appendToSamePersonsDistance(image_x,image_y,distance):
    file = open("samePersonsDistance.txt", "a")
    file.write(image_x + " " + image_y + " " + distance)
    file.write("\n")
    file.close()