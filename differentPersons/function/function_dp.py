def printDifferentPersons():
    file = open("differentPersons.txt", "w")
    file.write("image_x image_y distances\n")
    file.close()

def appendToDifferentPersons(image_x,image_y):
    file = open("differentPersons.txt", "a")
    file.write(image_x + " " +image_y)
    file.write("\n")
    file.close()