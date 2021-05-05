def printSamePersons():
    file = open("samePersons.txt", "w")
    file.write("image_x image_y distances\n")
    file.close()

def appendToSamePersons(image_x,image_y):
    file = open("samePersons.txt", "a")
    file.write(image_x + " " +image_y)
    file.write("\n")
    file.close()