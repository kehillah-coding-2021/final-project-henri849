import json
import random
# just a function that takes the dataset and splits in into training and testing using a decimal racio, ie: 90 % training would take (list,0.9)
def split(list,racio):
    list1 = list[:int(len(list)*racio)]
    list2 = list[int(len(list)*racio)+1:]
    return [list1,list2]
#open the two files (the first has the random captures that are not acselerating)
bads = open("bads.txt","r")
# the wheel+acse.txt is the captures of acseleration
moves = open("wheel+acse.txt","r")
# this function is supposed to take the two datasets, and sort them by content, take the labels out and put it into one list that has all the data points and a second list with all the labels (in an order that corresponds with the datapoint list)
def compact(pages,g):
    # here we will put all the points on one hand
    grandTotal = []
    # here we loop through the captures
    for i in pages:
        # translate the line into something python understands
        line = json.loads(i)
        # create a list for every point on the hand and give it a label for wether it is an acselerating hand gesture or normal
        total = [g]
        for f in line:
            hand = []
            for point in f:
                # each finger
                #it takes in [[id (point on the hand),"left or right hand"],[x,y,z] (positions of the points)]
                # if it's a right hand
                if point[0][1] == "Right":
                    #ouput will be [hand(1 = right),x,y,z]
                    hand += [float(1),point[1][0],point[1][1],int(point[1][2]*100)/100]
                else:
                    # left
                    #ouput will be [hand(0 = left)),x,y,z (positions of the points)] the int(point[1][2]*100)/100 is just rounding
                    hand += [float(0),point[1][0],point[1][1],int(point[1][2]*100)/100]
            # adding the points to a "hand"
            total += hand
        # appending the hand to the list of all hands
        grandTotal.append(total)
    return grandTotal
# creating a new ouput file where we will put these formated data points
ans = open("output.txt","w")
# add both sets
formated =compact(bads,1) + compact(moves,0)
# shuffle
random.shuffle(formated)
#add the labels
labels = []
for i in range(len(formated)):
    if formated[i][0] == 0:
        # goods
        labels.append([1.0,0.0])
        formated[i].pop(0)
    else:
        # "bads"
        labels.append([0.0,1.0])
        formated[i].pop(0)

#split them
formated = split(formated,0.9)
labels = split(labels,0.9)
#write to the file all these points and labels
ans.write(json.dumps([formated,labels]))
