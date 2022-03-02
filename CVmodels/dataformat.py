import json
import random
def split(list,racio):
    list1 = list[:int(len(list)*racio)]
    list2 = list[int(len(list)*racio)+1:]
    return [list1,list2]
bads = open("bads.txt","r")
moves = open("wheel+acse.txt","r")
def compact(pages,g):
    grandTotal = []
    for i in pages:
        line = json.loads(i)
        total = [g]
        for f in line:
            #print(f)
            #exit()
            #each hand
            hand = []
            for point in f:
                # each finger
                #[[id,"l or r"],[x,y,z]]
                if point[0][1] == "Right":
                    #ouput will be [hand,x,y,z]
                    hand += [float(1),point[1][0],point[1][1],int(point[1][2]*100)/100]
                else:
                    # left
                    hand += [float(0),point[1][0],point[1][1],int(point[1][2]*100)/100]
            total += hand
        grandTotal.append(total)
    return grandTotal
ans = open("output.txt","w")
formated =compact(bads,1) + compact(moves,0)
# add both datasets together
random.shuffle(formated)
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

#shuffle them
formated = split(formated,0.9)
labels = split(labels,0.9)
ans.write(json.dumps([formated,labels]))