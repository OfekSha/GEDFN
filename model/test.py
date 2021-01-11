import EmbedingLayer
file= open("USAir.txt",'r')
e=set()
for line in file:
    data=line.split()
    e.add((int(data[0]),int(data[1]),float(data[2])))
g=EmbedingLayer.Data(e)
data=g.getData()
