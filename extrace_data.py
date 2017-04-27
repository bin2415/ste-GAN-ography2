import re
fileName = "data.txt"
f = open(fileName,'r')
data = list()
for line in f.readlines():
    line = line.strip()
    regex = re.compile(r'*bob bit error (.+?),')
    print(regex.findall(line))
f.close()
    