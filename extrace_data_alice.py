import re
import matplotlib.pyplot as plt
fileName = "best_bob90.log"
f = open(fileName,'r')
data = list()
for line in f.readlines():
    line = line.strip()
    if line != "":
        regex = re.compile(r'(\w|\W)*alice bit error (.+?),(\w|\W)*')
        result = regex.findall(line)
        data.append(float(result[0][1]))
f.close()
xlabel = range(0, 50000, 100)
plt.figure()
plt.plot(xlabel, data)
plt.xlabel('training iteration', fontsize = 16)
plt.ylabel('Alice bit error', fontsize = 16)
plt.savefig("./training_alice.png")



    