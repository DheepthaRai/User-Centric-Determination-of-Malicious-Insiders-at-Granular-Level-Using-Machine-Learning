

# importing package
import matplotlib.pyplot as plt
import numpy as np


x = np.arange(4)
'''
#mine
y1 = [53,56,78,61]
y2 = [55,55,64,61]
y3 = [63,58,72,65]
'''
#real
y1 = [25,40,66,67]
y2 = [20,26,55,48]
y3 = [22,31,45,25]
width = 0.2

# plot data in grouped manner of bar type
plt.ylim(0,100)
plt.bar(x-0.2, y1, width, color='cyan')
plt.bar(x, y2, width, color='orange')
plt.bar(x+0.2, y3, width, color='green')
plt.xticks(x, ['LR', 'NN', 'RF', 'XG'])
plt.title("Realistic")
plt.ylabel("Instance based F1-score")
plt.legend(["Week", "Day", "Session"])
plt.show()


x = np.arange(4)
'''
#mine
y1 = [83,79,87,84]
y2 = [68,66,73,98]
y3 = [69,55,69,78]
'''
width=0.2
#real
y1 = [62,68,73,79]
y2 = [47,56,61,85]
y3 = [35,55,43,73]

# plot data in grouped manner of bar type
plt.ylim(0,100)
plt.bar(x-0.2, y1, width, color='cyan')
plt.bar(x, y2, width, color='orange')
plt.bar(x+0.2, y3, width, color='green')
plt.xticks(x, ['LR', 'NN', 'RF', 'XG'])
plt.title("Idealistic")
plt.ylabel("Instance based F1-score")
plt.legend(["Week", "Day", "Session"])
plt.show()


x = np.arange(4)
#real
y1 = [25,40,66,67]
y2 = [20,26,55,48]
y3 = [22,31,45,25]
y4 = [16,23,31,15]
y5 = [13,20,25,11]
y6 = [18,24,33,21]
y7 = [14,20,25,13]

'''
#mine
y1 = [53,56,78,61]
y2 = [55,55,64,61]
y3 = [63,58,72,65]
y4 = [57,52,61,55]
y5 = [59,53,66,58]
y6 = [58,53,62,57]
y7 = [60,54,66,60]
'''
width=0.2
# plot data in grouped manner of bar type
plt.ylim(0,100)
plt.bar(x-0.3, y1, width, color='cyan')
plt.bar(x-0.2, y2, width, color='orange')
plt.bar(x-0.1, y3, width, color='green')
plt.bar(x, y4, width, color='blue')
plt.bar(x+0.1, y5, width, color='yellow')
plt.bar(x+0.2, y6, width, color='brown')
plt.bar(x+0.3, y7, width, color='red')

plt.xticks(x, ['LR', 'NN', 'RF', 'XG'])
plt.title("Instance based results")
plt.ylabel("F1-score")
plt.legend(["Week", "Day", "Session","Subsession(n=25)","Subsession(n=50)","Subsession(t=120)","Subsession(t=240)"])
plt.show()



x = np.arange(4)
#real
y1 = [42,59,84,78]
y2 = [43,56,84,70]
y3 = [46,61,82,65]
y4 = [46,59,82,60]
y5 = [45,63,82,57]
y6 = [42,55,81,60]
y7 = [37,54,79,54]


'''
#mine
y1 = [45,59,86,82]
y2 = [43,63,84,87]
y3 = [56,67,80,87]
y4 = [44,62,79,65]
y5 = [47,58,80,74]
y6 = [39,61,76,78]
y7 = [41,65,87,83]
'''
width = 0.2

# plot data in grouped manner of bar type
plt.ylim(0,100)
plt.bar(x-0.3, y1, width, color='cyan')
plt.bar(x-0.2, y2, width, color='orange')
plt.bar(x-0.1, y3, width, color='green')
plt.bar(x, y4, width, color='blue')
plt.bar(x+0.1, y5, width, color='yellow')
plt.bar(x+0.2, y6, width, color='brown')
plt.bar(x+0.3, y7, width, color='red')

plt.xticks(x, ['LR', 'NN', 'RF', 'XG'])
plt.title("User based results")
plt.ylabel("F1-score")
plt.legend(["Week", "Day", "Session","Subsession(n=25)","Subsession(n=50)","Subsession(t=120)","Subsession(t=240)"])
plt.show()
