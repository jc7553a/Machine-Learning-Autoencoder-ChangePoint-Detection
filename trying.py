import random as ra

data = []

for i in range(1000):
    holder = []
    for j in range(8):
        if j == 0:
            holder.append(ra.randint(8, 20))
        if j == 1:
            holder.append(ra.randint(4, 9))
        if j ==2:
            holder.append(ra.randint (25, 45))
        if j ==3 :
            holder.append(ra.randint(100,200))
        if j == 4:
            holder.append(ra.randint(15, 20))
        if j == 5:
            holder.append(ra.randint(70,85))
        if j == 6:
            holder.append(ra.randint(2, 7))
        if j == 7:
            holder.append(ra.randint(5, 10))
    data.append(holder)

print(len(data))

for i in range(100):
    rando = ra.randint(0,1000)
    holder = []
    for j in range(8):
        if j == 0:
            holder.append(ra.randint(30, 40))
        if j == 1:
            holder.append(ra.randint(4, 9))
        if j ==2:
            holder.append(ra.randint (35, 75))
        if j ==3 :
            holder.append(ra.randint(400,600))
        if j == 4:
            holder.append(ra.randint(15, 20))
        if j == 5:
            holder.append(ra.randint(12,18))
        if j == 6:
            holder.append(ra.randint(2, 7))
        if j == 7:
            holder.append(ra.randint(55, 100))
    data[rando] = holder
    
