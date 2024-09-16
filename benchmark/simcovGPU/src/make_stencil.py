t = 0
for i in range(-1,2):
	for j in range(-1,2):
		for k in range(-1,2):
			if(not(i == 0 and j == 0 and k == 0)):
				print("loop_x[{}] = {};".format(t,i))
				t+=1
t = 0
for i in range(-1,2):
	for j in range(-1,2):
		for k in range(-1,2):
			if(not(i == 0 and j == 0 and k == 0)):
				print("loop_y[{}] = {};".format(t,j))
				t+=1
t = 0
for i in range(-1,2):
	for j in range(-1,2):
		for k in range(-1,2):
			if(not(i == 0 and j == 0 and k == 0)):
				print("loop_z[{}] = {};".format(t,k))
				t+=1