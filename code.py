file = open("d1a1xa_.dssp", "r")
ACC1 = []
residue1 = []
for line in file:
	if line[11] == 'I':
		ACC1.append(line[35:38])
		residue1.append(line[13])

file2 = open("d1a9xa1.dssp", "r")
ACC2 = []
residue2 = []
for line in file2:
	ACC2.append(line[35:38])
	residue2.append(line[13])


for i in range(len(ACC2)):
	if ACC2[i] != ACC1[i]:
		print(residue2[i], residue1[i], int(ACC2[i])-int(ACC1[i]))