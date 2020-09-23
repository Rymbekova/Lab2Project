file = open("d1a1xa_.dssp", "r")
c=0
seq = ""
SS = ""
for line in file:
	c = c + 1 
	if c > 28:
		seq=seq+str(line[13])
		if line[16] == 'H' or line[16] == "G" or line[16] == "I":
			ss = "H"
		elif line[16] == 'B' or line[16] == 'E':
			ss = "E"
		elif line[16] == 'T' or line[16] == "S" or line[16] == " ":
			ss = "C"
		SS = SS + ss

print (seq, SS)