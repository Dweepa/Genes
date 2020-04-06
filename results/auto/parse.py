def interpret_file(f):
	x = ['balanced', 'unbalanced']
	y = [3, 4, 14]
	z = ['knn', 'rf', 'xt']
	for a in x:
		for b in y:
			for c in z:
				try:
					with open(f"./{f}/{a}/report_{b}_{c}") as file:
						lines = file.readlines()
				except:
					continue
				accuracy = round(float(lines[0].split(" ")[1][:-1])*100, 2)
				precision = float(lines[-1].split("       ")[1].split("      ")[0])
				recall = float(lines[-1].split("       ")[1].split("      ")[1])
				f1 = float(lines[-1].split("       ")[1].split("      ")[2])
				write_file.write(f"{f},{a},{b},{c},{accuracy}%,{precision},{recall},{f1}\n")

write_file = open("./interpretation", "w")
write_file.write(f"File Name,State,Num Classes,Method,Accuracy,Precision,Recall,F1\n")
f = [50, 100, 150, 200, 250, 300]
for s in f:
	interpret_file(f'mine_{s}')
