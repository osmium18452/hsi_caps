import scipy.io as scio
print(scio.loadmat("./Data/SalinasA_corrected.mat")["salinasA_corrected"].shape)
dict=scio.loadmat("./Data/SalinasA_gt.mat")

print(dict["salinasA_gt"].shape)

with open("./out.txt","w+") as f:
	for i in range (dict["salinasA_gt"].shape[0]):
		for j in range (dict["salinasA_gt"].shape[1]):
			# print("%3d"%dict["salinasA_gt"][i][j])
			print("%3d"%dict["salinasA_gt"][i][j],file=f,end="")
		print(file=f)
