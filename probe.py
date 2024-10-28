import matplotlib.pyplot as plt
import numpy as np

data = np.load("synthetic/original/1_0_index_3_male_right_27.npz", allow_pickle=True)
for i in range(len(data["X1"])):
    plt.plot(data["X1"][i], data["Y1"][i])
plt.axis("equal")
plt.show()
print(data["X1"])