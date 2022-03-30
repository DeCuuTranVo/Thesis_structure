a = [1.1079,1.0917,1.1444,0.9714,1.1385]
b =  [1.1066,1.2353,1.2058,1.1108,1.0738]

import numpy as np
mean_a = np.mean(a)
mean_b = np.mean(b)

print(mean_a, mean_b)

max_index_a = np.argmax(a)
min_index_b = np.argmin(b)

print(max_index_a, min_index_b)

