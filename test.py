import numpy as np


testArr = np.random.rand(10)
diff1 = np.diff(testArr)
diff2 = np.diff(diff1)

print(testArr)
print(diff1)
print(diff2)
print(testArr.shape, diff1.shape, diff2.shape)
print()

print(np.cumsum(np.cumsum(diff2[3:]) + diff1[3]) + testArr[4])

