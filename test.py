import sys 
sys.path.append("./python")

import TransFTrain as train
import TransFTrain.backend_ndarray as nd

a = nd.array([[1,2,3], [1,2,3]])
# print(a[1])
print(a[:1, :2])
print(a.start, a.stop, a.step)