import unittest

import sys
sys.path.append('../python')
import TransFTrain as train
 
x = train.Tensor([1,2,3], dtype="int8")
y = train.Tensor(x)

print(x)
