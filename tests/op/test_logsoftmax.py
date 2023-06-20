# import unittest

# import sys
# sys.path.append('./python')
# # 你需要在.vscode里面添加extra地址 才能找到
# import TransFTrain as train

# class TestLogSoftmax(unittest.TestCase):
#     def test_case1(self):
#         x = train.Tensor([2,3,4], dtype="int8")
#         z = train.Tensor([0.0901, 0.2448, 0.6655], dtype="float32")
#         self.assertAlmostEqual(train.logsoftmax(x, axis=0), z, delta=train.Tensor(1e-5))

#     def test_case2(self):
#         x = train.Tensor([1,1,1], dtype="int8")
#         z = train.Tensor([0.3333, 0.3333, 0.3333], dtype="float32")
#         self.assertAlmostEqual(train.logsoftmax(x, axis=0), z, delta=train.Tensor(1e-5))

# if __name__ == '__main__':
#     unittest.main()