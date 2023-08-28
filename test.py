import numpy as np
class ComputeLoss:
    def __init__(self,var):
        self.var = var
    def __call__(self,input):
        print(2*input)
        print(self.var)

loss = ComputeLoss(1)
loss(-2)
