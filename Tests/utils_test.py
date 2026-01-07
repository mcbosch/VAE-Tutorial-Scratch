from Models.utils import *
import numpy as np
import sys
# We test the behaviour of the utils functions for different types of entrys
def ReLU_tst():
    # Test ReLU for scalar:
    R = ReLU()
    assert R(-1) == 0., "Should be a float 0."
    assert R(1) == 1., "Should be a float 1."
    
    # Test Dimensionalities
    A = np.array([[1, 2, 0.5],
                  [-1.2, 0, 2]])
    print("ReLU test: \033[1;32mPASSED\033[0m")

print(sys.path[0])