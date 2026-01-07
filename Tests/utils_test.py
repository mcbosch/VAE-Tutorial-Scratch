from Models.utils import *
import numpy as np

# ===== ACTIVATION FUNCTIONS TESTS

def ReLU_tst():

    # Test ReLU for scalar:
    R = ReLU()
    assert R(-1) == 0., "Should be a float 0."
    assert R(1) == 1., "Should be a float 1."
    
    # Test Dimensionalities
    A = np.array([[1, 2, 0.5],
                  [-1.2, 0, 2],
                  [-2, -1, -0.2]])
    result = np.array([[1. ,2. ,0.5],
                       [0. ,0. , 2.],
                       [0. ,0. , 0.]])
    
    B = R(A)
    for i in range(3):
        for j in range(3):
            assert B[i, j] == result[i, j], f"Matrix {A} should have as a result {B}"
            
    # Test Partial
    B = R.partial(A)
    result = np.array([[1, 1, 1],
                       [0, 0, 1],
                       [0, 0, 0]])
    for i in range(3):
        for j in range(3):
             assert B[i, j] == result[i, j], f"Matrix {A} should have as a result {B}"
            

    # Test string
    assert str(R) == 'ReLU'

    print("ReLU test: \033[1;32mPASSED\033[0m")

def Sigmoind_tst():
    
    # Test ReLU for scalar:
    S = Sigmoind()
    s = lambda x: 1/(1+np.exp(-x))
    p = lambda x: s(x)*(1-s(x))
    assert S(-1) == s(-1.), f"Should be a float {s(-1)}."
    assert S( 1) == s(1.), f"Should be a float {s(1)}."
    
    # Test Dimensionalities
    A = np.array([[1, 2, 0.5],
                  [-1.2, 0, 2],
                  [-2, -1, -0.2]])
    
    result = np.array([[s(1.) ,s(2.) ,s(0.5)],
                       [s(-1.2) ,s(0.) , s(2.)],
                       [s(-2.) ,s(-1.) , s(-0.2)]])
    
    B = S(A)
    for i in range(3):
        for j in range(3):
            assert B[i, j] == result[i, j], f"Matrix {A} should have as a result {B}"
            

    # Test Partial
    
    B = S.partial(A)
    result = np.array([[p(1.) ,p(2.) ,p(0.5)],
                       [p(-1.2) ,p(0.) , p(2.)],
                       [p(-2.) ,p(-1.) , p(-0.2)]])
    for i in range(3):
        for j in range(3):
             assert B[i, j] == result[i, j], f"Matrix {A} should have as a result {B}"
            

    # Test string
    assert str(S) == 'Sigmoind'
    print("Sigmoind test: \033[1;32mPASSED\033[0m")

def Softmax_tst():
    S = Softmax()
    b = np.array([1,2,3])
    assert np.sum(S(b)), f"Sum of {S(b)} should be 1"
    A = np.array([[1, 2, 0.5],[-1.2, 0, 2],[-2, -1, -0.2]])
    P = np.sum(S(A), axis=1)
    for i in range(3):
        assert P[i] == 1, f"Sum of rows {S(A)[i]} should be 1 and is {sum(S(A)[i])}"
    
    assert str(S) == "Softmax"
    print("Sigmoind test: \033[1;32mPASSED\033[0m")

# ===== LOSS FUNCTIONS TESTs

def CrossEntropy_tst():
    C = CrossEntropy()
    y_pred = np.array([1,0,1,0])
    y_true = np.array([1,0,0,1])
    result = -np.log(1e-13) - np.log(1+1e-13)
    
    warning_message = f"The cross entropy of {y_pred} and {y_true} should be {result}\nYour result: {C(y_pred,y_true)}"
    assert abs(C(y_pred, y_true)-result) < 1e-12, warning_message
    # Check for a batched input

    Y_pred = np.array([[1,0,1,0],
                      [1,1,1,1],
                      [0,0,0,0]])
    Y_true = np.array([[1,0,0,1],
                      [1,1,1,1],
                      [0,0,0,1]])
    E = C(Y_pred, Y_true)
    result = np.array([-np.log(1e-13) - np.log(1+1e-13), -4*np.log(1+1e-13), -np.log(1e-13)])
    for i in range(3):
        warning_message = f"The cross entropy of {Y_pred[i]} and {Y_true[i]} should be {result[i]}\nINPUT:\nY_pred:\n{Y_pred}\nY_true\n{Y_true}\nYoue result\n{E}\nExpected:\n{result}"
        assert abs(E[i]-result[i]) < 1e-12, warning_message
    assert str(C) == "CrossEntropy"
    print("CrossEntropy test: \033[1;32mPASSED\033[0m")