from cgo import gradient_calculator

def oracle(x,y,a):
    gx = y
    gy = x
    gxy = 1
    gax = (gx+a*gxy*gy)/(1+a**2)
    gay = (-gy+a*gxy*gx)/(1+a**2)
    return([gx,gy,gax,gay])


def test_gradient_calculator():
    pass

def gradeint_calculator(x, y, a, f):
    
