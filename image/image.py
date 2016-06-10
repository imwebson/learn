import numpy as np
from PIL import Image

beta, eta, h =1e-3,2.1e-3,0.0 
def E(x,y):
    xxm=np.zeros_like(x)
    xxm[:-1, :]=x[1:, :]
    xxm[1:, :]+=x[:-1, :]
    xxm[:, :-1] += x[:, 1:]  # right
    xxm[:, 1:] += x[:, :-1]  # left
    xx = np.sum(xxm * x)
    xy = np.sum(x * y)
    xsum = np.sum(x)
    return h * xsum - beta * xx - eta * xy

def is_valid(i, j, shape):
        """Check if coordinate i, j is valid in shape."""
        return i >= 0 and j >= 0 and i < shape[0] and j < shape[1]

def Elocal(E0, i, j, x, y):
    old = x[i,j]
    new = old * (-1)
    Enew=E0-h*old+h*new
    Enew+=eta*y[i,j]*old-eta*y[i,j]*new
    adjacent = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    neighbors = [x[i + di, j + dj] for di, dj in adjacent
                     if is_valid(i + di, j + dj, x.shape)]
    Enew+=beta * sum(a * old for a in neighbors)
    Enew-= beta * sum(a * new for a in neighbors)
    return old,new,E0,Enew

def ICM(y):
    x=np.array(y)
    Ebest=E(x,y)
    for idx in np.ndindex(y.shape):
        old, new, E0, Enew=Elocal(Ebest, idx[0],idx[1],x,y)
        if(Enew < Ebest):
            Ebest = Enew
            x[idx] = new
    return x

def sign(data, translate):
    temp = np.array(data)
    return np.vectorize(lambda x: translate[x])(temp)

im = Image.open('flipped.png')
im.show()
data = sign(im.getdata(), {0: -1, 255: 1}) 
y = data.reshape(im.size[::-1]) 
result=ICM(y)
result = sign(result, {-1: 0, 1: 255})
output_image = Image.fromarray(np.uint8(result))
output_image=output_image.convert('1', dither=Image.NONE)
output_image.save('1.png')
output_image.show()