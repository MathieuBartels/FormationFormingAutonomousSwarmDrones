import numpy as np
import os
import cv2
import numpy as np
import glob
import math

def xyzGaus(sigmas, location):
    c = np.divide(1, np.multiply(sigmas, math.sqrt(2 * math.pi)))    
    c2 = np.exp(-np.divide(np.multiply(location, location), np.multiply(2, np.power(sigmas, 2))))
    return sum(np.multiply(c, c2))
    
def xyzGausSecondOrder(sigmas, location):
    sigmasSquared = np.power(sigmas, 2)
    sigmasfifth = np.power(sigmas, 5)
    locationSquared = np.power(location, 2)
    c = - np.divide(np.subtract(sigmasSquared, locationSquared), np.multiply(sigmasfifth, math.sqrt(2 * math.pi)))
    c2 = np.exp( - (np.divide(locationSquared, np.multiply(2,sigmasSquared))))
    return sum(np.multiply(c,c2))

def euclideanDist(a, b): 
    return np.linalg.norm(np.asarray(a)-np.asarray(b))

def withinBoundary(location, boundary):
    z, x, y = location
    zMax, xMax, yMax = boundary 
    if z < 0 or x < 0 or y < 0 or z >= zMax or x >= xMax or y >= yMax:
        return False
    return True

def gausAtDistance(sigma, location, order):
    if order == 0:
        return - xyzGaus(sigma, location)
    if order == 2:
        return - xyzGausSecondOrder(sigma, location)

def sortKeyFunc(s):
    return int(os.path.basename(s)[:-4])

def createVideoFromFolder(folder, outputName, N):
    files = glob.glob(folder)
    files.sort(key=sortKeyFunc)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outputName, fourcc, 20.0, (432,288), 1)
    for filename in files[0:N]:
        img = cv2.imread(filename)
        cv2.imshow('frame',img)
        out.write(img)
        cv2.imwrite(filename, img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()
    cv2.destroyAllWindows()
