import drjit
from drjit.cuda.ad import UInt32, Array3f, Loop, Float, TensorXf
from camera import Camera
from imageio import imwrite
import numpy

def hitSphere(center, radius, cam):
    oc = cam.origin - center
    a = drjit.dot(cam.dir, cam.dir)
    b = 2.0 * drjit.dot(oc, cam.dir)
    c = drjit.dot(oc, oc) - radius * radius
    disc = b*b - 4*a*c
    return (disc > 0)

def writeColor(pixelColor):
    scale = 1.0 / samplesPerPixel
    pixelColor = drjit.sqrt(pixelColor * scale)
    pixelColor = drjit.clamp(pixelColor, 0.0, 0.99)

    return pixelColor

def initCamera():
    lookFrom = Array3f(25, 2, 3)
    lookAt = Array3f(0, 0, 0)
    vUp = Array3f(0, 1, 0)
    vFov = 10

    cam = Camera(lookFrom, lookAt, vUp, vFov, aspectRatio, x, y)
    return cam


def render():
    cam = initCamera()
    pixelColor = Array3f(0, 0, 0)
    
    noise = numpy.random.uniform(0, 1, size=(800, 800, 3))
    sample = drjit.unravel(Array3f, Float(noise.ravel()))
    cam.dir = cam.dir + .01*sample
    # print(cam.dir[1], sample[1], sep='\n'); exit()
    pixelColor[hitSphere(Array3f(0, 0, -1), 0.5, cam)] = Array3f(1, 0, 0)
    

    # pixelColor = writeColor(pixelColor)
    img = drjit.ravel(pixelColor)

    img_t = TensorXf(img, shape=(imgH, imgW, 3))
    imwrite('demo.jpg', img_t)
  

aspectRatio = 1.0
imgW = 800
imgH = int(imgW / aspectRatio)
samplesPerPixel = 100

x = drjit.linspace(dtype=drjit.cuda.Float, start=0, stop=1, num=imgW)
y = drjit.linspace(dtype=drjit.cuda.Float, start=1, stop=0, num=imgH)
x, y = drjit.meshgrid(x, y)

if __name__ == '__main__':
    render()
