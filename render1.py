import drjit
from drjit.cuda.ad import Float, UInt32, Array3f, Array2f, TensorXf, Loop
from imageio import imwrite
from camera import Camera
from model import MLP
import torch

@drjit.wrap_ad(source="drjit", target="torch")
def nn_wrap_ad(pos, dir):
    return model.trace(pos, dir)

def hitSphere(center, radius, cam):
    oc = cam.origin - center
    a = drjit.dot(cam.dir, cam.dir)
    b = 2.0 * drjit.dot(oc, cam.dir)
    c = drjit.dot(oc, oc) - radius * radius
    disc = b*b - 4*a*c
    return (disc > 0)

def initCamera():
    lookFrom = Array3f(13, 2, 3)
    lookAt = Array3f(0, 0, 0)
    vUp = Array3f(0, 1, 0)
    vFov = 10

    cam = Camera(lookFrom, lookAt, vUp, vFov, aspectRatio, x, y)
    return cam

def render(model):
    # Initialize camera
    cam = initCamera()
    origin = TensorXf(drjit.ravel(cam.origin), shape=drjit.shape(cam.origin))
    cam_dir = drjit.normalize(cam.dir)
    dir = TensorXf(drjit.ravel(cam_dir), shape=[imgW * imgH, 3])
    
    t = nn_wrap_ad(origin, dir)
    point = drjit.fma(dir, t, origin)
    print(point)
    exit()


# Image
aspectRatio = 1.0
imgW = 800
imgH = int(imgW / aspectRatio)

x = drjit.linspace(dtype=drjit.cuda.Float, start=0, stop=1, num=imgW)
y = drjit.linspace(dtype=drjit.cuda.Float, start=1, stop=0, num=imgH)
x, y = drjit.meshgrid(x, y)

# Color Map
# color = Array3f(x, y, 0)
# img = drjit.ravel(color)
# img_t = TensorXf(img, shape=(imgH, imgW, 3))
# imwrite('demo.jpg', img_t)



if __name__ == "__main__":
    model = MLP()
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    pixbuf = render(model)
    # imwrite("demo.jpg", pixbuf)