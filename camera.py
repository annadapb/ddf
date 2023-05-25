import drjit
from drjit.cuda.ad import Float, Array3f

class Camera:
    def __init__(self, lookFrom, 
                 lookAt, 
                 vup, 
                 vFov, 
                 aspectRatio,
                 x,
                 y):
        theta = self.degreesToRadians(vFov)
        h = drjit.tan(theta/2)
        viewportHeight = 2.0 * h
        viewportWidth = aspectRatio * viewportHeight

        w = drjit.normalize(lookFrom - lookAt)
        u = drjit.normalize(drjit.cross(vup, w))
        v = drjit.cross(w, u)

        self.origin = lookFrom
        self.horizontal = viewportWidth * u
        self.vertical = viewportHeight * v
        self.lowerLeftCorner = self.origin - self.horizontal/2 - self.vertical/2 - w
        self.dir = self.lowerLeftCorner + x*self.horizontal + y*self.vertical - self.origin

        # self.origin = Array3f(0., 0., 0.)
        # self.horizontal = Array3f(viewportWidth, 0., 0.)
        # self.vertical = Array3f(0., viewportHeight, 0.)
        # self.lowerLeftCorner = self.origin - self.horizontal/2 - self.vertical/2 - Array3f(0., 0., focalLength)

    def degreesToRadians(self, degrees):
        return degrees * drjit.pi / 180.0
