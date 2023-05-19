import torch
import drjit

device = 'cuda'

if device == 'cuda':
    from drjit.cuda.ad import ( Float, UInt32, Array3f, Array2f, Array1f,
        TensorXf, Texture3f, PCG32, Loop )
else:
    from drjit.llvm.ad import ( Float, UInt32, Array3f, Array2f, Array1f,
        TensorXf, Texture3f, PCG32, Loop )
from model import MLP
# drjit.set_log_level(drjit.LogLevel.Info)

@drjit.wrap_ad(source="drjit", target="torch")
def nn_eval(pos, dir):
    return sdf_model(pos)


def sdf(p: Array3f) -> Float:
    sphere1 = drjit.norm(p - Array3f(0.0, 1.0, -3.5)) - 1
    sphere2 = drjit.norm(p - Array3f(1.2, 1.0, -3.5)) - 1
    sphere3 = drjit.norm(p - Array3f(-1.2, 1.0, -3.5)) - 1
    plane = p.y
    return drjit.min([sphere1, sphere2, sphere3, plane])


# Functions
# def sphere_trace(origin, dir):
# 	i = UInt32(0)
# 	loop = Loop("Sphere tracing", lambda: (i))
# 	origin = drjit.fma(dir, 0, origin)
# 	while loop(i<100):
# 	# i = 0
# 	# sdf_val = 0
# 	# while i<100:
# 		sdf_val = nn_eval(origin)
# 		origin = drjit.fma(dir, sdf_val, origin)
# 		i += 1
# 	return origin


def sphere_trace(origin, dir):
    i = UInt32(0)
    dist = Float(0)
    origin = drjit.unravel(Array3f, origin)
    dir = drjit.unravel(Array3f, dir)
    loop = Loop("Sphere tracing", lambda: (dist, i))
    while loop(i < 1000):
        point = drjit.fma(dir, dist, origin)
        distFromScene = sdf(point)
        dist += distFromScene
        i += 1
    return dist




def sdf(p: Array3f) -> Float:
    sphere1 = drjit.norm(p - Array3f(0.0, 1.0, 0.0)) - 1.0
    sphere2 = drjit.norm(p - Array3f(1.2, 1.0, 0.0)) - 1
    sphere3 = drjit.norm(p - Array3f(-1.2, 1.0, 0.0)) - 1
    plane = p.y
    return drjit.min([sphere1, sphere2, sphere3, plane])




class renderer:
    def __init__(self, model_weights=None):
        # Image parameter
        aspect_ratio = 1.0
        self.img_w = 320
        self.img_h = int(self.img_w / aspect_ratio)

        # Camera parameters
        viewportHeight = 2.0
        viewportWidth = aspect_ratio * viewportHeight
        focal_length = 1.0
        self.cam_origin = Array3f([0.0, .5, 1.0])
        self.horizontal = Array3f(viewportWidth, 0.0, 0.0)
        self.vertical = Array3f(0.0, viewportHeight, 0.0)
        self.lowerLeftCorner = (
            self.cam_origin - self.horizontal/2-
            self.vertical/2 - Array3f(0.0, 0.0, focal_length)
        )

        cam_h = drjit.linspace(dtype=drjit.cuda.Float,
            start=0, stop=1, num=self.img_w)
        cam_v = drjit.linspace(dtype=drjit.cuda.Float,
            start=1, stop=0, num=self.img_h)

        self.w, self.h = drjit.meshgrid(cam_h, cam_v)

        # Scene
        # light = Array3f(0, -6, -2)
        self.light = Array3f(2, 3, 2)
        self.model = MLP().to(device)
        self.model.load_state_dict(torch.load(model_weights))


    def get_dir(self, u, v):
        return self.lowerLeftCorner+u*self.horizontal+v*self.vertical-self.cam_origin

    def points_trace(self, origin, direction, n_steps=10):
        pos = drjit.fma(direction, 0, origin)
        # t = self.model_eval(TensorXf(drjit.ravel(pos), drjit.shape(pos)))
        # points = pos.torch()
        # sdf = t.torch()
        # sdf.requires_grad = True

        for i in range(n_steps):
            # t = self.model_eval_tt(pos.torch())
            t = self.model_eval(TensorXf(drjit.ravel(pos), drjit.shape(pos)))
            model_out = Float(drjit.ravel(t))
            pos = drjit.fma(direction, model_out, origin)
            # points = torch.cat( tensors=(points, pos.torch()) )
            # sdf = torch.cat( tensors=(sdf, t.torch()) )

        return (torch.tensor(pos, device='cuda', requires_grad=True),
                torch.tensor(t,   device='cuda', requires_grad=True)  )

    @drjit.wrap_ad(source="drjit", target="torch")
    def model_eval(self, pos, dir):
        root = self.model.trace(pos.reshape(3), dir.reshape(-1, 3))
        return root

    def get_points(self):
        cam_dir = drjit.normalize(self.get_dir(self.w, self.h))
        origin = TensorXf(drjit.ravel(self.cam_origin),
            shape=(1, 3))
        dir = TensorXf(drjit.ravel(cam_dir),
            shape=[self.img_w * self.img_h, 3])
        # origin = drjit.unravel(Array3f, origin)
        # dir = drjit.unravel(Array3f, dir)


        root = self.model_eval(origin, dir)
        hit_point = root*dir+origin
        # print("hit points:", hit_point.shape, dir.shape)

        grad = drjit.grad(hit_point)

        # print(grad, grad.shape, sep='\n')
        # exit()


        color = self.shade(hit_point, light=Array3f(3, 3, 3))
        image = drjit.ravel(color * Array3f(0.8, 0.6, 0.1))
        self.pixbuf = image.numpy().reshape(self.img_h, self.img_w, 3)
        return;

        self.pixbuf = TensorXf(image, shape=(self.img_h, self.img_w, 3))
        return self.pixbuf
        exit()

        self.pixbuf = color.numpy().reshape(self.img_h, self.img_w, 3)
        return self.pixbuf
        return self.points_trace(origin, dir)

    def shade(self, grad, light):
        shade = drjit.dot(light, drjit.unravel(Array3f, grad))
        return drjit.maximum(0, shade)

    @drjit.wrap_ad(source="torch", target="drjit")
    def render_points(self, points, sdf):
        # return drjit.fma(sdf, 1., points)
        pos = Array3f(drjit.ravel(drjit.fma(sdf, 1, points)))
        color = get_light(pos, self.light)
        return TensorXf(color, shape=(3, self.img_h*self.img_w))


    def render(self,):
        # Rendering
        # dir_t = drjit.normalize(Array3f(w, h, 1))
        cam_dir = drjit.normalize(self.get_dir(self.w, self.h))
        origin = TensorXf(drjit.ravel(self.cam_origin),
            shape=drjit.shape(self.cam_origin))
        # org = TensorXf(drjit.ravel(cam_origin), shape=drjit.shape(cam_origin))

        dir = TensorXf(drjit.ravel(cam_dir), shape=[self.img_w * self.img_h, 3])
        # dir = TensorXf(drjit.ravel(dir_t), shape=[img_w*img_h, 3])
        # pos = sphere_trace(org, dir)
        dist = self.model_eval(origin, dir)
        origin = drjit.unravel(Array3f, origin)
        dir = drjit.unravel(Array3f, dir)
        pos = drjit.fma(dir, dist, origin)

        color = get_light(pos=pos, light=self.light)
        comp_graph = drjit.graphviz()
        comp_graph.render("computational_graph.gv")
        # exit()

        image = drjit.ravel(color * Array3f(0.8, 0.6, 0.1))
        self.pixbuf = TensorXf(image, shape=(self.img_h, self.img_w, 3))
        return self.pixbuf

    def imwrite(self, filename):
        from imageio import imwrite
        imwrite(filename, self.pixbuf)

if __name__=='__main__':
    r=renderer('./best_model.pt')
    r.get_points()
    # r.render()
    r.imwrite("demo-render.jpg")

