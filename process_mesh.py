#!/bin/env python
import meshio
import torch
from torch.utils.data import DataLoader
import numpy
# import click

dev = 'cuda'

class DDFDataProcess(DataLoader):
    def __init__(self, filename='./bunny.obj'):
        f = meshio.read('./bunny.obj', file_format='obj')
        self.face = torch.tensor(f.cells_dict['triangle'],
            dtype=torch.int32, device=dev,)
        self.vert = torch.tensor(f.points,
            dtype=torch.float32, device=dev,)
        self.n_face, self.n_vert = len(self.face), len(self.vert)
        self.dataset = list()

        self.bbox_min = torch.min(self.vert, dim=0)[0]
        self.bbox_max = torch.max(self.vert, dim=0)[0]

        self.bbox_min = torch.tensor([-3, -4, -5]).to(dev)
        self.bbox_max = torch.tensor([ 3,  4,  5]).to(dev)


    def get_face(self, idx):
        v1_idx, v2_idx, v3_idx = self.face[idx]
        return (self.vert[v1_idx], self.vert[v2_idx], self.vert[v3_idx],)

    def generate(self, s_face, s_hemi, s_pfh):
        data = list()
        for i in range(self.n_face-100, self.n_face):
            dir  = self.sample_hemisphere(s_hemi)
            orig = self.sample_face(self.get_face(i), s_face)
            data.append(self.sample_points(orig, dir, s_pfh))
        self.dataset = torch.vstack(data)

    def save(self, filename):
        if len(self.dataset) == 0:
            print("Dataset not computed yet. Use generate() first.")
            return False
        numpy.savez(filename, self.dataset.cpu().numpy())
        print("Saved to file: %s"%filename)

    # TODO: Add code to make area inverse importance sampling
    def face_area(self, face,):
        a, b, c = face
        ab, ac = b-a, c-a
        cos_th = torch.dot(ab, ac)/torch.linalg.norm(ab)/torch.linalg.norm(ac)
        sin_th = torch.sqrt(1-torch.pow(cos_th, 2))
        return torch.linalg.norm(ab)*torch.linalg.norm(ac)*sin_th*.5


    def sample_face(self, face, n):
        a, b, c = face
        samples = list()
        for i in range(n):
            u, v = torch.rand(2, device=dev)
            p = (1-u)*a + u*b
            q = (1-v)*p + v*c
            samples.append(q)
        return torch.vstack(samples)


    def bbox(self, center, direction): #, n):
        # this is not correct
        print(self.bbox_min, self.bbox_max)
        tx0 = (self.bbox_min[0] - center[0])/direction[0]
        ty0 = (self.bbox_min[1] - center[1])/direction[1]
        tz0 = (self.bbox_min[2] - center[2])/direction[2]
        t0 = torch.max(torch.tensor([tx0, ty0, tz0]))

        tx1 = (self.bbox_max[0] - center[0])/direction[0]
        ty1 = (self.bbox_max[1] - center[1])/direction[1]
        tz1 = (self.bbox_max[2] - center[2])/direction[2]
        t1 = torch.min(torch.tensor([tx1, ty1, tz1]))


        print(center+t1*direction,)
        print(center+t0*direction,)

        return t0, t1


        t_step = torch.linspace(t0, t1, n).to(dev)
        points = center+t_step.reshape(n, 1)*direction.unsqueeze(dim=0).expand(n, 3)
        return points

        # Visualize the samples
        xy = y[:, :2].cpu().numpy()
        print(xy[10])

        from matplotlib import pyplot
        pyplot.style.use('bmh')
        pyplot.gca().set_aspect('equal')

        pyplot.scatter(xy[:, 0], xy[:, 1],)
        pyplot.savefig('samples.jpg')

    def bbox_intersect(self, center, direction, n):
        print(center, direction, sep='\n')
        tx0 = (self.bbox_min[0] - center[0])/direction[0]
        ty0 = (self.bbox_min[1] - center[1])/direction[1]
        tz0 = (self.bbox_min[2] - center[2])/direction[2]
        t0 = torch.max(torch.tensor([tx0, ty0, tz0]))

        tx1 = (self.bbox_max[0] - center[0])/direction[0]
        ty1 = (self.bbox_max[1] - center[1])/direction[1]
        tz1 = (self.bbox_max[2] - center[2])/direction[2]
        t1 = torch.min(torch.tensor([tx1, ty1, tz1]))

        t_step = torch.linspace(t0, t1, n).to(dev)
        points = center+t_step.reshape(n, 1)*direction.unsqueeze(dim=0).expand(n, 3)
        return center

        # Visualize the samples
        xy = y[:, :2].cpu().numpy()
        print(xy[10])

        from matplotlib import pyplot
        pyplot.style.use('bmh')
        pyplot.gca().set_aspect('equal')

        pyplot.scatter(xy[:, 0], xy[:, 1],)
        pyplot.savefig('samples.jpg')

    def intersect(self, ray_orig, ray_dir, idx, EPS=1e-5):
        # Moller-Trumbore Algorithm
        EPS = 1e-5
        face = self.face[idx]
        v0, v1, v2 = self.vert[face]

        E1 = v1-v0
        E2 = v2-v0
        D = ray_dir
        T = ray_orig-v0
        P = torch.cross(D, E2)
        Q = torch.cross(T, E1)

        d = torch.dot(P, E1)
        is_hit = False if abs(d)<EPS else True

        c = 1./d
        t = c*torch.dot(Q, E2)
        return t

        is_hit = True if is_hit and t>0. else False
        u = c*dot(P, T)
        is_hit = True if is_hit and 0.<u<1. else False
        v = c*dot(Q, D)
        is_hit = True if is_hit and 0.<v and u+v<=1 else False
        hit_point = ray.at(t)
        N = cross(E1, E2)
        return is_hit, hit_point, N, t

    def moller_trombore(self, ray_o, ray_d,):
        t = 20
        # for f in range(len(self.face)):
        for f in range(100):
            root = self.intersect(ray_o, ray_d, f)
            if root>=0 and root<t: t=root
        return t

    def face_intersect(self, p0, p1):
        print("end points", p0, p1)
        dir = (p1-p0)/torch.norm(p1-p0)
        faces = list()
        for i in self.face:
            hit, root = self.moller_trombore(p0, dir, i)
            if hit: faces.append([i, root])

        print(*[ [f[0].tolist(), f[1].item()] for f in faces], sep='\n')
        exit()
        pass

    def sample_points(self, points, dir, n): #, eps=1e-1):
        ''' points: the points sampled from face
            dir   : direction sampled from the hemisphere
            n     : number of samples per point per direction '''
        dataset = list()
        for orig in points:
            for th in dir:
                d = torch.tensor([
                    torch.cos(th[0])*torch.sin(th[1]), 
                    torch.sin(th[0])*torch.sin(th[1]),
                    torch.cos(th[1])], device=dev,)
                # t0, t1 = self.bbox(orig, d )

                # points = self.bbox_samples(orig, torch.tensor([
                #     torch.cos(th[0])*torch.sin(th[1]), 
                #     torch.sin(th[0])*torch.sin(th[1]),
                #     torch.cos(th[1])], device=dev,), n)
                # faces = self.face_intersect( orig+t0*d, orig+t1*d, )
                # print("bbox", orig+t0*d, t1)

                t = self.moller_trombore(orig, d)
                step = torch.linspace(0, t, n).to(dev)
                for i in step:
                    dataset.append(torch.hstack((orig+d*t*i, -th, t*i)))
                # torch.hstack(orig, n*d, 



        return torch.stack(dataset)

        '''
                for i in range(n):
                    ray = points[0] + i*torch.tensor([
                        torch.cos(th[0])*torch.sin(th[1]), 
                        torch.sin(th[0])*torch.sin(th[1]),
                        torch.cos(th[1])], device=dev,)
                    datapoint = torch.hstack((ray, th,
                        torch.tensor([i*eps], device=dev)))
                    dataset.append(datapoint)

        return torch.stack(dataset)
        '''

        eps = 1e-1
        dataset = list()
        for orig in points:
            for th in dir:
                for i in range(n):
                    ray = points[0] + i*eps*torch.tensor([
                        torch.cos(th[0])*torch.sin(th[1]), 
                        torch.sin(th[0])*torch.sin(th[1]),
                        torch.cos(th[1])], device=dev,)
                    datapoint = torch.hstack((ray, th,
                        torch.tensor([i*eps], device=dev)))
                    dataset.append(datapoint)

        return torch.stack(dataset)
        # print(torch.stack(dataset))


# @click.command()
# @click.option('-i', '--input', help='Input OBJ file')
# @click.option('-o', '--output', help='NPZ filename')
# @click.option('-sf', '--samples_per_face',
#     default=5, help='Total number of points to sample from each face')
# @click.option('-sh', '--samples_per_hemi',
#     default=5, help='Total number of points to sample from hemisphere')
# @click.option('-sp', '--samples_on_ray',
#     default=5, help='Total number of points to sample on each ray')
def main(samples_per_face,
         samples_per_hemi,
         samples_on_ray,
         input, output):
    samples = {
        's_face': samples_per_face,
        's_hemi': samples_per_hemi,
        's_pfh' : samples_on_ray,
    }
    dataset = DDFDataProcess(filename=input)
    dataset.generate(**samples)
    dataset.save(output)


if __name__ == '__main__':
    kw = {
       'samples_per_face': 5,
       'samples_per_hemi': 30,
       'samples_on_ray'  : 50,
       'input' : 'bunny.obj',
       'output': 'bunny.npz',
    }
    main(**kw)
    pass


def sample_hemisphere(n):
    samples = list()
    for i in range(n):
        dir = torch.tensor([torch.pi, torch.pi*2.],).to(dev)*torch.rand(2).to(dev)
        samples.append(dir)
    return torch.vstack(samples)