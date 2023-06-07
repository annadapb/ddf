import numpy
import sys
from open3d.t import (geometry, io)

def init(ffile):
    global mesh, aabb, scene, id, filename

    filename = ffile
    mesh = io.read_triangle_mesh(filename)
    aabb = mesh.get_axis_aligned_bounding_box()
    scene = geometry.RaycastingScene()
    id = scene.add_triangles(mesh)

def random_sampling(num_samples=9):
    global mesh, aabb, scene, id

    ray0 = numpy.random.rand(num_samples, 6).astype(numpy.float32)
    ray1 = numpy.asarray(ray0[:,3:])

    r = numpy.squeeze(numpy.hypot(*numpy.hsplit(ray1, numpy.arange(1,3))))
    th_0 = numpy.arctan(ray0[:,4], ray0[:,3])
    th_1 = numpy.arccos(ray0[:,5]/r)

    t = scene.cast_rays(ray0)['t_hit'].numpy()


    dataset = numpy.hstack(( ray1, th_0.reshape(-1,1),
        th_1.reshape(-1,1), t.reshape(-1,1), ))


    outfile = "%s.random.npz"%filename[:-4]
    numpy.savez(outfile, dataset)
    print("%d random points sampled from %s to %s"%(
        dataset.shape[0], filename, outfile))

def pov_sampling(n_azi, n_pol, rad, num_samples=9):
    center = .5*(aabb.max_bound+aabb.min_bound)
    w = int(numpy.sqrt(num_samples))
    h = int(1.*num_samples/w)

    pol = numpy.linspace(0, numpy.pi, n_pol)
    azi = numpy.linspace(0, 2.*numpy.pi, n_azi)

    data = numpy.ndarray(shape=(0,6))
    for i in azi:
        for j in pol:
            eye = numpy.array([
                rad*numpy.cos(i)*numpy.sin(j),
                rad*numpy.sin(i)*numpy.sin(j),
                rad*numpy.cos(j) ])
            ray0 = geometry.RaycastingScene.create_rays_pinhole(
                fov_deg = 90, up = [0, 1, 0],
                center = center, eye = eye,
                width_px = w, height_px = h)
            ans = scene.cast_rays(ray0)['t_hit'].numpy().reshape(-1)
            ray = ray0.numpy().copy().reshape(-1, 6)

            r = numpy.squeeze(numpy.hypot(
                *numpy.hsplit(ray[:,3:], numpy.arange(1,3))))
            th_0 = numpy.arctan(ray[:,4], ray[:,3])
            th_1 = numpy.arccos(ray[:,5]/r)
            ray[:,3], ray[:,4], ray[:,5] = th_0, th_1, ans

            data = numpy.append(data, ray, axis=0)

    outfile = "%s.pov.npz"%filename[:-4]
    numpy.savez(outfile, data)
    print("%d pov sampled from %s to %s"%(
        data.shape[0], filename, outfile))




if __name__=='__main__':
    init('bunny.obj')
    random_sampling(100_000)
