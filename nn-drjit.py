import numpy
import drjit
from drjit.llvm.ad import (Float, TensorXf)
from matplotlib import pyplot
pyplot.style.use('bmh')

def relu(x):
    return numpy.where(x<0, 0, x)

x = numpy.linspace(-2, 2, 100)
y = relu(x)

pyplot.plot(x, y)
pyplot.savefig('numpy-drjit.png')



input = TensorXf(numpy.random.randn(5))
layer_w0 = TensorXf(numpy.random.randn(3*5), shape=(5, 3))

print(input, layer_w0)

layer_b0 = Float(numpy.random.randn(3))

output = layer_w0 @ input

print(
    type(input),
    type(layer_w0),
sep='\n')
