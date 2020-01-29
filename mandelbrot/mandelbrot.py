import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(h, w, maxit=20):
    """ return an image of the Mandelbrot fractal of size(h, w)."""
    y, x = np.ogrid[-1.4:1.4:h*1j, -2:0.8:w*1j]
    c = x + y * 1j
    z = c
    divtime = maxit + np.zeros(z.shape, dtype=int)

    for i in range(maxit):
        z = z**2 + c
        # print("z, shape: {}\n{}".format(z.shape, z))
        diverge = z * np.conj(z) > 2**2
        # print("diverge, shape: {}\n{}".format(diverge.shape, diverge))
        div_now = diverge & (divtime == maxit)
        # print("div_now, shape: {}\n{}".format(div_now.shape, div_now))
        divtime[div_now] = i
        z[diverge] = 2


    return divtime

plt.imshow(mandelbrot(600, 600, 40))
plt.show()