import random
import numpy as np
from matplotlib import pyplot

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    x = np.random.rand(2000) * 9 - 4.5
    y = np.random.rand(1000) * 18 - 9

    x[x <= 0] -= 9
    x[x >= 0] += 9

    x += np.random.randn(*x.shape)
    y += np.random.randn(*y.shape)

    x_t = np.random.randn(2000).clip(-4.5, 4.5)
    y_t = np.random.randn(1000).clip(-4.5, 4.5)

    pyplot.clf()
    pyplot.figure(figsize=[5, 3])
    pyplot.grid()
    pyplot.xlabel(r"$x$")
    pyplot.ylabel(r"$y$")
    pyplot.xlim([-6, 6])
    pyplot.scatter(x_t, x, color="blue", s=1.0)
    pyplot.scatter(y_t, y, color="red", s=1.0)
    pyplot.savefig(f"Sample-Raising-1.jpg", dpi=720, bbox_inches="tight")
    pyplot.close()

    pyplot.clf()
    pyplot.figure(figsize=[5, 3])
    pyplot.grid()
    pyplot.xlabel(r"$x$")
    pyplot.ylabel(r"$y^2$")
    pyplot.xlim([-6, 6])
    pyplot.scatter(x_t, x ** 2, color="blue", s=1.0)
    pyplot.scatter(y_t, y ** 2, color="red", s=1.0)
    pyplot.hlines(y=80, xmin=-6, xmax=6, color="green", linestyles="--")
    pyplot.savefig(f"Sample-Raising-2.jpg", dpi=720, bbox_inches="tight")
    pyplot.close()
