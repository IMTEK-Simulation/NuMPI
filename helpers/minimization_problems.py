
import numpy as np

"""

"""


# Extended Rosenbrock Function



def rosen_ref(n):
    """
    It's not the only minimum
    :param n: number of Dimensions
    :return:
    """
    assert n%2 == 0, "number of dimention should be even"

    return np.array()
import abc
class ObjectiveFunction(object,metaclass=abc.ABCMeta):


    @staticmethod
    @abc.abstractmethod
    def f_grad(x):
        pass

    @classmethod
    def f(cls,x):
        return cls.f_grad(x)[0]

    @classmethod
    def grad(cls,x):
        return cls.f_grad(x)[1]

    @classmethod
    def plot_2D(cls):
        import matplotlib.pyplot as plt
        from helpers.plot_helpers import draw_npArrow2D
        fun = cls.f

        npts = 100

        xg, yg = np.linspace(*cls.bounds, 51), np.linspace(*cls.bounds, 51)

        def mat_fun(xg, yg, fun):
            Z = np.zeros((xg.size, yg.size))

            for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    Z[j, i] = fun(np.array([xg[i], yg[j]]))
            return Z

        X, Y = np.meshgrid(xg, yg)

        plt.colorbar(plt.pcolormesh(X, Y, np.log10(mat_fun(xg, yg, fun)))).set_label(r"log10($\phi$)")
        plt.plot(1 / 2, 1 / 2, '+')  # standart starting point
        ax = plt.gca()
        for x in [np.array([[0], [1]]),
                  np.array([[-2], [-2]]),
                  np.array([[2], [-3]]),
                  np.array([[-3], [1]])]:
            draw_npArrow2D(ax, x, delta=cls.grad(x) / 10)

            eps = 1e-4
            # unit Vector with random Direction
            phi = np.asscalar(np.random.random(1) * 2 * np.pi)
            u = np.array([[np.cos(phi)], [np.sin(phi)]])

            print((cls.f(x + u * eps) - cls.f(x)) / eps)
            print(cls.grad(x).T @ u)

        plt.show(block=True)





class Trigonometric(ObjectiveFunction):
    """
    all values of n are allowed

    :param x: 1d array
    :return:
    """
    bounds=(-np.pi,np.pi)
    @staticmethod
    def f_grad(x):
        n = x.size

        x.shape=(-1,1)
        idxVector = np.reshape(np.arange(1, n + 1), (-1, 1))
        f = n - np.sum(np.cos(x)) +  idxVector * (1 - np.cos(x)) - np.sin(x)

        jac = np.reshape(np.sin(x),(1,-1)) + np.diag((-np.cos(x) + idxVector * np.sin(x)).flat)

        return np.sum(f**2), 2*jac.T@f

    @staticmethod
    def startpoint(n):
        """
        standard starting point
        :param x:
        :return:
        """

        return np.reshape(np.ones(n)/n,(-1,1))

    @staticmethod
    def minVal(*args):
        return 0

    @staticmethod
    def plot_2D():
        import matplotlib.pyplot as plt
        from helpers.plot_helpers import draw_npArrow2D
        fun = Trigonometric.f

        npts = 100

        xg, yg = np.linspace(-np.pi, np.pi, 51), np.linspace(-np.pi, np.pi, 51)

        def mat_fun(xg, yg, fun):
            Z = np.zeros((xg.size, yg.size))

            for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    Z[j, i] = fun(np.array([xg[i], yg[j]]))
            return Z

        X, Y = np.meshgrid(xg, yg)

        plt.colorbar(plt.contour(X, Y, mat_fun(xg, yg, fun)))
        plt.plot(1 / 2, 1 / 2, '+')  # standart starting point
        ax = plt.gca()
        for x in [np.array([[0], [1]]),
                  np.array([[-2],[-2]]),
                  np.array([[2], [-3]]),
                  np.array([[-3], [1]])]:
            draw_npArrow2D(ax, x, delta=Trigonometric.grad(x)/10)

            eps = 1e-4
            # unit Vector with random Direction
            phi = np.asscalar(np.random.random(1)* 2* np.pi)
            u = np.array([[np.cos(phi)],[np.sin(phi)]])

            print((Trigonometric.f(x + u * eps) - Trigonometric.f(x)) / eps)
            print(Trigonometric.grad(x).T@u)

            Trigonometric.grad(x)


        plt.show(block = True)

class Extended_Rosenbrock(ObjectiveFunction):
    """

    https://github.com/cjtonde/optimize_rosenbrock/blob/master/src/optimize_rosenbrock.py

    n should be even

    :param x: 1d array
    :return:
    """
    bounds=(-4,4)
    @staticmethod
    def f_grad(x):
        n = x.size

        x.shape=(-1,1)

        sumf2  = np.sum(100 * (x[1::2] - x[:-1:2]**2)**2 + (1 - x[:-1:2])**2 )

        grad = np.zeros_like(x)
        grad[1::2]  = 200 * (x[1::2] - x[:-1:2]**2)
        grad[:-1:2] = -400 * x[:-1:2] * (x[1::2] - x[:-1:2]**2) - 2 * (1-x[:-1:2]) # # d / dx2l-1

        return sumf2, grad

    @staticmethod
    def startpoint(n):
        """
        standard starting point
        :param x:
        :return:
        """

        return np.reshape(np.ones(n)/n,(-1,1))

    @staticmethod
    def minVal(*args):
        return 0


if __name__=="__main__":
    Extended_Rosenbrock.plot_2D()
    Trigonometric.plot_2D()