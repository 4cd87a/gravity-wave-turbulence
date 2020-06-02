import numpy as np
import sys
if str(sys.version)[:1]=='3':
    import matplotlib.pyplot as plt

def plot3d(x, y, z, fig=None):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    xx, yy = np.meshgrid(x, y)
    # ax.plot_wireframe(xx, yy, z, color='blue')
    ax.plot_surface(xx, yy, z, color='blue')

    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_zlabel('z')

    # plt.draw()
    plt.show()
    return ax


def getfft(dat, dim):
    fft = np.fft.fft2(dat)
    # return fft
    fft[:, int(dim / 3):int(2 * dim / 3)] = 0
    fft[int(dim / 3):int(2 * dim / 3), :] = 0
    return fft


def plotfft(fft, dim, axs=1):
    if len(fft) != dim: print("dim is not right")
    a = np.abs(np.fft.fftshift(fft))

    mind = int(dim / 2)
    r = np.zeros(mind)

    for i in range(1, mind + 1):
        sm = 0
        count = 0
        for j in range(mind - i, mind + i):
            count += 2
            # print(mind-i,j)
            sm += (a[mind - i, j])
            # print(mind+i,j)
            sm += (a[mind + i - 1, j])

        for k in range(mind - i + 1, mind + i - 1):
            count += 2
            # print(mind-i+1,k)
            sm += (a[k, mind - i])
            # print(mind+i-1,k)
            sm += (a[k, mind + i - 1])
        r[i - 1] = sm / count

    # plt.semilogy(r)
    plt.figure()
    plt.loglog(r)
    plt.show()

colors = ['b','r','c','g','m','y','k']
def pltupdate(fig,ax,dats,tp="loglog"):
    if ax.lines:
        for i, line in enumerate(ax.lines):
            if i>=len(dats): break
            if type(dats[i])!=int or dats[i]!=0:
                if type(dats[i])==tuple:
                    line.set_xdata(dats[i][0])
                    line.set_ydata(dats[i][1])
                else:
                    line.set_ydata(dats[i])

        for j in range(i+1,len(dats)):
            if type(dats[j])!=int or dats[j]!=0:
                color = colors[j%len(colors)] #'b' if j>=len(colors) else colors[j]
                if type(dats[j])==tuple:
                    if tp=="loglog":
                        ax.loglog(dats[j][0],dats[j][1], color)
                    elif tp=="plot":
                        ax.plot(dats[j][0],dats[j][1], color)
                else:
                    if tp=="loglog":
                        ax.loglog(dats[j], color)
                    elif tp=="plot":
                        ax.plot(dats[j], color)
            else:
                print("there are some plots missing ({}), but it'll be skiped".format(j))
    else:
        for i, dat in enumerate(dats):
            if type(dat)!=int or dat!=0:
                color = colors[i%len(colors)] #'b' if i>=len(colors) else colors[i]
                if type(dat)==tuple:
                    if tp=="loglog":
                        ax.loglog(dat[0],dat[1], color)
                    elif tp=="plot":
                        ax.plot(dat[0],dat[1], color)
                else:
                    if tp=="loglog":
                        ax.loglog(dat, color)
                    elif tp=="plot":
                        ax.plot(dat, color)
            else:
                print("there are some plots missing ({}), but it'll be skiped".format(i))
    fig.canvas.draw()


def calculateDer(dat, dimx, dx, direc='x', n=1, plot=False, fft=None, retfft=False):
    if type(fft) == type(None):
        fft = np.fft.fft2(dat)
        fft[:, int(dimx / 3):int(2 * dimx / 3)] = 0
        fft[int(dimx / 3):int(2 * dimx / 3), :] = 0

    kx = np.fft.fftfreq(dimx, dx) * 2 * np.pi
    kx[1:] = (kx[1:]) ** (n)
    kx[0] = 0

    if direc == 'x':
        ff = fft * (((1j) ** n) * (kx[:, np.newaxis]))
    else:
        ff = fft * (((1j) ** n) * (kx))

    a = np.real(np.fft.ifft2(ff))

    if (plot):
        x = np.arange(0, dimx) * dx
        plot3d(x, x, a)

    if retfft:
        return a, ff

    return a


def energyk(l, L, dx):
    dim = len(l)
    lk = np.fft.fftshift(getfft(dat=l, dim=dim))
    Lk = np.fft.fftshift(getfft(dat=L, dim=dim))
    kx = np.fft.fftshift(np.fft.fftfreq(dim, dx) * 2 * np.pi)
    kx = np.repeat([kx], dim, axis=0) ** 2
    kx = np.sqrt(np.transpose(kx) + kx)

    kxd = np.zeros([dim, dim])
    kx[int(dim / 2), int(dim / 2)] = 1
    kxd = kx ** (-1)
    kx[int(dim / 2), int(dim / 2)] = 0
    kxd[int(dim / 2), int(dim / 2)] = 0

    ak = np.zeros(dim, dtype=complex)
    ak = 1. * np.sqrt(kx) * lk / np.sqrt(2) + 1j * np.sqrt(kxd) * Lk / np.sqrt(2)
    return 1.*kx / (2 * np.pi), ak


def energy(l, L, dx):
    k, ak = energyk(l, L, dx)
    return np.sum(k * np.abs(ak) ** 2)


def hatmodule(ld, Ld, ddx):
    k, eng = energyk(l=ld, L=Ld, dx=ddx)
    hat = np.zeros((int(np.ceil(np.max(np.abs(k)))) + 1, 2))
    for i in range(len(eng)):
        for j in range(len(eng[i])):
            y = int(np.ceil(np.abs(k[i][j])))
            hat[y][0] += 1
            hat[y][1] += k[i][j] * (np.abs(eng[i][j]) ** 2)
    # print(hat)
    for i in range(len(hat)):
        if hat[i][0] != 0:
            hat[i][1] = hat[i][1] / hat[i][0]
    return hat[:int(len(ld) / 3), 1]


def rms(nparray):
    return np.sqrt(np.sum(nparray ** 2) - np.sum(nparray) ** 2)


def deflimit(limit, value, ax=None, mode="plot"):
    if value < limit[0]:
        if mode == "log":
            limit = (value / 2, limit[1])
        else:
            limit = (0.8 * value, limit[1])

        if ax != None:
            ax.set_ylim(limit)
    if value > limit[1]:
        if mode == "log":
            limit = (limit[0], value * 2)
        else:
            limit = (limit[0], value * 1.2)

        if ax != None:
            ax.set_ylim(limit)
    return limit