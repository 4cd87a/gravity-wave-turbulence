import numpy as np
import sys
if str(sys.version)[:1]=='3':
    import matplotlib.pyplot as plt
import fun_functions

class fun:
    def __init__(self, dimx, dimy, Lx=1, Ly=1):
        self.dimx = dimx;
        self.dimy = dimy

        self.d = np.zeros((dimx, dimy), dtype=float)
        self.do = np.zeros((dimx, dimy), dtype=float)

        self.ddx = np.zeros((dimx, dimy), dtype=float)
        self.ddy = np.zeros((dimx, dimy), dtype=float)

        self.ddx2 = np.zeros((dimx, dimy), dtype=float)
        self.ddy2 = np.zeros((dimx, dimy), dtype=float)

        self.ddxo = np.zeros((dimx, dimy), dtype=float)
        self.ddyo = np.zeros((dimx, dimy), dtype=float)

        self.ddx2o = np.zeros((dimx, dimy), dtype=float)
        self.ddy2o = np.zeros((dimx, dimy), dtype=float)

        self.fft = np.zeros((dimx, dimy), dtype=complex)
        self.ffto = np.zeros((dimx, dimy), dtype=complex)

        self.ddxCalculated = False
        self.ddyCalculated = False
        self.ddx2Calculated = False
        self.ddy2Calculated = False
        self.ddxoCalculated = False
        self.ddyoCalculated = False
        self.ddxo2Calculated = False
        self.ddyo2Calculated = False

        self.dx = 1.* Lx / (dimx - 1)
        self.kx = np.fft.fftfreq(dimx, self.dx) * 2 * np.pi

        self.dy = 1.*Ly / (dimy - 1)
        self.ky = np.fft.fftfreq(dimy, self.dy) * 2 * np.pi

        self.x = np.arange(0, dimx) * self.dx  # initialize space coordinates
        self.y = np.arange(0, dimy) * self.dy

    def getk(self): # return 2-D array of
        return (np.zeros(self.dimx, dtype=complex), np.zeros(self.dimy, dtype=complex))

    def derx(self, n=1, plot=False, mode='a',mask=None):  # mode: 'a'=actual, 'o'=old
        if mode == 'a' and self.ddxCalculated and n == 1:
            if plot: self.plot(self.ddx)
            return self.ddx

        if mode == 'a' and self.ddx2Calculated and n == 2:
            if plot: self.plot(self.ddx2)
            return self.ddx2

        if mode == 'o' and self.ddxoCalculated and n == 1:
            if plot: self.plot(self.ddxo)
            return self.ddxo

        if mode == 'o' and self.ddxo2Calculated and n == 2:
            if plot: self.plot(self.ddxo2)
            return self.ddxo2

        kx = np.copy(self.kx)
        kx[1:] = (kx[1:]) ** (n)
        kx[0] = 0

        if mode == 'a':
            if type(mask)==type(None):
                ff = self.fft * (((1j) ** n) * (kx[:, np.newaxis]))
            else:
                ff = self.fft * (((1j) ** n) * (kx[:, np.newaxis]))*mask

        elif mode == 'o':
            if type(mask)==type(None):
                ff = self.ffto * (((1j) ** n) * (kx[:, np.newaxis]))
            else:
                ff = self.ffto * (((1j) ** n) * (kx[:, np.newaxis]))*mask
        else:
            return 0

        a = np.real(np.fft.ifft2(ff))

        if mode == 'a' and n == 1:
            self.ddx = a
            self.ddxCalculated = True

        if mode == 'a' and n == 2:
            self.ddx2 = a
            self.ddx2Calculated = True

        if plot: self.plot(a)
        return a

    def dery(self, n=1, plot=False, mode="a",mask=None):  # mode: 'a'=actual, 'o'=old
        if mode == 'a' and self.ddyCalculated and n == 1:
            if plot: self.plot(self.ddy)
            return self.ddy

        if mode == 'a' and self.ddy2Calculated and n == 2:
            if plot: self.plot(self.ddy2)
            return self.ddy2

        if mode == 'o' and self.ddyoCalculated and n == 1:
            if plot: self.plot(self.ddyo)
            return self.ddyo

        if mode == 'o' and self.ddyo2Calculated and n == 2:
            if plot: self.plot(self.ddyo2)
            return self.ddyo2

        ky = np.copy(self.ky)
        ky[1:] = (ky[1:]) ** (n)
        ky[0] = 0

        if mode == 'a':
            if type(mask) == type(None):
                ff = self.fft * (((1j) ** n) * (ky))
            else:
                ff = self.fft * (((1j) ** n) * (ky))*mask
        elif mode == 'o':
            if type(mask) == type(None):
                ff = self.ffto * (((1j) ** n) * (ky))
            else:
                ff = self.ffto * (((1j) ** n) * (ky))*mask

        a = np.real(np.fft.ifft2(ff))

        if mode == 'a' and n == 1:
            self.ddy = a
            self.ddyCalculated = True

        if mode == 'a' and n == 2:
            self.ddy2 = a
            self.ddy2Calculated = True

        if plot: self.plot(a)
        return a

    def upd(self, newf, plot=False, forcage=None, forcage_fft=None, forcage_mode='add_fft'):
        self.do = self.d
        if type(forcage) != type(None) and forcage_mode == 'multiply':
            self.d = newf * (1 + forcage)
        else:
            self.d = newf

        self.ddxoCalculated = self.ddxCalculated
        if self.ddxCalculated:
            self.ddxo = self.ddx

        self.ddxo2Calculated = self.ddx2Calculated
        if self.ddx2Calculated:
            self.ddxo2 = self.ddx2

        self.ddyoCalculated = self.ddyCalculated
        if self.ddyCalculated:
            self.ddyo = self.ddy

        self.ddyo2Calculated = self.ddy2Calculated
        if self.ddy2Calculated:
            self.ddyo2 = self.ddy2

        self.ffto = self.fft

        self.fft = np.fft.fft2(self.d)
        self.fft[:, int(self.dimy / 3):int(2 * self.dimy / 3)] = 0
        self.fft[int(self.dimx / 3):int(2 * self.dimx / 3), :] = 0

        if type(forcage) != type(None) and type(forcage_fft) != type(None) and forcage_mode == 'add_fft':
            self.fft += forcage * forcage_fft
            self.d = np.real(np.fft.ifft2(self.fft))

        self.ddxCalculated = False
        self.ddx2Calculated = False
        self.ddyCalculated = False
        self.ddy2Calculated = False

        if plot: self.plot()

    def setByFourier(self, fft, kx=None, ky=None, Ampl=1, plot=False):
        # if len(kx) != self.dimx or len(ky) != self.dimy:
        #     print("not right length for SetByFourier")
        #
        # k = np.zeros((self.dimx, self.dimy), dtype=complex)
        #
        # for i in range(int(2 * self.dimx / 3)):
        #     for j in range(int(2 * self.dimy / 3)):
        #         k[i][j] = kx[i] * ky[j]
        # a = np.real(np.fft.ifft2(k))

        a = np.real(np.fft.ifft2(fft))
        self.upd(a, plot=plot)

        return a

    def getfft(self, plot=False, d2=False, axs=1):
        if plot:
            if not d2:
                self.plot(np.abs(self.fft), x=self.kx, y=self.ky)
            else:
                plt.figure()
                a = 1.*np.abs((self.fft)).sum(axis=axs) / (self.dimy if axs else self.dimx)
                plt.semilogy(a[:int(len(a) / 2)])
                plt.show()
            return
        else:
            return self.fft

    def plot(self, whatToPlot=None, x=None, y=None):
        if type(x) == type(None): x = self.x
        if type(y) == type(None): y = self.y

        if type(whatToPlot) == type(None):
            return fun_functions.plot3d(x, y, self.d)
        else:
            return fun_functions.plot3d(x, y, whatToPlot)

    def rms(self):
        return np.sum(self.d ** 2) - np.sum(self.d) ** 2