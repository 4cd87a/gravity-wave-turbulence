import numpy as np
import os, datetime


import sys, argparse
if str(sys.version)[:1]=='3':
    import matplotlib.pyplot as plt
from fun_functions import *
from fun_class import fun
from stack_class import stack, stackdif

import logging
logging.basicConfig(format='%(levelname)s : %(message)s',
                    level=logging.INFO)

#path = os.path.abspath(os.getcwd())
#path = '~/project_k/v_1'
path = 'txts'

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print(dname)
os.chdir(dname)
datadir = "data" #.format(path)

print(os.path.exists(datadir))
if not os.path.exists(datadir):
    print("dir {} was created".format(datadir))
    os.makedirs(datadir)
if not os.path.exists(path):
    print("dir {} was created".format(path))
    os.makedirs(path)

gversion = 0
gdebug = 0

def _print(text,save=True,saveto="lastprint.txt",version=None, debug_level = 0):
    if version == None: version = gversion
    if save:
        f = open('{}/{}'.format(path,saveto), "w")
        f.write("{}\n{}\n{:.0f}".format(version, text, (datetime.datetime.utcnow()-datetime.datetime(1970, 1, 1)).total_seconds()))
        f.close()

    if gdebug<=debug_level:
        logging.info("{}".format(text))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", help="Debug", nargs='?', type=int, const=0, default=1)
    args = parser.parse_args()
    gdebug = args.d

    nx = 128
    l = fun(nx, nx)
    L = fun(nx, nx)
    #kx, ky = l.getk()
    rnd = np.random.random(size=nx*nx).reshape((nx,nx))
    fhat = np.zeros((nx,nx))

    mask1 = np.zeros((nx,nx))
    ki = l.kx[int(nx/4)]
    kdif = l.kx[int(nx/20)]
    for i in range(nx):
        for j in range(nx):
            kmean = np.sqrt(l.kx[i]**2 + l.ky[j]**2)

            if ki-kdif < kmean < ki+kdif:
                fhat[i,j] = .5*10**0*np.exp(1j*2*np.pi*rnd[i,j])
            else:
                fhat[i,j] = .5*10**-2*np.exp(1j*2*np.pi*rnd[i,j])
            if ki < kmean:  # j>ki and j<10-ki and i>ki and i<10-ki:
                mask1[i, j] = 1  # int(np.sqrt(i**2+j**2))

    l.setByFourier(fft=fhat)
    forcage_curve = l.fft  # 5*10**-4

    res = []
    name = "sim_7"
    description = "change cond init"

    time = 500 #1200 #*(10**0)
    dt = 10*10**-5

    startime=0
    starti = 0
    time_pass = 0
    time_rest = 0

    nt = int(time/dt)
    savet = int(1/dt)#int(nt*0.05) if nt>20 else 1 #int(.05/dt)
    plott = int(nt*0.1) if nt>10 else 1
    checkt = int(.01/dt) #int(nt*0.001) if nt>1000 else 1

    l0 = l
    eng = 0
    engz = 0

    try:
        with open('{}/lastprint.txt'.format(path)) as f:
            read_data = f.read()
            gversion = int(read_data.split('\n')[0])
            _print("gversion was {}".format(gversion),debug_level=3,save=False)
    except:
        gversion = 0


    L = fun(nx,nx)
    a = fun(nx,nx)
    A = fun(nx,nx)
    b = fun(nx,nx)
    B = fun(nx,nx)
    g = fun(nx,nx)
    G = fun(nx,nx)

    nua = 0 #.000001
    nub = 0 #.000001
    nug = 0 #.000001
    nul = 5*10**-7

    res0 = (0,l.d.copy(),L.d.copy(),0)
    hatmodule0 = hatmodule(res0[1],res0[2],l.dx)

    # line23 = (np.linspace(1,l.dimx,l.dimx)[:int(l.dimx/3)])**(-2/3)
    # q = np.argmax(hatmodule0)
    # line23 =line23/line23[q]*hatmodule0[q]

    forcage = None
    forcage_coef = 10**-5 #10**-5
    forcage_sign = 0
    forcage_signi = 0

    if not os.path.exists("{}/{}/".format(datadir,name)):
        _print('created',save=False)
        os.chdir(datadir)
        os.mkdir(name)
        os.chdir('..')
    truei = 0
    for i in range(len(os.listdir("{}/{}/".format(datadir,name)))):
        _print('{}/{}/{}_{}.npy'.format(datadir,name,name,i),save=False)
        if os.path.isfile('{}/{}/{}_{}.npy'.format(datadir,name,name,i)):
            truei = i #break
        _print(truei,save=False)

    truei -= 1
    if truei>1 and 1:
        try:
            res0 = (np.load('{}/{}/{}_{}.npy'.format(datadir,name,name,truei)))
            startime = res0[0]
            starti = int(startime/dt)
            l.upd(res0[1])
            L.upd(res0[2])
            forcage = res0[4]
            forcage_coef = res0[5]
            _print("will start calculation from t={} (i={})".format(res0[0],truei),save=False, debug_level=3)
        except Exception as e:
            _print(e)
    else:
        _print("start new calculation",save=False)
        gversion+=1

    _print("Version now is {}".format(gversion), save=False)

    f = open("{}/lastinfo.txt".format(path), "w")
    f.write("{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}".format(gversion, name, description, nx, int(nt / savet), nt, savet, dt, l.dx, ki, nul))
    f.close()

    #--------------------------------------------------------------------------------------------------------------------------------------------------


    sdif = stackdif(25)

    _print("time={:.3f}; nt={:.0f}; savet = {:.4f}; plott = {:.4f}; checkt= {:.4f}".format(nt * dt, nt, savet * dt,
                                                                                          plott * dt, checkt * dt), debug_level=3)
    # res.append((0,l.d.copy(),L.d.copy(),0))
    start = datetime.datetime.now()

    Lold = 0

    for it in range(starti, nt):
        # 1) l'=L
        lf = ((3./ 2.*L.d - 1./2.*L.do)*dt +l.d)

        # 2) dxy(g) = -2(dx l)(dy l)
        gf = -2 * l.derx() * l.dery()
        gf, gfft = calculateDer(gf, g.dimx, g.dx, direc='x', n=-1, retfft=True)
        gf, gfft = calculateDer(gf, g.dimy, g.dy, direc='y', n=-1, retfft=True, fft=gfft)

        # 3) g'=G + nug*(l,xx+l,yy)

        Gf = (2. * ((gf - g.d) / dt) + G.d) / 3
              #- nug * (calculateDer(gf, g.dimx, g.dx, direc='x', n=2, fft=gfft) + calculateDer(gf, g.dimy, g.dy, direc='y', n=2, fft=gfft)) / 2. +
              # - nug * (g.derx(n=2) + g.dery(n=2))/ 2.)

        # 4) L' = ...

        Lnew = (calculateDer((1 + a.d - b.d + g.d) * l.derx(), l.dimx, l.dx, direc='x', n=1) +
                calculateDer((1 - a.d + b.d + g.d) * l.dery(), g.dimy, l.dy, direc='y', n=1) -
                (A.d + B.d - G.d) * L.d) * ((1 - (a.d + b.d - g.d)))  # ((a.d+b.d-g.d)**2))**(1)

        if type(Lold) == int and Lold == 0: Lold = Lnew
        Lf = (3 * Lnew / 2 - Lold / 2) * dt + L.d + nul * (L.derx(n=2,mask=mask1) + L.dery(n=2,mask=mask1)) * dt
        Lold = Lnew

        # rnd = np.random.random(size=nx * nx).reshape((nx, nx))
        # forcage_curve = forcage_curve

        l.upd(lf)#, forcage=forcage, forcage_fft=forcage_curve, forcage_mode='add_fft')
        g.upd(gf)
        G.upd(Gf)

        # 5)
        Af = Lf * l.derx()
        Af = -2 * calculateDer(Af, l.dimx, l.dx, direc='x', n=-1)

        # 6)
        Bf = Lf * l.dery()
        Bf = -2 * calculateDer(Bf, l.dimy, l.dy, direc='y', n=-1)

        # 7)
        af = (3. * Af / 2 - A.d / 2.) * dt  + a.d  # + nua * (a.derx(n=2) + a.dery(n=2)))

        # 8)
        bf = (3. * Bf / 2 - B.d / 2.) * dt  + b.d  # + nub * (b.derx(n=2) + b.dery(n=2))) * dt

        #    l.upd(lf)
        L.upd(Lf)
        a.upd(af)
        A.upd(Af)
        b.upd(bf)
        B.upd(Bf)

        if it % checkt == 0:
            eng1 = energy(l=l.d, L=L.d, dx=l.dx)
            sdif.push(eng1)
            if engz == 0:
                engz = eng1
            elif forcage != None:
                if sdif.full and sdif.dif() < -10 ** -6:
                    # if forcage_sign!=1:
                    #    forcage_sign = 1
                    forcage_signi += 1.
                    if forcage_signi > 1.:
                        forcage += forcage_coef
                    if forcage_signi > 3.:
                        forcage_signi = 0
                        if forcage_sign != 1.:
                            forcage_sign = 1.
                            if forcage_coef > 10 ** -8: forcage_coef /= 2.

                if sdif.full and sdif.dif() > 10 ** -6:
                    # if forcage_sign!=-1:
                    #    forcage_sign = -1
                    forcage_signi -= 1.
                    if forcage_signi < -1.:
                        forcage -= forcage_coef
                    if forcage_signi < -3.:
                        forcage_signi = 0.
                        if forcage_sign != -1.:
                            forcage_sign = -1.
                            if forcage_coef > 10 ** -8: forcage_coef /= 2.

            if (eng != 0 and (eng1 - eng) / eng1 > 2) or eng1 == np.nan or np.isnan(eng):
                # res.append((it*dt,l.d.copy(),L.d.copy()))
                out += ("\nconverge (energy goes form {:.4f} to {:.4f})".format(eng / engz, eng1 / engz))
                _print(out,debug_level=4)
                break;
            engold = eng
            eng = eng1

            q = eng / engz

            # print(toplot2[0][0][int(it/checkt)], toplot2[0][1][int(it/checkt)])

            time_pass = (datetime.datetime.now() - start).total_seconds()
            time_rest = 0 if it-starti <= 0 else time_pass / (it-starti) * (nt - it)
            sign = "+" if forcage_sign > 0 else "-"

            try:
                forcage_true = 0 if forcage==None else forcage
                forcage_coef_true = 0 if forcage==None else forcage

                out = "step {:.2f}M = simTime {:.2f}; forcage = {:.5f}m{}{:.5f}mm; \nwith energy form {:.4f} to {:.4f} in {:.0f} steps; energyChange = {:.5f}m (<{:.2f}m>); \ntime passed = {:.1f}min; time rest = {:.1f}min ({:.1f}h)".format(
                        it / (10 ** 6), it * dt, forcage_true * 10 ** 3, sign, forcage_coef_true * 10 ** 6, engold / engz, q,
                        checkt, (eng1 - engold) / eng1 * 10 ** 6, sdif.dif() * 10 ** 6, time_pass/60, time_rest/60, time_rest/60/60)

            except Exception as e:
                out = ("text:{}".format(e))
            _print(out)
            #_print('@', debug_level=1,save=False, saveto=False)

        if it % savet == 0:
            if engz == 0: engz = eng1
            _print('---save--- {}/{}/{}_{}.npy'.format(datadir, name, name, int(it / savet)),save=False,debug_level=2)
            # res.append((it*dt,l.d.copy(),L.d.copy(),eng))
            res0 = (it * dt, l.d.copy(), L.d.copy(), eng, forcage, forcage_coef, l.dimx, sdif.dif(), a.d.copy(), A.d.copy(),
            b.d.copy(), B.d.copy(), g.d.copy(), G.d.copy())
            x = np.asarray(list(res0), dtype=object)
            np.save('{}/{}/{}_{}.npy'.format(datadir, name, name, int(it / savet)), x)
            with open('{}/runstatus.txt'.format(path)) as f:
                read_data = f.read()
                if int(read_data)==0:
                    dif = (datetime.datetime.now() - start)
                    out += ("\nEnd. total time {} ({:.3f}s)".format(dif.__str__(), dif.total_seconds()))
                    _print(out, debug_level=5)
                    sys.exit(0)
            # time_pass = (datetime.datetime.now() - start).total_seconds()
            # time_rest = 0 if it==0 else time_pass/it*(nt-it)

            # txt.set_text("step {:.2f}mill = simTime {:.2f}; forcage = {:.5f}+-{:.5f}; energyChange = {:.5f} \nwith energy form {:.4f} to {:.4f} in {:.0f} steps; \ntime passed = {:.1f}min; time rest = {:.1f}min".format(
            #    it/(10**6),it*dt,forcage,forcage_coef,(eng1-engold)/eng1,engold/engz,eng/engz,savet,time_pass/60,time_rest/60))

            # print("step {:.2f}mill = simTime {:.3f}; with energy form {:.4f} to {:.4f} in {:.0f} steps; time passed = {:.1f}min; time rest = {:.1f}min".format(
            #    it/(10**6),it*dt,engold/engz,eng/engz,savet,time_pass/60,time_rest/60))
            # hatmodule1 = hatmodule(res[-1][1],res[-1][2],l.dx)


        # if not it % int(nt/100):
        #    # Display Solution
        #    # --------------------------------------
        #    res.append((it*dt,sp))
        #    print("{} = {}".format(it,it*dt))
    dif = (datetime.datetime.now() - start)
    out += ("\nEnd. total time {} ({:.3f}s)".format(dif.__str__(), dif.total_seconds()))
    _print(out, debug_level=5)
