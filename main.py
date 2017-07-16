# -*- coding: utf-8 -*-

import TransmissionClass, SignalClass
from plotting import Oscilloscope, plotForTex
import numpy as np

def timing(func, *args):
    import time
    start = time.time()
    ret = func(*args)
    end = time.time()
    print '\n Function ', func.__name__, ' needed %s s'%(end-start)
    return ret, end-start
    
def estimateTime(tshift, t):
    treg = np.roll(tshift, 1)
    treg[0] = t
    return treg
    
def sweep_deemphasis_settings(signalType, amp, amp1, dly, windowing , filename):
    from timeit import default_timer as timer
    from plotting import printProgress
    EyeOpening = np.zeros((len(dly),len(amp),len(amp1)))
    MeanOpening = np.zeros((len(dly),len(amp),len(amp1)))
    band = np.zeros((len(dly),len(amp),len(amp1)))
    
    tot = len(amp)*len(amp1)*len(dly)
    dt = 0.0   
    if tot > 32*32*4:
        tshift = np.ones(20)
        tshift[:] = 1e2    
    else:
        tshift = np.ones(5)
        tshift[:] = 1e2
    rel = True
    #Signal._params['Amplitude'] = 1.
    #Signal.run(saveToFile=False, filename='SignalInput', rdm='lfsr7')            
    #wv = Signal.waveform
                
    for id, d in enumerate(dly):
        for ia, a in enumerate(amp):
            for ia1, a1 in enumerate(amp1):
                st = timer()
                num = id*len(amp)*len(amp1)+len(amp1)*ia+ia1+1    
                tleft = dt *(tot-num)
                m, s = divmod(tleft, 60)
                h, m = divmod(m, 60)
                printProgress(num, tot, prefix = 'Time left %dh %02dmin %02ds'%(h,m,s), suffix = 'Complete', decimals=3, barLength = 50)             
                Signal._params['Amplitude'] = a
                Signal.run(False,'init', signalType, False)             
                #Signal.waveform['Sample 0']['Single'] = wv['Sample 0']['Single']*a
                Signal.addDeemphasis(deemphasisAmplitude=a1, deemphasisDelay=d, rel=rel)
                Signal.getSpectrum(saveToFile=False,filename='init', fromFile=False, verbose=False, window=windowing)
                Signal.waveform = TransmissionLine*Signal
                veye, vmean, deltaV = Signal.eyeOpening()
                veyediff = np.min(veye[1]) - np.max(veye[0])
                vmeandiff = vmean[1]-vmean[0]
                MeanOpening[id,ia,ia1] = vmeandiff
                EyeOpening[id,ia,ia1] = veyediff
                band[id,ia,ia1] = 0.5*(np.max(veye[1]) - np.min(veye[1]) + np.max(veye[0] - np.min(veye[0])))
                ed = timer()
                tshift = estimateTime(tshift, ed-st)
                dt = np.average(tshift)
        np.savez(filename+'_del_%s.npz'%d, Eye=EyeOpening[id], Average=MeanOpening[id])   
    return EyeOpening, MeanOpening, band, deltaV
    
def plot_bias(eyeopening, meanopening, band, dly, filename, titel, vmaxEye=0., vmaxMean=0.):
    import matplotlib.pyplot as plt
    import os
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.ticker as ticker
    
    savenamepdf = os.path.join(os.path.abspath('./'), filename+'.pdf')
    print 'Saved plot ', savenamepdf
    pdf = PdfPages(savenamepdf)    
    fig1 = plt.figure(figsize=(16, 9), dpi=80)
    fig2 = plt.figure(figsize=(16, 9), dpi=80)
    fig3 = plt.figure(figsize=(16, 9), dpi=80)
    fig4 = plt.figure(figsize=(16, 9), dpi=80)
    
    relDiff = np.abs(meanopening-eyeopening)/(meanopening+eyeopening)
    if vmaxEye == 0.:
        vmaxEye = np.max(eyeopening)
    if vmaxMean == 0.:
        vmaxMean = np.max(mean)
    vmaxRD = np.max(relDiff)
    
    for id, d in enumerate(dly):  
        if len(dly)>1:  
            ax1 = fig1.add_subplot(np.ceil(np.sqrt(len(dly))),np.ceil(np.sqrt(len(dly))),id+1)
            ax2 = fig2.add_subplot(np.ceil(np.sqrt(len(dly))),np.ceil(np.sqrt(len(dly))),id+1)
            ax3 = fig3.add_subplot(np.ceil(np.sqrt(len(dly))),np.ceil(np.sqrt(len(dly))),id+1)
            ax4 = fig4.add_subplot(np.ceil(np.sqrt(len(dly))),np.ceil(np.sqrt(len(dly))),id+1)
        else:
            ax1 = fig1.add_subplot(111)
            ax2 = fig2.add_subplot(111)
            ax3 = fig3.add_subplot(111)
            ax4 = fig4.add_subplot(111)
            
        fig1.suptitle(titel+'\nEye opening EO')
        fig2.suptitle(titel+'\nAverage amplitude AA')
        fig3.suptitle(titel+'\nRelative difference (AA-EO)/(AA+EO)') 
        fig4.suptitle(titel+'\nBand thickness')
    
        im1 = ax1.imshow(eyeopening[id,:,:], interpolation='None', origin='lower', vmax=vmaxEye)
        ax1.contour(eyeopening[id,:,:], [0.05, 0.125], colors=['red', 'green'], origin='lower')
        ax1.set_title('Delay %3.2f'%d)
        ax1.set_ylabel('Bias')
        ax1.set_xlabel('Biasd')
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(16.))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(16.))
        ax1.set_aspect('auto')
        
        im2 = ax2.imshow(meanopening[id,:,:], interpolation='None', origin='lower', vmin=0.0,  vmax=vmaxMean)
        ax2.contour(meanopening[id,:,:], [0.05, 0.125, 0.2], colors=['red', 'green', 'white'], origin='lower')
        ax2.set_title('Delay %3.2f'%d)
        ax2.set_ylabel('Bias')
        ax2.set_xlabel('Biasd')
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(16.))
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(16.))
        ax2.set_aspect('auto')
        
        im3 = ax3.imshow(relDiff[id,:,:], interpolation='None', origin='lower', vmax=vmaxRD)
        ax3.set_title('Delay %3.2f'%d)
        ax3.set_ylabel('Bias')
        ax3.set_xlabel('Biasd')
        ax3.xaxis.set_major_locator(ticker.MultipleLocator(16.))
        ax3.yaxis.set_major_locator(ticker.MultipleLocator(16.))
        ax3.set_aspect('auto')
        
        im4 = ax4.imshow(band[id,:,:], interpolation='None', origin='lower')
        ax4.set_title('Delay %3.2f'%d)
        ax4.set_ylabel('Bias')
        ax4.set_xlabel('Biasd')
        ax4.xaxis.set_major_locator(ticker.MultipleLocator(16.))
        ax4.yaxis.set_major_locator(ticker.MultipleLocator(16.))
        ax4.set_aspect('auto')
      
        fig1.colorbar(im1)
        fig2.colorbar(im2)
        fig3.colorbar(im3)
        fig4.colorbar(im4)
    
    pdf.savefig(fig1, dpi=80)
    pdf.savefig(fig2, dpi=80)
    pdf.savefig(fig3, dpi=80)
    pdf.savefig(fig4, dpi=80)
    pdf.savefig(Osci._fig, dpi=80)  
    pdf.close()
    
    return eyeopening, meanopening, band

def get_bias_dhpt12b():
    off = 1.37
    ibias = []
    ibias.append(1.37)
    ibias.append(3.08)
    ibias.append(3.82)
    ibias.append(4.40)
    
    ibias.append(4.89)
    ibias.append(5.32)
    ibias.append(5.74)
    ibias.append(6.11)
    
    ibias.append(6.45)
    ibias.append(6.78)
    ibias.append(7.09)
    ibias.append(8.43)
    
    ibias.append(9.53)
    ibias.append(11.25)
    ibias.append(12.54)
    ibias.append(13.54)
    
    ibias.append(14.32)
    ibias.append(14.94)
    ibias.append(15.39)
    ibias.append(15.99)
    
    ibias.append(16.65)
    ibias.append(16.91)
    ibias.append(17.07)
    ibiasx = [0,1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,60,70,80,100,150,200,255]
    from numpy import interp
    x = np.linspace(0,255,256, endpoint=True)
    
    return interp(x, ibiasx, ibias) - off

def get_biasd_dhpt12b():
    off = 1.37
    ibias = []
    ibias.append(1.37)
    ibias.append(2.01)
    ibias.append(2.19)
    ibias.append(2.34)
    
    ibias.append(2.47)
    ibias.append(2.58)
    ibias.append(2.68)
    ibias.append(2.78)
    
    ibias.append(2.88)
    ibias.append(2.97)
    ibias.append(3.06)
    ibias.append(3.43)
    
    ibias.append(3.75)
    ibias.append(4.31)
    ibias.append(4.76)
    ibias.append(5.17)
    
    ibias.append(5.54)
    ibias.append(5.89)
    ibias.append(6.19)
    ibias.append(6.75)
    
    ibias.append(7.91)
    ibias.append(8.79)
    ibias.append(9.56)
    ibiasx = [0,1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,60,70,80,100,150,200,255]
    from numpy import interp
    x = np.linspace(0,255,256, endpoint=True)
    
    return interp(x, ibiasx, ibias) - off

def linCorr(arrIn):
    offset = np.zeros(arrIn.shape[0])
    for i, v in enumerate(arrIn):
        offset[i] = -0.05*v
    
    return offset

'''
Parameter definitions
'''
SignalParameters = {}
SignalParameters['Number of bits'] = int(np.power(2,7))*10
SignalParameters['Number of samples'] = 1
SignalParameters['Frequency'] = 76.23e6
SignalParameters['Amplitude'] = 0.4
SignalParameters['Rise time'] = 0.04 # Percentage of period
SignalParameters['Fall time'] = 0.04 # Percentage of period
SignalParameters['Number of sampling points per period'] = 4096/4 #int(20e9/SignalParameters['Frequency'])
SignalParameters['Is signal differential'] = False

DeemphasisParameters = {}
DeemphasisParameters['Amplitude'] = 0.05
DeemphasisParameters['Delay'] = 0.4

TransmissionParameters = {}
TransmissionParameters['Filename21'] = '../sparam/kaptonPP/DUT_1_S21'
TransmissionParameters['Initial guess for fit model'] = np.array([1.1, 4.e-9, 1.e-19, 1.e-28, 1.e-40, 1.e-50])
TransmissionParameters['Max frequency to fit'] = 2.15e9
TransmissionParameters['Max frequency'] = 20.e9

Osciparams = {}
Osciparams['Index of bit in center'] = int(SignalParameters['Number of bits']/2)
Osciparams['Title'] = 'kapton and PP/Infiniband with fitted Transferfunction'
Osciparams['Vertical range'] = '.7'
Osciparams['Sampling rate'] = 0.01

if __name__ == "__main__":
    '''
    Object initialisation
    '''
    
    from matplotlib import interactive
    interactive(True)
    
    Signal = SignalClass.SignalClass(SignalParameters)
    TransmissionLine = TransmissionClass.TransmissionClass(TransmissionParameters)
    #TransmissionLine.enableTransferfunctionModel()
    Osci = Oscilloscope(Osciparams)
    plotForTex()
    
    b = get_bias_dhpt12b()
    bd = get_biasd_dhpt12b()
    
    curr2volt = 100*2/4/1000.
    '''
    get_bias() returns interpolated data taken from DHPT1.2a
    curr2volt is the correction factor for getting the output voltage at the 100Ohm termination in volts
    ''' 
    sweepStepB  = 1
    sweepStepBD = 1
    amplitudes = b[1:256:sweepStepB]*curr2volt
    amplitudes1 = 0.8*bd[0:256:sweepStepBD]*curr2volt #+linCorr(b[0:256:sweepStepBD]))
    delays = np.linspace(0.0, 1., 8, endpoint=True)
    
    filename = 'D:/workspace/sparam/EyeSim/eyes/test_no_window2' #, '../eyes/kaptonPPBB1mInfini_fittedT'] 
    
    signal = 'clk'
    windowing = 'Hamming'
    #eye, mean, band, dv = sweep_deemphasis_settings(signal, amplitudes, amplitudes1, delays, windowing, filename)
    '''
    eye = []
    mean = []
    for s in delays:
        d = np.load(filename+'_del_%s.npz'%s)
        eye.append(d['Eye'])
        mean.append(d['Average'])
    eye = np.array(eye)
    mean = np.array(mean)
    '''
    
    #optDly, optBias, optBiasd  = np.unravel_index(band.argmin(), band.shape)
    optDly, optBias, optBiasd = (0, 50, 0) #np.unravel_index(eye.argmax(), eye.shape)
    
    
    #emax = np.max(eye)
    #amax = np.max(mean)
    titel = 'kapton and PP/Infiniband with fitted Transferfunction'#, 'kapton and PP/Infiniband and BB/Infiniband with fitted Transferfunction']
     
    SignalParameters['Amplitude'] = amplitudes[optBias]
    Signal.run(False,'init', signal, False)
    Osci.trigger_single(signal= Signal, rdm=False)
    Signal.getSpectrum(saveToFile=False,filename='init', fromFile=False, verbose=False, window=windowing)
    Osci.plotSpectrum(Signal, 'SignalIn', 'lin')
    
    Signal.addDeemphasis(amplitudes1[optBiasd], delays[optDly], False)
    Osci.trigger_single(signal= Signal, rdm=False)
    Signal.getSpectrum(saveToFile=True,filename='init', fromFile=False, verbose=False, window=windowing)
    Osci.plotSpectrum(Signal, 'SignalIn+Preemph.', 'lin')
    Signal.waveform = TransmissionLine.transfer(Signal, False)
    
    Signal.getSpectrum(saveToFile=False,filename='init', fromFile=False, verbose=False, window=windowing)
    #Osci.plotEye(Signal, 'optimal eye (dly%s, b%s, bd%s) ~ (%d +- %d)mV'%(optDly, optBias, optBiasd, eye[optDly, optBias, optBiasd]*1e3, dv*1e3),  True, False)
    #Osci.plotEye(Signal, 'optimal eye (dly%s, b%s, bd%s)'%(optDly, optBias, optBiasd),  True, False)
    Osci.trigger_single(signal= Signal, rdm=False)
    Osci.plotSpectrum(Signal, 'SignalOut', 'lin')
     
    Osci.plotTransferfunction(TransmissionLine, 'lin')
    Osci.plotTransferfunctionModel(TransmissionLine, 'lin')
    
    #plot_bias(eye, mean, band, delays, filename=filename, titel=titel, vmaxEye=emax, vmaxMean=amax)    
    
    # Normalization Parseval
    #print np.mean(Signal.waveform['Sample 0']['Single']**2)
    #sp = Signal.spectrum['Data'][0]
    #print (sp[0]*sp[0].conj() + 2*np.sum(sp[1:]*sp[1:].conj())).real / len(Signal.waveform['Sample 0']['Single'])