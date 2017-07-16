# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import configparser
import numpy as np
import sys
from plotting import printProgress
import SignalClass
        
class TransmissionClass():
    def __init__(self, config):
        self._data21 = self.readSParameterFile(config['Filename21'])
        self._attenuation = self.getSParameters(self._data21)
        self.gotBode = False
        self._fitMax = config['Max frequency to fit']
        self._guess = config['Initial guess for fit model']
        self._fMax = config['Max frequency']
        
        
    def readSParameterFile(self, filename):
        ''' 
        @param filename: This is the filename of the .wfm file containing the s21 attenuation vs frequency       
        @return: Dictionary with metaData 'Number of Sampling points', 'Step Size' and raw data 'Data' 
        '''
        import csv
        data = []
        phase = []
        self.dataDict  = {}
        with open(filename+'.wfm', 'rb') as f:
            reader = csv.reader(f)
            while 'Data:' not in reader.next():
                pass
            for row in reader:
                r = row[0].split()
                data.append(float(r[0]))
                phase.append(float(r[1]))
                
        # TODO - be more clever!!!
        with open(filename+'.wfm', 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 0:
                    pass
                else:
                    if 'Samples:' in row[0]:
                        self.dataDict['Number of sampling points'] = int(row[0].split()[1])
                    elif 'Step Size:' in row[0]:
                        self.dataDict['Step Size'] = row[0].split()[2]
        
        self.dataDict['Phase'] = np.array(phase)*2*np.pi/360.
        self.dataDict['Data'] = np.array(data)
        return self.dataDict
    
    
    def getSParameters(self, data):
        '''
        @param data: This is a dictionary from the return of readParameterFile 
        @return: Dictionary with assigned attenuation for frequencies
        '''
        bode = []
        if 'meg' in data['Step Size']:
            self._stepSize = float(data['Step Size'][0:-3])*1e6
        else:
            print 'Error: Step size is not in MHz... panic, What should I do??????'
            sys.exit(-1)  
        
        for sidx, sval in enumerate(data['Data']):
            bode.append(sval)
        return np.array(bode)
    
    def roundFrequency(self, f):
        '''
        @param f: Frequency to round by the Step size 
        @return: closest frequency sampled 
        '''
        div = self._stepSize
        if f < div:
            return div
        else:
            factor = int(f/div)
            res = f%div
            if res >= div/2:
                return int((factor+1)*div)
            else:
                return int(factor*div)
    
    def transfer(self, signal, verbose=False):
        from numpy.fft import  irfft
        if not self.gotBode:
            from scipy.interpolate import interp1d
            fmax = self._fMax
            if self._stepSize == 4e6:
                step = 10
            if self._stepSize == 2e6:
                step = 20
            else:
                step = 10
            tmlBode = np.power(10,self._attenuation/20.)[::step]
            phase = self.dataDict['Phase'][::step]
            #tmlBode = tmlBode[:tmlBode.shape[0]/4:2]
            oldx = np.linspace(0,tmlBode.shape[0], num = tmlBode.shape[0], endpoint=True)
            fbode = interp1d(oldx, tmlBode, kind='cubic')
            fphase = interp1d(oldx, phase, kind='slinear')
            
            if signal.spectrum['Frequency'][len(signal.spectrum['Frequency'])-1] < fmax:
                sp = signal.spectrum['Frequency']
            else:    
                sp = signal.spectrum['Frequency'][0:len(np.where(signal.spectrum['Frequency']<fmax)[0])]
            
            newx = np.linspace(0,tmlBode.shape[0], num = len(sp), endpoint = True)
            interpBode = fbode(newx)
            interpPhase = fphase(newx)
            
            #import matplotlib.pyplot as plt
            #plt.plot(oldx, phase)
            #plt.plot(newx, interpPhase)
            #plt.show()
            self._bode = interpBode#/interpBode[0] 
            self.gotBode = True
        else:
            interpBode = self._bode
        wf = {}                
        
        complexBode = 1j*np.sin(2*np.pi*interpPhase)+np.cos(2*np.pi*interpPhase)
        complexBode *= interpBode
        
        for cnt in range(len(signal.waveform)-1):
            if verbose:
                printProgress(cnt, len(signal.waveform)-1, prefix = 'Transmitting wavefrom:', suffix = 'Complete', decimals=3, barLength = 50)            
            
            wf['Sample %s'%cnt] = {}
            if 'Single' in signal.waveform['Sample %s'%cnt].keys():
                fftwf = signal.spectrum['Data'][cnt]
                wf['Sample %s'%cnt]['Single'] = irfft(fftwf*np.append(complexBode,np.array([0.]*(fftwf.shape[0]-interpBode.shape[0]))), norm=None)
                wf['Sample %s'%cnt]['Single'] *= np.sqrt(2*len(fftwf))
                if signal._window == 'Hamming':
                    wf['Sample %s'%cnt]['Single'] /= signal._hamm
                elif signal._window == 'Tukey':
                    wf['Sample %s'%cnt]['Single'] /= signal._tuk
            
        wf['Time'] = signal.waveform['Time']
        return wf
        
    def __mul__(self, signal):
        return self.transfer(signal)
    
    def db(self, x):
        return 20*np.log10(x)
    
    def spectrumFit(self, x, y, func, guess):
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(func, x, y, guess)
        return popt, pcov
               
    def Transferfunction(self, s, a, t1, t2, t3, t4, t5):
        nominator   = 0.8*a
        denominator = 1.+t1*s+t2*np.power(s,2)+t3*np.power(s,3)+t4*np.power(s,4)+t5*np.power(s,5)
        return nominator/denominator 
   
    def fitModel(self):
        f = np.arange(0,self.dataDict['Data'].shape[0])*self._stepSize
        y = np.power(10,self.dataDict['Data']/20.)
        popt, pcov = self.spectrumFit(f[f<self._fitMax], y[f<self._fitMax], self.Transferfunction, self._guess)
        self._transferFunctionModel = np.array([self.Transferfunction(i, a=popt[0], t1=popt[1], t2=popt[2], t3=popt[3], t4=popt[4], t5=popt[5]) for i in f])
        return self._transferFunctionModel
    
    def enableTransferfunctionModel(self):
        self._rawBode = self._attenuation
        self.fitModel()
        self._attenuation = 20*np.log10(self._transferFunctionModel)
        
    def disableTransferfunctionModel(self):
        if self._rawBode:
            self._attenuation = self._rawBode
        else:
            pass 
        