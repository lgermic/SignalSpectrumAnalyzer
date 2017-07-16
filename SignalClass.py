# -*- coding: utf-8 -*-

import numpy as np
from plotting import printProgress
import os
from lfsr.db import max_len_lfsr_min_taps

class SignalClass():
    def __init__(self, params):
        '''
        @param params: Dictionary containing ... 
        '''
        self._params = params
        self._chunksize = 1000
        self._window = ''
        
    def saveSpectrum(self, filename='init', idx=1):
        np.savez('./rawdata/'+filename+'spectrum'+'_%s.npz'%idx, Spectrum=self.spectrum)        
        
    def saveWaveform(self, filename='init', idx=1):
        Time    = self.waveform['Time']
        if self._params['Is signal differential']:
            Sample = np.zeros((self._chunksize,2,self._params['Number of sampling points per period']*self._params['Number of bits']))
            for i in range(self._chunksize):
                Sample[i:0:] = self.waveform['Sample %s'%i]['Positive']
                Sample[i:1:] = self.waveform['Sample %s'%i]['Negative']
        else:
            Sample = np.zeros(len(self.waveform['Sample 0']['Single']))
            for i in range(self._chunksize):
                Sample[i:] = self.waveform['Sample %s'%i]['Single']
        np.savez('./rawdata/'+filename+'waveform_%s.npz'%idx, Time=Time, Samples=Sample, Differential=self._params['Is signal differential'])        
        
    def loadWaveform(self, filename, all=True, id=1):
        waveform = {}
        file = './rawdata/'+filename+'waveform_0.npz'
        
        if all:
            filecnt = 0
            waveform['Time'] = np.load(file)['Time']
            while os.path.isfile(file):
                d = np.load(file)        
                if d['Differential']:
                    for idx, v in enumerate(d['Samples']):
                        waveform['Sample %s'%(idx+self._chunksize*filecnt)]= {'Positive':v[0],'Negative':v[1]}
                else:
                    for idx, v in enumerate(d['Samples']):
                        waveform['Sample %s'%(idx+self._chunksize*filecnt)] = {'Single' : v}
                tmp = np.load(file)['FrequencyDivider']
                filecnt += 1
                file = './rawdata/'+filename+'waveform_%s.npz'%filecnt
        else:
            filename = './rawdata/'+filename+'waveform_%s.npz'%id
            if os.path.isfile(file):
                d = np.load(file)        
                if d['Differential']:
                    for idx, v in enumerate(d['Samples']):
                        waveform['Sample %s'%idx]= {'Positive':v[0],'Negative':v[1]}
                else:
                    for idx, v in enumerate(d['Samples']):
                        waveform['Sample %s'%idx] = {'Single' : v}
                filecnt += 1
            waveform['Time'] = np.load(file)['Time']
        self.waveform = waveform
    
    def loadSpectrum(self, filename, all=True, idx=1):
        file = './rawdata/'+filename+'spectrum'+'_1.npz'
        filecnt = 0
        self.spectrum = []
        if all:
            while os.path.isfile(file):
                if os.path.isfile(file):    
                    spectrum = np.load(file=file)
                tmp = spectrum['Spectrum']
                self.spectrum += (list(tmp)) 
                filecnt += 1
                file = './rawdata/'+filename+'spectrum'+'_%s.npz'%filecnt
        else:
            file = './rawdata/'+filename+'spectrum'+'_%s.npz'%idx
            if os.path.isfile(filename):    
                spectrum = np.load(file=file)
                self.spectrum = spectrum['Spectrum']
        self.spectrum = np.array(self.spectrum)
            
    def run(self, saveToFile=True, filename='init', rdm='random', verbose=False):
        if self._params['Number of samples']<self._chunksize:
            self._chunksize = self._params['Number of samples']
        self.waveform = {}
        bitList = []
        for filecnt in range(int(self._params['Number of samples']/self._chunksize)):
            if rdm == 'random':
                bitList = self.getRandomBits(samples=self._chunksize)    
            elif rdm == 'lfsr7':
                bitList = self.getLFSR(samples=self._chunksize, N=7)
            elif rdm == 'lfsr32':
                bitList = self.getLFSR(samples=self._chunksize, N=32)
            else:    
                bitList = self.getClock(samples=self._chunksize)
        
            for i in range(self._chunksize):
                self._bits = bitList[i]
                if verbose:
                    printProgress(i+self._chunksize*filecnt, self._params['Number of samples']-1, prefix = 'Generating waveforms:', suffix = 'Complete', decimals=3, barLength = 50)
                self.waveform['Sample %s'%i] = {}
                if self._params['Is signal differential']:
                    a, b = self.generateWaveform()
                    self.waveform['Time'] = a[0]
                    self.waveform['Sample %s'%i]['Positive'] = a[1] 
                    self.waveform['Sample %s'%i]['Negative'] = b[1]
                else:
                    if rdm == 'sin':
                        a = self.generateSin()
                    else:
                        a = self.generateWaveform()
                    #a = self.cml_driver(a)
                    self.waveform['Sample %s'%i]['Single'] = a[1]
                    self.waveform['Time'] = a[0] 
            
            if saveToFile:
                self.saveWaveform(filename=filename, idx=filecnt)    
            if filecnt < int(self._params['Number of samples']/self._chunksize)-1:
                self.waveform = {}
            
        self._hamm = self.hammingWindow()
        self._tuk = self.tukeyWindow()
                
    def getSpectrum(self, saveToFile=True,filename='init', fromFile=False, verbose=False, window='Hamming'):
        from numpy.fft import rfft
        #baseFreq = self.roundFrequency(self._params['Frequency'])
        baseFreq = self._params['Frequency']
        self.spectrum = {}
        N = len(self.waveform['Sample 0']['Single'])
        nqf = float(baseFreq)*self._params['Number of sampling points per period']/2.
        t = []
        for k in range(N/2+1):
            t.append(2*nqf/(N-1)*k)
        self.spectrum['Frequency'] = np.array(t)
        spectrum = []
        self._window = window
        if fromFile:
            filecnt = 1
            file = './rawdata/'+filename+'waveform'+'_%s.npz'%filecnt
            while os.path.isfile(filename):
                file = './rawdata/'+filename+'waveform'+'_%s.npz'%filecnt
                waveform = self.loadWaveform(file, False, filecnt)
                for i in range(self._chunksize):
                    if verbose:
                        printProgress(i+self._chunksize*filecnt, self._params['Number of samples']-1, prefix = 'Receive spectrum:', suffix = 'Complete', decimals=3, barLength = 50)
                    spect = rfft(waveform['Sample %s'%i]['Single'])
                    spectrum.append(spect)        
                filecnt += 1
            self.spectrum['Data'] = np.array(spectrum)                
        else:
            for filecnt in range(int(self._params['Number of samples']/self._chunksize)):
                for i in range(self._chunksize):
                    if window == 'Tukey':
                        spect = rfft(self._tuk*self.waveform['Sample %s'%i]['Single'], norm=None)
                    elif window == 'Hamming':
                        spect = rfft(self._hamm*self.waveform['Sample %s'%i]['Single'], norm=None)
                    else:
                        spect = rfft(self.waveform['Sample %s'%i]['Single'], norm=None)
                    if verbose:
                        printProgress(i+self._chunksize*filecnt, self._params['Number of samples']-1, prefix = 'Receive spectrum:', suffix = 'Complete', decimals=3, barLength = 50)
                    spect /= np.sqrt(2*len(spect))
                    spectrum.append(spect)        
                self.spectrum['Data'] = np.array(spectrum)
                if saveToFile:                
                    self.saveSpectrum(filename=filename, idx=filecnt)
                if filecnt < int(self._params['Number of samples']/self._chunksize)-1:
                    self.spectrum['Data'] = np.empty()
                    
    def db(self, x):
        x = np.array(x)
        return 20.*np.log10(np.abs(x))
        
    def roundFrequency(self, f):
        '''
        @return: closest frequency sampled 
        '''
        # TODO get the stepSize
        div = 4e7
        if f < div:
            return div
        else:
            factor = int(f/div)
            res = f%div
            if res >= div/2:
                return int((factor+1)*div)
            else:
                return int(factor*div)
    
    def getLFSR(self, samples, N=7):    
        import lfsr
        n=N
        taps = max_len_lfsr_min_taps[n]
        poly = lfsr.taps_to_poly(taps)
        prng = lfsr.lfsr_if(poly)
        
        bitList = []
        sampleString = ''
        for i in range(self._params['Number of bits']):
            sampleString += '%s'%('{0:07b}'.format(next(prng))[6])
        sampleString = '0'*20 + sampleString + '0'*20   
        for s in range(samples):
            bitList.append(sampleString)
        return bitList
    
    def getRandomBits(self, samples):    
        number4Bytes = int(self._params['Number of bits']-1)/32
        
        bitList = []
        for i in range(samples):
            sample = [''.join(format(ord(x), 'b') for x in os.urandom(32)) for j in range(number4Bytes+1)]
            sampleString = ''.join(s for s in sample)
            sampleString = sampleString[0:self._params['Number of bits']]
            sampleString = '0'*20 + sampleString + '0'*20   
            bitList.append(sampleString)
            
        return bitList
    
    def getClock(self, samples):    
        bitList = []
        if self._params['Number of bits']%2==0:
            seq = '10'*(self._params['Number of bits']/2)
        else:
            seq = '10'*(self._params['Number of bits']/2)+'0' 
        seq = '0'*20 + seq + '0'*20 
        for i in range(samples):
            bitList.append(seq)
        return bitList
    
    def generateSin(self):
        baseFreq = self._params['Frequency']
        T = 1./baseFreq
        numberOfSamplingPoints = self._params['Number of sampling points per period']
        ampl = self._params['Amplitude']
        pattern = np.zeros((2,self._params['Number of bits']*numberOfSamplingPoints))
        pattern[0] = np.linspace(0, T/2*self._params['Number of bits'], self._params['Number of bits']*numberOfSamplingPoints, endpoint=True)
        pattern[1] = np.array([ampl*np.sin(2*np.pi*baseFreq*t) for t in pattern[0]])
        return pattern
    
    def generateWaveform(self):
        #baseFreq = self.roundFrequency(self._params['Frequency'])
        baseFreq = self._params['Frequency']
        T = 1./baseFreq
        tr = T*self._params['Rise time']
        tf = T*self._params['Fall time']
        ampl = self._params['Amplitude']
        numberOfSamplingPoints = self._params['Number of sampling points per period']
        differential = self._params['Is signal differential']
        
        pattern = np.zeros((2,len(self._bits)*numberOfSamplingPoints))
        patternBar = np.zeros((2,len(self._bits)*numberOfSamplingPoints))
        
        bitUp = np.ones(numberOfSamplingPoints)*ampl
        bitDown = np.zeros(numberOfSamplingPoints)
        bitHigh = np.ones(numberOfSamplingPoints)*ampl    
        bitLow = np.zeros(numberOfSamplingPoints)    
        
        bbitUp = np.ones(numberOfSamplingPoints)*ampl
        bbitDown = np.zeros(numberOfSamplingPoints)
        
        
        t = 0.
        tcnt = 0
        #dtEdge = T/(1.8*numberOfSamplingPoints)
        dt = T/numberOfSamplingPoints
        tm = [tr,tf]
        
    
        bitt = np.arange(0,T,dt)    
        if bitt.shape[0] > numberOfSamplingPoints:
            bitt = bitt[:-1] 
        
        while t < tm[np.argmax(tm)]: 
            if t<tr:
                bitUp[tcnt]   = ampl*(1-np.exp(-t/(.1*tr)))
                #bitUp[tcnt]   = ampl/tr*t
                bbitDown[tcnt] = ampl*np.exp(-t/(.1*tr))
                #bbitDown[tcnt] = ampl-ampl/tr*t
            if t<tf:
                bitDown[tcnt]   = ampl*np.exp(-t/(.1*tf))
                #bitDown[tcnt] = ampl-ampl/tf*t
                bbitUp[tcnt] = ampl*(1-np.exp(-t/(.1*tf)))
                #bbitUp[tcnt]   = ampl/tf*t
            t += dt
            tcnt += 1
            
        lastVal = int(self._bits[0])    
        for bitidx, bitval in enumerate(self._bits):
            pattern[0,bitidx*numberOfSamplingPoints:(bitidx+1)*numberOfSamplingPoints] = bitt+bitidx*T
            patternBar[0,bitidx*numberOfSamplingPoints:(bitidx+1)*numberOfSamplingPoints] = bitt+bitidx*T
            
            if int(bitval)-lastVal == 1:
                pattern[1,bitidx*numberOfSamplingPoints:(bitidx+1)*numberOfSamplingPoints] = bitUp
                patternBar[1,bitidx*numberOfSamplingPoints:(bitidx+1)*numberOfSamplingPoints] = bbitDown
            elif int(bitval)-lastVal == -1:
                pattern[1,bitidx*numberOfSamplingPoints:(bitidx+1)*numberOfSamplingPoints] = bitDown
                patternBar[1,bitidx*numberOfSamplingPoints:(bitidx+1)*numberOfSamplingPoints] = bbitUp
            elif int(bitval)==1: 
                pattern[1,bitidx*numberOfSamplingPoints:(bitidx+1)*numberOfSamplingPoints] = bitHigh
                patternBar[1,bitidx*numberOfSamplingPoints:(bitidx+1)*numberOfSamplingPoints] = bitLow
            elif int(bitval)==0:
                pattern[1,bitidx*numberOfSamplingPoints:(bitidx+1)*numberOfSamplingPoints] = bitLow
                patternBar[1,bitidx*numberOfSamplingPoints:(bitidx+1)*numberOfSamplingPoints] = bitHigh
        
            lastVal = int(bitval)
            
        if differential: 
            pattern[1] = pattern[1]-self._params['Amplitude']/2
            patternBar[1] = patternBar[1]-self._params['Amplitude']/2  
            return pattern, patternBar
        
        else:
            pattern[1] = pattern[1]-self._params['Amplitude']/2
           
        return pattern


    def addDeemphasis(self, deemphasisAmplitude, deemphasisDelay, rel=True): 
    # TODO roundFrequency!!!
        dly = int(deemphasisDelay*self._params['Number of sampling points per period'])
        
        if 'Single' in self.waveform['Sample 0'].keys():
            for i in range(self._params['Number of samples']):
                waveform_cpy = self.waveform['Sample %s'%i]['Single']
                
                waveform_dly = np.append(np.array([waveform_cpy[0]]*dly),np.roll(waveform_cpy,dly)[dly:])
                if rel:
                    waveform_inv = -1*deemphasisAmplitude*waveform_dly
                else:
                    waveform_inv = -1*deemphasisAmplitude*(waveform_dly/np.max(waveform_dly))
                
                waveform_cpy = waveform_cpy + waveform_inv                         
                self.waveform['Sample %s'%i]['Single'] = waveform_cpy
            
        else:
            for i in range(self._params['Number of samples']):
                waveform_cpy = self.waveform['Sample %s'%i]['Positive']
                
                waveform_dly = np.append(np.array([waveform_cpy[0]]*dly),np.roll(waveform_cpy,dly)[dly:])
                
                waveform_inv = -1*deemphasisAmplitude*waveform_dly
                
                waveform_cpy = waveform_cpy+waveform_inv                         
                self.waveform['Sample %s'%i]['Positive'] = waveform_cpy
                
                ##########################
                
                waveformBar_cpy = self.waveform['Sample %s'%i]['Negative']
                
                waveformBar_dly = np.append(np.array([waveformBar_cpy[0]]*dly),np.roll(waveformBar_cpy,dly)[dly:])
                
                waveformBar_inv = -1*deemphasisAmplitude*waveformBar_dly
                
                waveformBar_cpy = waveformBar_cpy+waveformBar_inv                         
                   
                self.waveform['Sample %s'%i]['Negative'] = -waveformBar_cpy
       
        
    def __mul__(self, deemphasis):
        self.addDeemphasis(deemphasis['Amplitude'], deemphasis['Delay'])
        return self
    
    def shaper(self):
        from numpy.fft import irfft
        cnt = 0
        bode = np.ones(self.spectrum['Frequency'].shape[0])
        for k in range(bode.shape[0]):
            bode[k] = 1./(1.+k*1e-14*self._params['Frequency'])
                    
        for keysample, valsample in self.waveform.iteritems():
            if 'Time' not in keysample:
                printProgress(cnt, self._params['Number of samples']-1, prefix = 'Shaping waveform:', suffix = 'Complete', decimals=3, barLength = 50)            
                for key, val in valsample.iteritems():    
                    spec = self.spectrum['Data'][cnt]
                    inv = irfft(spec*bode)
                    self.waveform[keysample][key] = inv 
                cnt += 1
    
    
    def eyeOpening(self, verbose=False):
        from plotting import simpleHist
                      
        h = []
        cnt = 0
        if 'Single' in self.waveform['Sample 0'].keys():
            for keysamples, valsamples in self.waveform.iteritems():
                if verbose:
                    printProgress(cnt, len(self.waveform)-1, prefix = 'Recording eye diagram:', suffix = 'Complete', decimals=3, barLength = 50)
                if 'Time' not in keysamples:
                    for b in range(22,len(self._bits)-22):
                        st = int((b+1./2-0.05)*self._params['Number of sampling points per period'])
                        ed = int((b+1./2+0.05)*self._params['Number of sampling points per period'])
                        h.append(valsamples['Single'][st:ed])
                cnt += 1
        else:
            pass
            #for keysamples, valsamples in self.waveform.iteritems():
            #    printProgress(cnt, len(self.waveform)-1, prefix = 'Recording eye:', suffix = 'Complete', decimals=3, barLength = 50)
            #    if 'Time' not in keysamples:
            #        rx = self.differentialProbing(valsamples['Positive'], valsamples['Negative'])
            #        h.append(rx[startPointHist:endPointHist])
            #        
            #    cnt += 1    
        
        h = np.array(h)
        
        histCnt, x, dv = simpleHist(h, 128)    
        
        #gauss = x[np.where(histCnt!=0)[0]]
        gauss = x[histCnt!=0]
        
        gauss1 = gauss[gauss < 0.]
        
        #gauss1peak = x[np.argmax(histCnt[x<0.])]
        
        gauss1peak = 0.
        N = 0.
        for id in np.where(x < 0.)[0]:
            gauss1peak  += histCnt[id] * x[id]
            N += histCnt[id]
        if N != 0:
            gauss1peak /= N 
        else:
            gauss1peak = 0.
            
        gauss2 = gauss[gauss>0.]
        #gauss2peak = x[len(x[x<0.]) + np.argmax(histCnt[x>0.])]
        gauss2peak = 0.
        N = 0.
        for id in np.where(x > 0.)[0]:
            gauss2peak  += histCnt[id] * x[id]
            N += histCnt[id]
        if N != 0:
            gauss2peak /= N 
        else:
            gauss2peak = 0.
            
        return [gauss1, gauss2], [gauss1peak, gauss2peak], dv
            
    def cml_driver(self, vin):
        vt=0.4 #+ 0.01*np.random.normal()
        mosparam = .01
        vgs = vin
        ids = np.zeros(vgs.shape[1])
        ids[vgs[1]>vt] = mosparam*np.power(vgs[1]-vt,2)[vgs[1]>vt]
        ids /= np.max(ids)
        vgs[1] = ids*self._params['Amplitude']
        vgs[1] -= np.max(vgs[1])/2.
        return vgs
    
    def iv(self, val):
        vt=0.4 #+ 0.01*np.random.normal()
        mosparam = 1.0
        if val > vt:
            ids = mosparam*np.power(val-vt,2)
        else:
            ids = 0.
        #ids /= mosparam*(1.2-vt)**2
        return ids
    
    def hammingWindow(self): 
        return np.array([(0.54 - 0.46*np.cos(2*np.pi*idx/(len(self.waveform['Time'])-1))) for idx in range(len(self.waveform['Time']))])
        #return np.array([((0.5+1.e-10) - (0.5-1.e-10)*np.cos(2*np.pi*idx/(len(self.waveform['Time'])-1))) for idx in range(len(self.waveform['Time']))])
        
    def tukeyWindow(self): 
        wind = 5.*self.hammingWindow()
        wind[wind >= 1.] = 1.
        return wind