# -*- coding: utf-8 -*-

import os, numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
from matplotlib import gridspec
from matplotlib import interactive
interactive(True)
from termcolor import colored

def update_label(old_label, exponent_text):
    if exponent_text == "":
        return old_label
    
    try:
        units = old_label[old_label.index("[") + 1:old_label.rindex("]")]
    except ValueError:
        units = ""
    label = old_label.replace("[{}]".format(units), "")
    
    exponent_text = exponent_text.replace("\\times", "")
    
    return "{} [{} {}]".format(label, exponent_text, units)

def format_label_string_with_exponent(ax, axis='both'):  
    """ Format the label string with the exponent from the ScalarFormatter """
    ax.ticklabel_format(axis=axis, style='sci')
    
    axes_instances = []
    if axis in ['x', 'both']:
        axes_instances.append(ax.xaxis)
    if axis in ['y', 'both']:
        axes_instances.append(ax.yaxis)
    
    for ax in axes_instances:
        ax.major.formatter._useMathText = True
        plt.draw() # Update the text
        exponent_text = ax.get_offset_text().get_text()
        #print ax.get_offset_text().get_text()
        label = ax.get_label().get_text()
        ax.offsetText.set_visible(True)
        #print label, exponent_text
        ax.set_label_text(update_label(label, exponent_text))

def plotForTex(w=418.25555, ratio=0.8):
    rcParams['figure.titlesize'] = 'medium'
    rcParams['axes.labelsize'] = 10
    rcParams['axes.titlesize'] = 12
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 'small'

    #rcParams['font.family'] = 'serif'
    #rcParams['font.serif'] = ['Computer Modern Roman']
    #rcParams['text.usetex'] = True
    '''
    WIDTH = w #get from tex compiler with "\showthe\textwidth" in tex document / the number latex spits out
    FACTOR = ratio  # the fraction of the width you'd like the figure to occupy
    fig_width_pt  = WIDTH * FACTOR

    inches_per_pt = 1.0 / 72.27
    golden_ratio  = (np.sqrt(5) - 1.0) / 2.0  # because it looks good

    fig_width_in  = fig_width_pt * inches_per_pt  # figure width in inches
    fig_height_in = fig_width_in * golden_ratio   # figure height in inches
    rcParams['figure.figsize'] = fig_width_in, fig_height_in # fig dims as a list
    '''
    #return fig_dims

plotColors = ['b', 'g', 'r' ,'c', 'm', 'y', 'k', 'w']

class Oscilloscope():
    def __init__(self, params):
        rcParams['figure.titlesize'] = 'medium'
        rcParams['axes.labelsize'] = 10
        rcParams['axes.titlesize'] = 12
        rcParams['xtick.labelsize'] = 10
        rcParams['ytick.labelsize'] = 10
        rcParams['legend.fontsize'] = 'small'
        self._params = {}
        self._params['Index of bit in center'] = params['Index of bit in center']
        self._params['Vertical range'] = params['Vertical range']
        self._params['Sampling rate'] = params['Sampling rate']
        self._params['Title'] = params['Title']
   
        gs = gridspec.GridSpec(5, 3)
        self._fig = plt.figure(figsize=(16, 9), dpi=80)
        self._fig.suptitle(self._params['Title'])
        self._ax1 = self._fig.add_subplot(gs[:2,:])
        self._ax5 = self._fig.add_subplot(gs[3:5,:2])
        self._ax2 = self._fig.add_subplot(gs[3:5,2:], sharey = self._ax5)
        self._ax3 = self._fig.add_subplot(gs[2:3,:])
        self._ax4 = self._ax5.twinx() #self._fig.add_subplot(gs[4:,:], sharex = self._ax5)
        self._ax6 = self._ax3.twinx()
        self._ax4.get_yaxis().set_visible(False)
        self._ax2.get_yaxis().set_visible(False)
        
        self._fig.subplots_adjust(wspace=0.3, hspace=0.3)
        
        self._ax1.grid(lw=1, ls='-', alpha=0.1, c='black')
        self._ax2.grid(lw=1, ls='-', alpha=0.1, c='black')
        self._ax3.grid(lw=1, ls='-', alpha=0.1, c='black')
        self._ax5.grid(lw=1, ls='-', alpha=0.1, c='black')
        
        self._ax1plotCnt = 0           
        self._ax2plotCnt = 0           
        self._ax3plotCnt = 0           
        
        self._ax1ColorCnt = 0
        self._ax3ColorCnt = 0
        self._ax5ColorCnt = 0
        
    def clear(self):
        self._fig.clf()
        gs = gridspec.GridSpec(5, 4)
        self._ax1 = self._fig.add_subplot(gs[0:3,0:2])
        self._ax2 = self._fig.add_subplot(gs[0:3,2:], sharey = self._ax1)
        self._ax3 = self._fig.add_subplot(gs[3:,2:])
        self._ax4 = self._fig.add_subplot(gs[3:,0:2], sharex = self._ax1)
        
       

        self._ax1plotCnt = 0           
        self._ax2plotCnt = 0           
        self._ax3plotCnt = 0           
        
        self._ax1ColorCnt = 0
        self._ax3ColorCnt = 0
        self._ax5ColorCnt = 0
        
    def differentialProbing(self, waveform, waveformbar, amplification=1.):
        return amplification*(waveform-waveformbar) 

    def trigger(self, signal, rdm=True):
        #self.clear()
        #startPoint = int(samplesPerPeriod/2   +bitShown*samplesPerPeriod)
        #endPoint   = int(samplesPerPeriod*5/2 +bitShown*samplesPerPeriod)
        c = plotColors[self._ax1ColorCnt%len(plotColors)]  
        q = ''
        while 'q' not in q:
            if rdm:
                sample = np.random.randint(signal._params['Number of samples'])
            else:
                sample = int(signal._params['Number of samples']/2)
            if 'Single' in signal.waveform['Sample 0'].keys():
                self._ax1.plot(signal.waveform['Time'][::2], signal.waveform['Sample %s'%sample]['Single'][::2], ls='-', lw=1, marker='', alpha=1, color=c)
            else:
                self._ax1.plot(signal.waveform['Time'][::2], signal.waveform['Sample %s'%sample]['Positive'][::2], ls='-', lw=1, marker='', alpha=1, color=c)
                self._ax1.plot(signal.waveform['Time'][::2], signal.waveform['Sample %s'%sample]['Negative'][::2], ls='-', lw=1, marker='', alpha=1, color=c)
            
            q = raw_input('press to trigger')
        
        self._ax1.set_title('Waveform')
        self._ax1.set_ylabel("amplitude")
        self._ax1.set_xlabel("time [s]")
        
        if 'Auto' in self._params['Vertical range']: 
            ymin, ymax = self._ax1.get_ylim()
            self._ax1.set_ylim(ymin*1.1,ymax*1.1)
        else:    
            self._ax1.set_ylim(-(self._params['Vertical range']/2),self._params['Vertical range']/2)
    
        self._ax1ColorCnt += 1
      
    def trigger_single(self, signal, rdm=True):
        c = plotColors[self._ax1ColorCnt%len(plotColors)]  
        
        #self.clear()
        #startPoint = int(samplesPerPeriod/2   +bitShown*samplesPerPeriod)
        #endPoint   = int(samplesPerPeriod*5/2 +bitShown*samplesPerPeriod)
        fifth = len(signal.waveform['Time'])/5
        if rdm:
            sample = np.random.randint(signal._params['Number of samples'])
        else:
            sample = int(signal._params['Number of samples']/2)
        if 'Single' in signal.waveform['Sample 0'].keys():
            self._ax1.plot(signal.waveform['Time'][2*fifth:3*fifth:int(signal._params['Number of sampling points per period']*self._params['Sampling rate'])], signal.waveform['Sample %s'%sample]['Single'][2*fifth:3*fifth:int(signal._params['Number of sampling points per period']*self._params['Sampling rate'])], ls='-', lw=1, marker='', alpha=1, color=c)
            #self._ax1.plot(signal.waveform['Time'][::int(signal._params['Number of sampling points per period']*self._params['Sampling rate'])], signal.waveform['Sample %s'%sample]['Single'][::int(signal._params['Number of sampling points per period']*self._params['Sampling rate'])], ls='-', lw=1, marker='', alpha=1, color=c)
        else:
            self._ax1.plot(signal.waveform['Time'][::int(signal._params['Number of sampling points per period']*self._params['Sampling rate'])], signal.waveform['Sample %s'%sample]['Positive'][::int(signal._params['Number of sampling points per period']*self._params['Sampling rate'])], ls='-', lw=1, marker='', alpha=1, color=c)
            self._ax1.plot(signal.waveform['Time'][::int(signal._params['Number of sampling points per period']*self._params['Sampling rate'])], signal.waveform['Sample %s'%sample]['Negative'][::int(signal._params['Number of sampling points per period']*self._params['Sampling rate'])], ls='-', lw=1, marker='', alpha=1, color=c)
            
        self._ax1.set_title('Waveform')
        self._ax1.set_ylabel("amplitude")
        self._ax1.set_xlabel("time [s]")
        
        #if self._ax1plotCnt == 0:
        #    if 'Auto' in self._params['Vertical range']: 
        #        ymin, ymax = self._ax1.get_ylim()
        #        self._ax1.set_ylim(ymin*1.1,ymax*1.1)
        #    else:    
        #        self._ax1.set_ylim(-(self._params['Vertical range']/2),self._params['Vertical range']/2)
        
        self._ax1ColorCnt += 1
        self._ax1plotCnt += 1   
       
    def plotTransferfunction(self, tml, scale='db'):
        c = plotColors[self._ax3ColorCnt%len(plotColors)]  
        xl, yl = self._ax3.get_xlim()
        s21 = tml.dataDict
        f = np.arange(0,s21['Data'].shape[0])*tml._stepSize
        if scale == 'db':
            self._ax6.set_ylim(-60.,0.)
            self._ax6.plot(f[f<tml._fitMax], s21['Data'][f<tml._fitMax], marker='x', ls='', label='Transfer function data', c=c, ms=2)
           
        else:
            self._ax6.set_ylim(0.,1.)
            y = np.power(10,s21['Data']/20.)
            self._ax6.plot(f[f<tml._fitMax], y[f<tml._fitMax], marker='x', ls='', label='Transfer function data', c='black', ms=1)
              
        self._ax6.set_xlim(xl, yl)
        self._ax6.set_xlim(1e6,tml._fitMax)
        self._ax6.set_xscale('log')
        self._ax6.legend()
        self._ax3ColorCnt += 1
       
    def plotTransferfunctionModel(self, tml, scale='db'):
        c = plotColors[self._ax3ColorCnt%len(plotColors)]  
        xl, yl = self._ax3.get_xlim()
        if not hasattr(tml, '_transferFunctionModel'):
            tml.fitModel()
        y = tml._transferFunctionModel
        f = np.arange(0,y.shape[0])*tml._stepSize
        if scale == 'db':
            self._ax6.set_ylim(-60.,0.)
            self._ax6.plot(f, 20.*np.log10(y), label='Transfer function fit', c=c, lw=2, ls='-') 
        else:
            self._ax6.set_ylim(0.,1.)
            self._ax6.plot(f, y, label='Transfer function fit', c='black', lw=1) 
        
        self._ax6.set_xlim(xl, yl)
        self._ax6.set_xlim(1e6,tml._fitMax)
        self._ax6.set_xscale('log')
        self._ax6.legend(loc='lower left')
        self._ax3ColorCnt += 1
        
    def plotSpectrum(self, signal, titel='Default', scale='db', verbose=False):
        c = plotColors[self._ax3ColorCnt%len(plotColors)]  
        if 'Single' in signal.waveform['Sample 0'].keys():
            cnt = 0
            container = np.zeros((signal._params['Number of samples'],signal.spectrum['Data'].shape[1]))
            for sampleId, sampleValue in enumerate(signal.spectrum['Data']):
                    if verbose:
                        printProgress(cnt, signal._params['Number of samples']-1, prefix = 'plotting spectrum:', suffix = 'Complete', decimals=3, barLength = 50)
                    #self._ax3.semilogx(np.arange(0,sampleValue.shape[0])[::10]*40e6,signal.db(sampleValue)[::10], ls='', marker='_', color='red')
                    container[cnt,:] = np.abs(sampleValue)
                    cnt +=1
            mean = np.mean(container,axis=0)
            std = np.std(container,axis=0)
           
            mean_norm = mean #/np.max(mean)
            std_norm = std #/np.max(std)
                
            #t = np.arange(0,sampleValue.shape[0])[::1]*40e6*501/sampleValue.shape[0]
            if scale == 'db':
                ylo = (signal.db(mean_norm-std_norm/2))
                yhi = (signal.db(mean_norm+std_norm/2))
                y = signal.db(mean_norm)
                self._ax3.set_ylim(-60.,0.)       
                self._ax3.set_ylabel("Attenuation [dB]")
 
            else:
                ylo = (mean_norm-std_norm/2)
                yhi = (mean_norm+std_norm/2)
                y = mean_norm
                self._ax3.set_ylabel("Amplitude")
                #self._ax3.set_ylim('auto')   
                
            self._ax3.fill_between(signal.spectrum['Frequency'],ylo,yhi, alpha=0.15, color=c) 
            self._ax3.plot(signal.spectrum['Frequency'], y, ls='', marker='o', color=c, ms=2,label='%s\nf=%.2EHz'%(titel, Decimal(signal._params['Frequency'])), alpha=0.3)
        
        else:
            pass 
        self._ax3.set_xscale('log')
       #self._ax3.set_title('Spectrum')
        self._ax3.set_xlabel("Frequency")
        self._ax3.set_xlim(1e6,20.e9)
        self._ax3.legend(loc='lower left')
        self._ax3ColorCnt += 1
         
    def plotEye(self, signal, label='ref', histo=True, verbose=False):
        c = plotColors[self._ax5ColorCnt%len(plotColors)] 
        startPoint = int(-signal._params['Number of sampling points per period']/2   +self._params['Index of bit in center']*signal._params['Number of sampling points per period'])
        endPoint   = int(signal._params['Number of sampling points per period']*3/2 +self._params['Index of bit in center']*signal._params['Number of sampling points per period'])
        
        #baseFreq = signal.roundFrequency(signal._params['Frequency'])
        baseFreq = signal._params['Frequency']
        T = 1./(baseFreq)
            
        if not histo:
            cnt = 0
            if 'Single' in signal.waveform['Sample 0'].keys():
                for keysamples, valsamples in signal.waveform.iteritems():
                    if verbose:
                        printProgress(cnt, len(signal.waveform)-1, prefix = 'Recording eye:', suffix = 'Complete', decimals=3, barLength = 50)
                    if 'Time' not in keysamples:
                        self._ax1.plot(signal.waveform['Time'][startPoint:endPoint:int(signal._params['Number of sampling points per period']*self._params['Sampling rate'])], valsamples['Single'][startPoint:endPoint:int(signal._params['Number of sampling points per period']*self._params['Sampling rate'])], ls='-', lw=2,marker='', c=c, alpha=0.3)
                    cnt += 1
            else:
                for keysamples, valsamples in signal.waveform.iteritems():
                    if verbose:
                        printProgress(cnt, len(signal.waveform)-1, prefix = 'Recording eye:', suffix = 'Complete', decimals=3, barLength = 100)
                    if 'Time' not in keysamples:
                        rx = self.differentialProbing(valsamples['Positive'][startPoint:endPoint:int(signal._params['Number of sampling points per period']*self._params['Sampling rate'])], valsamples['Negative'][startPoint:endPoint:int(signal._params['Number of sampling points per period']*self._params['Sampling rate'])])
                        self._ax1.plot(rx, ls='-', lw=2,marker='', c=c, alpha=0.2)#signal.waveform['Time'][startPoint:endPoint:int(signal._params['Number of sampling points per period']*self._params['Sampling rate'])], 
                    cnt += 1
                       
            if self._plotCnt == 0:
                self._ax1.grid(lw=2, alpha=0.5, c='gray')
                self._ax1.set_title('Eye of bit number %s'%self._params['Index of bit in center']) 
                self._ax1.set_ylabel("amplitude")
                self._ax1.set_xlabel("time [s]")
            
                if 'Auto' in self._params['Vertical range']: 
                    ymin, ymax = self._ax1.get_ylim()
                    self._ax1.set_ylim(-(ymin)*1.1,(ymin)*1.1)
                else:    
                    self._ax1.set_ylim(-(self._params['Vertical range']/2),self._params['Vertical range']/2)
            #format_label_string_with_exponent(ax, axis='x')
        else:
            startPointHist = int((self._params['Index of bit in center']+1./2-0.1)*signal._params['Number of sampling points per period'])
            endPointHist   = int((self._params['Index of bit in center']+1./2+0.1)*signal._params['Number of sampling points per period'])
            
            h = []
            h1 = []
            cnt = 0
            if 'Single' in signal.waveform['Sample 0'].keys():
                for keysamples, valsamples in signal.waveform.iteritems():
                    if verbose:
                        printProgress(cnt, len(signal.waveform)-1, prefix = 'Recording eye diagram:', suffix = 'Complete', decimals=3, barLength = 50)
                    if 'Time' not in keysamples:
                        for b in range(22,len(signal._bits)-22):
                            st = int((b+1./2-0.05)*signal._params['Number of sampling points per period'])
                            ed = int((b+1./2+0.05)*signal._params['Number of sampling points per period'])
                            h.append(valsamples['Single'][st:ed])
                            st = int((b-0.5)*signal._params['Number of sampling points per period'])
                            ed = int((b+1.5)*signal._params['Number of sampling points per period'])
                            h1.append(valsamples['Single'][st:ed])
                            self._ax5.plot(signal.waveform['Time'][0:signal._params['Number of sampling points per period']*2:int(signal._params['Number of sampling points per period']*self._params['Sampling rate'])]-T, valsamples['Single'][st:ed:int(signal._params['Number of sampling points per period']*self._params['Sampling rate'])], ls='-', lw=0.5, marker='', c=c, alpha=1.)
                    cnt += 1
            else:
                for keysamples, valsamples in signal.waveform.iteritems():
                    if verbose:
                        printProgress(cnt, len(signal.waveform)-1, prefix = 'Recording eye:', suffix = 'Complete', decimals=3, barLength = 50)
                    if 'Time' not in keysamples:
                        rx = self.differentialProbing(valsamples['Positive'], valsamples['Negative'])
                        self._ax1.plot(signal.waveform['Time'][startPoint:endPoint:int(signal._params['Number of sampling points per period']*self._params['Sampling rate'])], rx[startPoint:endPoint:int(signal._params['Number of sampling points per period']*self._params['Sampling rate'])], ls='-', marker='', c=c, alpha=0.2)
                        h.append(rx[startPointHist:endPointHist])
                        h1.append(rx[startPoint:endPoint])
                    cnt += 1    

            self._ax5.text(0.5, 0.9, label, horizontalalignment='center', verticalalignment='center', fontsize=12, color='black', transform=self._ax5.transAxes)
            h = np.array(h)
            h1 = np.array(h1)            
            
            h1[h1<-0.01] = 0
            h1[h1>0.01] = 0
            
            histCnt, x, dv = simpleHist(h, 128)    
            self._ax2.barh(x, histCnt, height=x[1]-x[0], color=c, alpha=0.3, label='f=%.2EHz'%Decimal(signal._params['Frequency']))

            histCnt, x, step = TIEHist(h1)    
            width=signal.waveform['Time'][1]-signal.waveform['Time'][0]
        
            self._ax4.bar(x*width+signal.waveform['Time'][0]-T, histCnt/np.max(histCnt), width=width, color=plotColors[self._ax1ColorCnt%len(plotColors)+1] , alpha=0.3, label='f=%.2EHz'%Decimal(signal._params['Frequency']))
            self._ax4.set_ylabel("number of entries")
            self._ax4.set_xlabel("Time [s]")
            
            self._ax5.set_ylabel("amplitude [V]")
            self._ax5.set_xlim(-T,T)
            #self._ax2.set_ylabel("amplitude")
            self._ax2.set_xlabel("number of entries")
            if 'Auto' in self._params['Vertical range']: 
                ymin, ymax = self._ax1.get_ylim()
                self._ax1.set_ylim((ymin)*1.1,(ymax)*1.1)
                self._ax2.set_ylim((ymin)*1.1,(ymax)*1.1)
            else:    
                amp = float(self._params['Vertical range'])
                ymin, ymax = (-amp/2,amp/2)
                self._ax5.set_ylim(ymin, ymax)
                self._ax2.set_ylim(ymin, ymax)
            
            self._ax2.legend()        
            self._ax2plotCnt += 1
            self._ax5ColorCnt += 1
      
    
    def saveToDisk(self, filename):    
        savenamepdf = os.path.join(os.path.abspath('./'), filename+'.pdf')
        print 'Saved plot ', savenamepdf
        pdf = PdfPages(savenamepdf)    
        pdf.savefig(self._fig, dpi=900)
        pdf.close()

def simpleHist(data, div=100.):
    data = np.array(data)
    dmin = np.min(data)
    dmax = np.max(data)
    b = np.linspace(dmin, dmax, num=div)
    l, x = np.histogram(a=data, bins=b)
    return l, x[:len(x)-1], (dmax-dmin)/div

def TIEHist(data, step=1):
    cntInBin = 1
    l = []
    x = []
    data = data.T
    for i in range(0, data.shape[0], step):
        cntInBin = len(np.where(data[i*step:(i+1)*step] != 0)[0])   
        l.append(cntInBin)
        x.append(i*step)
    return np.array(l), np.array(x), step


def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """

    import sys
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = 100 * (iteration / float(total))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '|' * filledLength + '-' * (barLength - filledLength)
    if percents < 80.:
        clr = 'red'
    elif percents >= 80.:
        clr = 'green'
    sys.stdout.write(colored('\r%s\t |%s| %s%s %s' % (prefix, bar, formatStr.format(percents), '%', suffix), clr))
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
    
##############################################################