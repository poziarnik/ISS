import numpy as np
from scipy.signal.ltisys import impulse2
from scipy.signal.signaltools import lfilter
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

np.set_printoptions(threshold=np.inf)
mySignal, fs = sf.read('xhatal01.wav')
print(fs)
s=mySignal.size
print("dlzka signalu:",s)
print("cas:",s/16000)
t = np.arange(mySignal.size) / fs

print("max:",mySignal.max())
print("min:",mySignal.min())

plt.subplot(2,1,2)
plt.plot(t, mySignal)
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title('Zvukový signál')

plt.tight_layout()
#plt.close()

myMatrix = np.zeros((1024,0))
y=0
z=1024

for x in range(64):
    newColon = np.array(mySignal[y:z])
    y=y+512
    z=z+512
    myMatrix = np.insert(myMatrix,x,newColon,axis=1)

############################################### hladanie rusivych frekvencii

firstFrame=myMatrix[:,1]
WaveFr=np.fft.fft(firstFrame)
findingFr=np.fft.fftfreq(1024,d=1/16000)
halfSpec=findingFr
x=findingFr
y=np.abs(WaveFr)
plt.figure(figsize=(6,3))
graph = plt.plot(abs(x),y)
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title('Hladanie rusivych frekvencii')


############################################### vykreslenie ramca
frame=myMatrix[:,50]


t = np.arange(0,frame.size) / fs
plt.figure(figsize=(9,3))
plt.plot(t, frame)
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title('Rámec')

############################################### spektrogram

plt.figure(figsize=(9,3))
plt.specgram(mySignal,NFFT=1024,Fs=fs,noverlap=512)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
plt.tight_layout()

transformedSignal=np.fft.fft(mySignal)



############################################### cosinusovky

t = np.arange(0,mySignal.size)
freq = 819 
cos1 = np.cos(2*np.pi*freq*(t/16000))

t = np.arange(0,mySignal.size)
freq = 1796
cos2 = np.cos(2*np.pi*freq*(t/16000))

t = np.arange(0,mySignal.size)
freq = 2688
cos3 = np.cos(2*np.pi*freq*(t/16000))

t = np.arange(0,mySignal.size)
freq = 3594
cos4 = np.cos(2*np.pi*freq*(t/16000))

cos=cos1+cos2+cos3+cos4

plt.figure(figsize=(9,3))
plt.specgram(cos,NFFT=1024,Fs=fs,noverlap=512)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
plt.tight_layout()

wavfile.write('4cos.wav', 16000,(cos * np.iinfo(np.int16).max).astype(np.int16))

################################################ filtracia

nyc=8000 #16000/2

fr1=891
left,right=signal.buttord([(fr1-90)/nyc,(fr1+90)/nyc],[(fr1-30)/nyc,(fr1+30)/nyc],3,40)
b,a= signal.butter(left, right, btype = 'bandstop')

fr2=1782
left,right=signal.buttord([(fr2-90)/nyc,(fr2+90)/nyc],[(fr2-30)/nyc,(fr2+30)/nyc],3,40)
d,c= signal.butter(left, right, btype = 'bandstop')

fr3=2673
left,right=signal.buttord([(fr3-90)/nyc,(fr3+90)/nyc],[(fr3-30)/nyc,(fr3+30)/nyc],3,40)
f,e= signal.butter(left, right, btype = 'bandstop')

fr4=3565
left,right=signal.buttord([(fr4-90)/nyc,(fr4+90)/nyc],[(fr4-30)/nyc,(fr4+30)/nyc],3,40)
h,g= signal.butter(left, right, btype = 'bandstop')

toFilter = mySignal
signalSize = mySignal.size

filteredSig = signal.lfilter(b,a,toFilter)
filteredSig = signal.lfilter(d,c,filteredSig)
filteredSig = signal.lfilter(f,e,filteredSig)
filteredSig = signal.lfilter(h,g,filteredSig)

################################################ impluzna odozva

imp = [1, *np.zeros(32-1)]
imp1 = lfilter(b,a,imp)

plt.figure(figsize=(5,3))
plt.stem(np.arange(32), imp1, basefmt=' ')
plt.gca().set_xlabel('$n$')
plt.gca().set_title('Impulsní odezva filter1')

imp2 = lfilter(d,c,imp)

plt.figure(figsize=(5,3))
plt.stem(np.arange(32), imp2, basefmt=' ')
plt.gca().set_xlabel('$n$')
plt.gca().set_title('Impulsní odezva filter2')

imp3 = lfilter(f,e,imp)

plt.figure(figsize=(5,3))
plt.stem(np.arange(32), imp3, basefmt=' ')
plt.gca().set_xlabel('$n$')
plt.gca().set_title('Impulsní odezva filter3')

imp4 = lfilter(h,g,imp)

plt.figure(figsize=(5,3))
plt.stem(np.arange(32), imp4, basefmt=' ')
plt.gca().set_xlabel('$n$')
plt.gca().set_title('Impulsní odezva filter4')

print("b:",b)
print("a:",a)
print("d:",d)
print("c:",c)
print("f:",f)
print("e:",e)
print("h:",h)
print("g:",g)

################################################ nuly a poly

z, p, k = signal.tf2zpk(b, a)
plt.figure(figsize=(4,3.5))
ang = np.linspace(0, 2*np.pi,100)
plt.plot(np.cos(ang), np.sin(ang))
plt.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='r', label='nuly')
plt.scatter(np.real(p), np.imag(p), marker='x', color='g', label='póly')
plt.gca().set_title('Nulove body a poly filter1')

z, p, k = signal.tf2zpk(d, c)
plt.figure(figsize=(4,3.5))
ang = np.linspace(0, 2*np.pi,100)
plt.plot(np.cos(ang), np.sin(ang))
plt.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='r', label='nuly')
plt.scatter(np.real(p), np.imag(p), marker='x', color='g', label='póly')
plt.gca().set_title('Nulove body a poly filter2')

z, p, k = signal.tf2zpk(f, e)
plt.figure(figsize=(4,3.5))
ang = np.linspace(0, 2*np.pi,100)
plt.plot(np.cos(ang), np.sin(ang))
plt.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='r', label='nuly')
plt.scatter(np.real(p), np.imag(p), marker='x', color='g', label='póly')
plt.gca().set_title('Nulove body a poly filter3')

z, p, k = signal.tf2zpk(h, g)
plt.figure(figsize=(4,3.5))
ang = np.linspace(0, 2*np.pi,100)
plt.plot(np.cos(ang), np.sin(ang))
plt.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='r', label='nuly')
plt.scatter(np.real(p), np.imag(p), marker='x', color='g', label='póly')
plt.gca().set_title('Nulove body a poly filter4')

################################################ frekvencna charakteristika

w, H = signal.freqz(b, a)

plt.figure(figsize=(5,3))
plt.plot(w / 2 / np.pi * fs, np.abs(H))
plt.gca().set_xlabel('Frekvence [Hz]')
plt.gca().set_title('Modul frekvenční charakteristiky filter1')

plt.figure(figsize=(5,3))
plt.plot(w / 2 / np.pi * fs, np.angle(H))
plt.gca().set_xlabel('Frekvence [Hz]')
plt.gca().set_title('Argument frekvenční charakteristiky filter1')

w, H = signal.freqz(d, c)

plt.figure(figsize=(5,3))
plt.plot(w / 2 / np.pi * fs, np.abs(H))
plt.gca().set_xlabel('Frekvence [Hz]')
plt.gca().set_title('Modul frekvenční charakteristiky filter2')

plt.figure(figsize=(5,3))
plt.plot(w / 2 / np.pi * fs, np.angle(H))
plt.gca().set_xlabel('Frekvence [Hz]')
plt.gca().set_title('Argument frekvenční charakteristiky filter2')

w, H = signal.freqz(f, e)

plt.figure(figsize=(5,3))
plt.plot(w / 2 / np.pi * fs, np.abs(H))
plt.gca().set_xlabel('Frekvence [Hz]')
plt.gca().set_title('Modul frekvenční charakteristiky filter3')

plt.figure(figsize=(5,3))
plt.plot(w / 2 / np.pi * fs, np.angle(H))
plt.gca().set_xlabel('Frekvence [Hz]')
plt.gca().set_title('Argument frekvenční charakteristiky filter3')

w, H = signal.freqz(h, g)

plt.figure(figsize=(5,3))
plt.plot(w / 2 / np.pi * fs, np.abs(H))
plt.gca().set_xlabel('Frekvence [Hz]')
plt.gca().set_title('Modul frekvenční charakteristiky filter4')

plt.figure(figsize=(5,3))
plt.plot(w / 2 / np.pi * fs, np.angle(H))
plt.gca().set_xlabel('Frekvence [Hz]')
plt.gca().set_title('Argument frekvenční charakteristiky filter4')


################################################ vykreslenie upraveneho signalu
plt.figure(figsize=(9,3))
plt.subplot(2,1,2)
plt.plot(np.arange(0, signalSize),filteredSig)

plt.subplot(2,1,2)
plt.plot(np.arange(0, signalSize),mySignal)

# plt.subplot(2,1,2)
# plt.plot(np.arange(0, signalSize),frame)

wavfile.write('clean_bandstop.wav', 16000,(filteredSig * np.iinfo(np.int16).max).astype(np.int16))