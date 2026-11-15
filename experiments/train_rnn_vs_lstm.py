# Trains a plain RNN and an LSTM on the same task for lstm.html, and measures how far back each can remember.
import os,sys,json
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np, tensorflow as tf, keras
from keras import layers
FILL=["very ","really ","quite "]; KMAX=int(sys.argv[1]) if len(sys.argv)>1 else 8; H=24; E=8
def sent(rng,animal,k):
    return "the "+("cat" if animal==0 else "dog")+" "+''.join(FILL[i] for i in rng.integers(0,3,k))+"sat on the "+("mat" if animal==0 else "log")+"."
chars=sorted(set(''.join(FILL)+"the cat dog sat on the mat log.")); V=len(chars); ix={c:i+1 for i,c in enumerate(chars)}
def batch(rng,n,kfix=None):
    S=[sent(rng,int(rng.integers(0,2)),int(rng.integers(0,KMAX+1)) if kfix is None else kfix) for _ in range(n)]
    L=max(len(s) for s in S)-1; X=np.zeros((n,L),'int32'); Y=np.zeros((n,L),'int32')
    for i,s in enumerate(S):
        a=[ix[c] for c in s]; X[i,:len(a)-1]=a[:-1]; Y[i,:len(a)-1]=a[1:]
    return X,Y,S
def build(kind,seed):
    keras.utils.set_random_seed(seed)
    cell=layers.SimpleRNN(H,return_sequences=True) if kind=='rnn' else layers.LSTM(H,return_sequences=True)
    m=keras.Sequential([layers.Input((None,)),layers.Embedding(V+1,E,mask_zero=True),cell,layers.Dense(V+1,activation='softmax')])
    m.compile(optimizer=keras.optimizers.Adam(3e-3,clipnorm=1.0),loss='sparse_categorical_crossentropy'); return m
def acc_by_k(m,rng,ks,n=200):
    out=[]
    for k in ks:
        X,Y,S=batch(rng,n,k); P=m.predict(X,verbose=0); ok=0; pr=0
        for i,s in enumerate(S):
            t=len(s)-5  # input position of the space before mat/log -> predicts first letter
            want=ix[s[-4]]; ok+=int(P[i,t].argmax()==want); pr+=P[i,t,want]
        out.append((k,ok/n,float(pr/n)))
    return out
if __name__=='__main__':
    seed=int(sys.argv[2]) if len(sys.argv)>2 else 0; res={}
    rng=np.random.default_rng(seed); Xtr,Ytr,_=batch(rng,4000)
    for kind in ('rnn','lstm'):
        m=build(kind,seed); h=m.fit(Xtr,Ytr,batch_size=32,epochs=12,verbose=0)
        a=acc_by_k(m,np.random.default_rng(99),list(range(0,15)))
        res[kind]=dict(acc=a,loss=[float(x) for x in h.history['loss']],params=m.count_params())
        print(kind,'params',m.count_params(),'loss',round(h.history['loss'][-1],4)); print('  ',[(k,round(x,2)) for k,x,_ in a],flush=True)
        m.save(f'{kind}_{KMAX}_{seed}.keras')
    json.dump(res,open(f'res_{KMAX}_{seed}.json','w'))
