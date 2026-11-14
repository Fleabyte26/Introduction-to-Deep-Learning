# Shared pieces for the experiments behind the explainer pages.
# Needs: numpy, tensorflow-cpu, and a clone of github.com/zalandoresearch/fashion-mnist (set FASHION_MNIST to its data/fashion/ folder).
import os,sys,json,gzip
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np, tensorflow as tf, keras
from keras import layers
D=os.environ.get('FASHION_MNIST','fashion-mnist/data/fashion/')
def load(p,off): return np.frombuffer(gzip.open(D+p).read(),np.uint8,offset=off)
Xtr=load('train-images-idx3-ubyte.gz',16).reshape(-1,28,28,1).astype('float32')/255; Ytr=load('train-labels-idx1-ubyte.gz',8).astype('int64')
Xte=load('t10k-images-idx3-ubyte.gz',16).reshape(-1,28,28,1).astype('float32')/255; Yte=load('t10k-labels-idx1-ubyte.gz',8).astype('int64')

def body():
    return keras.Sequential([layers.Input((28,28,1)),layers.Conv2D(32,3,padding='same',activation='relu'),layers.MaxPooling2D(),
        layers.Conv2D(64,3,padding='same',activation='relu'),layers.MaxPooling2D(),layers.Flatten(),layers.Dense(128,activation='relu')])

def run(b,head,xtr,ytr,xte,yte,epochs,lr_body,lr_head,aug=None,bs=32,seed=0):
    ob=keras.optimizers.Adam(lr_body) if lr_body>0 else None; oh=keras.optimizers.Adam(lr_head)
    lossf=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    @tf.function
    def step(x,y):
        if aug is not None: x=aug(x,training=True)
        with tf.GradientTape() as t:
            l=lossf(y,head(b(x,training=True)))
        vb=b.trainable_variables if ob is not None else []
        g=t.gradient(l,vb+head.trainable_variables)
        if ob is not None: ob.apply_gradients(zip(g[:len(vb)],vb))
        oh.apply_gradients(zip(g[len(vb):],head.trainable_variables))
    def acc(x,y):
        p=np.concatenate([head(b(x[i:i+1000],training=False)).numpy().argmax(1) for i in range(0,len(x),1000)]); return float((p==y).mean())
    rng=np.random.default_rng(seed); h={'tr':[],'te':[]}
    for ep in range(epochs):
        idx=rng.permutation(len(xtr))
        for s in range(0,len(idx),bs): step(tf.constant(xtr[idx[s:s+bs]]),tf.constant(ytr[idx[s:s+bs]]))
        h['tr'].append(round(acc(xtr,ytr),4)); h['te'].append(round(acc(xte,yte),4))
    return h

