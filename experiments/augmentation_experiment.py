# The experiment behind image-augmentation.html: one small CNN trained four ways on 1,000 Fashion-MNIST pictures,
# then tested on normal, mirrored and shifted test pictures. Three seeds; the page shows the average.
import os,sys,json
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np, tensorflow as tf, keras
from keras import layers
import train_utils as kexp
from train_utils import Xtr,Ytr,Xte,Yte,body
def shift(x,dx,dy):
    o=np.zeros_like(x); o[:,dy:,dx:]=x[:,:28-dy,:28-dx]; return o
def one(seed,per=100,epochs=40):
    rng=np.random.default_rng(seed)
    tr=np.concatenate([rng.permutation(np.where(Ytr==c)[0])[:per] for c in range(10)])
    ti=rng.permutation(len(Xte))[:3000]; xo=Xte[ti]; y=Yte[ti]
    sets={'orig':xo,'mirror':xo[:,:,::-1].copy(),'shift':shift(xo,2,2)}
    augs={'none':None,'flip':keras.Sequential([layers.RandomFlip('horizontal')]),
          'crop':keras.Sequential([layers.RandomTranslation(0.1,0.1,fill_mode='constant')]),
          'both':keras.Sequential([layers.RandomFlip('horizontal'),layers.RandomTranslation(0.1,0.1,fill_mode='constant')])}
    res={}
    for k,a in augs.items():
        keras.utils.set_random_seed(seed+1)
        b=body(); head=layers.Dense(10)
        h=kexp.run(b,head,Xtr[tr],Ytr[tr],xo,y,epochs,1e-3,1e-3,a,seed=seed)
        def acc(x): return float((head(b(x,training=False)).numpy().argmax(1)==y).mean())
        h['final']={n:round(acc(x),4) for n,x in sets.items()}
        # per-class on mirror for shoes
        p=head(b(sets['mirror'],training=False)).numpy().argmax(1)
        h['mirror_cls']=[round(float((p[y==c]==c).mean()),3) for c in range(10)]
        p=head(b(sets['orig'],training=False)).numpy().argmax(1)
        h['orig_cls']=[round(float((p[y==c]==c).mean()),3) for c in range(10)]
        res[k]=h; print(seed,k,h['final'],h['tr'][-1],flush=True)
    return res
out=[one(s) for s in (0,1,2)]
json.dump(out,open('res_aug.json','w'))
