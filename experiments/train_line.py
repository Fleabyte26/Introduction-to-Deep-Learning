# Trains the small network that grades the drawn fruit, and runs the measurements shown in line.html.
# Needs numpy and tensorflow, and the .npz picture files written by gen.py (which draws the fruit in headless Chrome).
import os,json
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np, tensorflow as tf
L=tf.keras.layers; CONDS=['good','bruised','rotten','underripe','cut']
ld=lambda n: np.load(n+'.npz'); tr=ld('train'); te=ld('test'); Xtr=tr['X'][:,0]/255.; Ytr=tr['Y']; Xte=te['X'][:,0]/255.; Yte=te['Y']
def net(seed):
    tf.keras.utils.set_random_seed(seed)
    return tf.keras.Sequential([L.Input((32,32,3)),L.Conv2D(12,3,padding='same',activation='relu'),L.MaxPooling2D(),L.Conv2D(24,3,padding='same',activation='relu'),L.MaxPooling2D(),L.Conv2D(32,3,padding='same',activation='relu'),L.MaxPooling2D(),L.Flatten(),L.Dense(5)])
def fit(seed,n,steps):
    m=net(seed); m.compile(tf.keras.optimizers.Adam(1e-3),tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    rng=np.random.default_rng(seed); idx=np.concatenate([np.where(Ytr==c)[0][:n//5] for c in range(5)]); bs=min(32,n)
    m.fit(Xtr[idx],Ytr[idx],batch_size=bs,epochs=max(1,steps*bs//len(idx)),verbose=0); return m
prob=lambda m,X: tf.nn.softmax(m.predict(X,verbose=0,batch_size=500)).numpy()
out={}
out['sizes']={n:[round(float(np.mean(prob(fit(s,n,1500),Xte).argmax(1)==Yte)),4) for s in (0,1)] for n in (25,100,500,6000)}; print('sizes',out['sizes'],flush=True)
m=fit(0,6000,3000); p=prob(m,Xte); out['acc']=round(float(np.mean(p.argmax(1)==Yte)),4)
out['confusion']=[[int(np.sum((Yte==a)&(p.argmax(1)==b))) for b in range(5)] for a in range(5)]; print('acc',out['acc'],out['confusion'],flush=True)
out['blur']=[]
for i in range(6): d=ld(f'blur{i}'); out['blur'].append(round(float(np.mean(prob(m,d['X'][:,0]/255.).argmax(1)==d['Y'])),4))
print('blur',out['blur'],flush=True)
ln=ld('line'); Xl=ln['X']/255.; Yl=ln['Y']; P=prob(m,Xl.reshape(-1,32,32,3)).reshape(len(Yl),9,5)
out['votes']={k:round(float(np.mean(P[:,:k].mean(1).argmax(1)==Yl)),4) for k in (1,3,5,9)}; print('votes',out['votes'],flush=True)
out['line']=dict(truth=Yl.tolist(),pgood1=np.round(P[:,0,0]*255).astype(int).tolist(),pgood5=np.round(P[:,:5].mean(1)[:,0]*255).astype(int).tolist(),pred5=P[:,:5].mean(1).argmax(1).tolist())
out['params']=int(m.count_params()); r=lambda a: np.round(a,3).tolist()
ws=m.get_weights(); out['net']=dict(k1=r(ws[0]),b1=r(ws[1]),k2=r(ws[2]),b2=r(ws[3]),k3=r(ws[4]),b3=r(ws[5]),W=r(ws[6]),b=r(ws[7]))
json.dump(out,open('results.json','w'),separators=(',',':')); print('params',out['params'])
