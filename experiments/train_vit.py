# Trains the small vision transformer and the small CNN compared in vit.html. Needs numpy and tensorflow.
# Data: Fashion-MNIST (github.com/zalandoresearch/fashion-mnist), set FASHION_MNIST to its data/fashion/ folder
#   python3 train.py sizes     -> both models at four training-set sizes, two random starts each
#   python3 train.py shuffle   -> both models on images whose 49 patches are put in a fixed scrambled order
#   python3 train.py page      -> the vision transformer that runs in the page, plus a version with no position tags
import os,sys,json,gzip
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np, tensorflow as tf
DD=os.environ.get('FASHION_MNIST','/home/claude/zalandoresearch/fashion-mnist/data/fashion/')
def load(p,off): return np.frombuffer(gzip.open(DD+p).read(),np.uint8,offset=off)
Xtr=load('train-images-idx3-ubyte.gz',16).reshape(-1,28,28).astype('float32')/255; Ytr=load('train-labels-idx1-ubyte.gz',8).astype('int32')
Xte=load('t10k-images-idx3-ubyte.gz',16).reshape(-1,28,28).astype('float32')/255; Yte=load('t10k-labels-idx1-ubyte.gz',8).astype('int32')
PS=4; G=7; NP=G*G; D=32; NH=2; DH=16; F=64; LAYERS=2
def patches(X): return X.reshape(-1,G,PS,G,PS).transpose(0,1,3,2,4).reshape(-1,NP,PS*PS)      # 49 patches of 16 pixels, row by row
def unpatch(Pt): return Pt.reshape(-1,G,G,PS,PS).transpose(0,1,3,2,4).reshape(-1,28,28)
PERM=np.random.default_rng(7).permutation(NP)
def scramble(X): return unpatch(patches(X)[:,PERM])                                                # same scrambled order for every image
def make_vit(seed,pos=True):
    g=tf.random.Generator.from_seed(seed)
    var=lambda *s,sc=None: tf.Variable(g.normal(s,stddev=sc or 1/np.sqrt(s[0])))
    P=dict(We=var(PS*PS,D),be=tf.Variable(tf.zeros(D)),cls=var(D,sc=.2),pos=var(NP+1,D,sc=.2),gf=tf.Variable(tf.ones(D)),cf=tf.Variable(tf.zeros(D)),Wh=var(D,10),bh=tf.Variable(tf.zeros(10)),layers=[])
    for _ in range(LAYERS): P['layers'].append(dict(g1=tf.Variable(tf.ones(D)),c1=tf.Variable(tf.zeros(D)),Wq=var(D,D),Wk=var(D,D),Wv=var(D,D),Wo=var(D,D),g2=tf.Variable(tf.ones(D)),c2=tf.Variable(tf.zeros(D)),W1=var(D,F),b1=tf.Variable(tf.zeros(F)),W2=var(F,D),b2=tf.Variable(tf.zeros(D))))
    def ln(x,g_,c_): m=tf.reduce_mean(x,-1,keepdims=True); s=tf.math.reduce_std(x,-1,keepdims=True); return (x-m)/(s+1e-5)*g_+c_
    def split(t): return tf.transpose(tf.reshape(t,(-1,NP+1,NH,DH)),[0,2,1,3])
    def fwd(Pt):
        x=Pt@P['We']+P['be']; x=tf.concat([tf.tile(P['cls'][None,None],[tf.shape(x)[0],1,1]),x],1)
        if pos: x=x+P['pos'][None]
        for Lr in P['layers']:
            h=ln(x,Lr['g1'],Lr['c1']); w=tf.nn.softmax(tf.matmul(split(h@Lr['Wq']),split(h@Lr['Wk']),transpose_b=True)/np.sqrt(DH),-1)
            o=tf.reshape(tf.transpose(tf.matmul(w,split(h@Lr['Wv'])),[0,2,1,3]),(-1,NP+1,D))@Lr['Wo']; x=x+o
            h=ln(x,Lr['g2'],Lr['c2']); x=x+tf.nn.relu(h@Lr['W1']+Lr['b1'])@Lr['W2']+Lr['b2']
        return ln(x[:,0],P['gf'],P['cf'])@P['Wh']+P['bh']
    flat=[v for k,v in P.items() if k!='layers' and (pos or k!='pos')]+[t for Lr in P['layers'] for t in Lr.values()]
    return P,fwd,flat
def make_cnn(seed):
    tf.keras.utils.set_random_seed(seed); L=tf.keras.layers
    m=tf.keras.Sequential([L.Input((28,28,1)),L.Conv2D(16,3,padding='same',activation='relu'),L.MaxPooling2D(),L.Conv2D(32,3,padding='same',activation='relu'),L.MaxPooling2D(),L.Flatten(),L.Dense(10)])
    return m,(lambda X: m(X[...,None])),m.trainable_variables
STEPS={500:1500,2000:2500,8000:4000,60000:6000}
def run(kind,seed,n,shuf=False,pos=True):
    tf.random.set_seed(seed); rng=np.random.default_rng(seed); idx=rng.permutation(len(Xtr))[:n]
    A,B=(scramble(Xtr[idx]),scramble(Xte)) if shuf else (Xtr[idx],Xte); Y=Ytr[idx]
    if kind=='vit': P,f,flat=make_vit(seed,pos); A,B=patches(A),patches(B)
    else: P,f,flat=make_cnn(seed)
    opt=tf.keras.optimizers.Adam(1e-3)
    @tf.function
    def step(x,y):
        with tf.GradientTape() as t: loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y,f(x)))
        opt.apply_gradients(zip(t.gradient(loss,flat),flat)); return loss
    for it in range(STEPS[n]):
        i=rng.integers(0,n,64); step(A[i],Y[i])
    pred=np.concatenate([f(B[i:i+2000]).numpy().argmax(1) for i in range(0,len(B),2000)])
    return P,f,float(np.mean(pred==Yte)),sum(int(np.prod(v.shape)) for v in flat)
mode=sys.argv[1]
if mode=='sizes':
    res={}
    for n in STEPS:
        for kind in ('cnn','vit'):
            accs=[];
            for seed in (0,1): _,_,a,p=run(kind,seed,n); accs.append(round(a,4))
            res[f'{kind}_{n}']=dict(accs=accs,params=p); print(kind,n,accs,p,flush=True); json.dump(res,open('sizes.json','w'))
elif mode=='shuffle':
    res={}
    for kind in ('cnn','vit'):
        accs=[]
        for seed in (0,1): _,_,a,p=run(kind,seed,60000,shuf=True); accs.append(round(a,4))
        res[kind]=accs; print('scrambled',kind,accs,flush=True); json.dump(res,open('shuffle.json','w'))
else:
    P,f,a,p=run('vit',0,60000); print('page vit',a,p,flush=True)
    _,_,a0,p0=run('vit',0,60000,pos=False); print('no position tags',a0,p0,flush=True)
    r=lambda t: np.round(t.numpy(),3).tolist()
    pick=[]; 
    for c in range(10): pick+= [int(i) for i in np.where(Yte==c)[0][:2]]
    pr=f(patches(Xte[pick])).numpy().argmax(1)
    out=dict(acc=a,acc_nopos=a0,params=p,perm=PERM.tolist(),imgs=[''.join('%02x'%v for v in (Xte[i]*255).astype('uint8').ravel()) for i in pick],labels=[int(Yte[i]) for i in pick],pred=[int(v) for v in pr],
             **{k:r(v) for k,v in P.items() if k!='layers'},layers=[{k:r(v) for k,v in Lr.items()} for Lr in P['layers']])
    json.dump(out,open('page.json','w'),separators=(',',':')); print('picked labels',out['labels'],'pred',out['pred'])
