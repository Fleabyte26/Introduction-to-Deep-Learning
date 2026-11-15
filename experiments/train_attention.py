# Trains the tiny two-head self-attention network that runs inside attention.html. Needs numpy and tensorflow.
import os,sys,json
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np, tensorflow as tf
FILL=["very ","really ","quite "]; KMAX=8; D=16; NH=2; DH=8
chars=sorted(set(''.join(FILL)+"the cat dog sat on the mat log.")); V=len(chars); ix={c:i+1 for i,c in enumerate(chars)}
def sent(rng,animal,k,fixed=False):
    f=''.join(FILL[i%3] for i in range(k)) if fixed else ''.join(FILL[i] for i in rng.integers(0,3,k))
    return "the "+("cat" if animal==0 else "dog")+" "+f+"sat on the "+("mat" if animal==0 else "log")+"."
def posenc(L):
    p=np.arange(L)[:,None]; i=np.arange(D//2)[None,:]; a=p/np.power(10000,2*i/D); pe=np.zeros((L,D),'float32'); pe[:,0::2]=np.sin(a); pe[:,1::2]=np.cos(a); return pe
def batch(rng,n,k=None):
    S=[sent(rng,int(rng.integers(0,2)),int(rng.integers(0,KMAX+1)) if k is None else k) for _ in range(n)]
    L=max(len(s) for s in S)-1; X=np.zeros((n,L),'int32'); Y=np.zeros((n,L),'int32')
    for i,s in enumerate(S):
        a=[ix[c] for c in s]; X[i,:len(a)-1]=a[:-1]; Y[i,:len(a)-1]=a[1:]
    return X,Y,S
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
tf.random.set_seed(seed); g=tf.random.Generator.from_seed(seed)
def var(*s,sc=None): return tf.Variable(g.normal(s,stddev=sc or (1/np.sqrt(s[0]))))
Emb=var(V+1,D,sc=.5); Wq=var(NH,D,DH); Wk=var(NH,D,DH); Wv=var(NH,D,DH); Wo=var(NH*DH,D); W1=var(D,32); b1=tf.Variable(tf.zeros(32)); W2=var(32,V+1); b2=tf.Variable(tf.zeros(V+1))
P=[Emb,Wq,Wk,Wv,Wo,W1,b1,W2,b2]
def fwd(X):
    L=X.shape[1]; x=tf.gather(Emb,X)+posenc(L)[None]
    q=tf.einsum('bld,hde->bhle',x,Wq); k=tf.einsum('bld,hde->bhle',x,Wk); v=tf.einsum('bld,hde->bhle',x,Wv)
    s=tf.matmul(q,k,transpose_b=True)/np.sqrt(DH)
    mask=np.triu(np.ones((L,L),'float32'),1)[None,None]*-1e9; pad=tf.cast(X==0,'float32')[:,None,None,:]*-1e9
    w=tf.nn.softmax(s+mask+pad,-1); o=tf.matmul(w,v); o=tf.reshape(tf.transpose(o,[0,2,1,3]),(-1,L,NH*DH))
    h=x+o@Wo; z=tf.nn.relu(h@W1+b1)@W2+b2; return z,w
opt=tf.keras.optimizers.Adam(3e-3)
rng=np.random.default_rng(seed)
for it in range(1,2501):
    X,Y,_=batch(rng,32)
    with tf.GradientTape() as t:
        z,_=fwd(X); l=tf.nn.sparse_softmax_cross_entropy_with_logits(Y,z); m=tf.cast(Y>0,'float32'); loss=tf.reduce_sum(l*m)/tf.reduce_sum(m)
    opt.apply_gradients(zip(t.gradient(loss,P),P))
    if it%500==0: print(it,float(loss),flush=True)
acc=[]
r2=np.random.default_rng(99)
for k in range(0,15):
    X,Y,S=batch(r2,200,k); z,w=fwd(X); z=z.numpy(); ok=0
    for i,s in enumerate(S): ok+=int(z[i,len(s)-5].argmax()==ix[s[-4]])
    acc.append(ok/200)
print('acc',acc)
for animal in (0,1):
    s=sent(None,animal,5,True)[:-4]; X=np.array([[ix[c] for c in s]]); z,w=fwd(X); p=tf.nn.softmax(z[0,-1]).numpy(); w=w.numpy()[0,:,-1,:]
    print(repr(s),'m',round(float(p[ix['m']]),3),'l',round(float(p[ix['l']]),3))
    for h in range(NH): top=np.argsort(-w[h])[:5]; print('  head',h,[(int(j),s[j],round(float(w[h,j]),2)) for j in top])
json.dump(dict(chars=chars,D=D,NH=NH,DH=DH,Emb=np.round(Emb.numpy(),4).tolist(),Wq=np.round(Wq.numpy(),4).tolist(),Wk=np.round(Wk.numpy(),4).tolist(),Wv=np.round(Wv.numpy(),4).tolist(),Wo=np.round(Wo.numpy(),4).tolist(),W1=np.round(W1.numpy(),4).tolist(),b1=np.round(b1.numpy(),4).tolist(),W2=np.round(W2.numpy(),4).tolist(),b2=np.round(b2.numpy(),4).tolist(),acc=acc),open(f'attn_{seed}.json','w'),separators=(',',':'))
print('params',sum(int(np.prod(p.shape)) for p in P))
