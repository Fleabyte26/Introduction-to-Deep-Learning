# Trains the tiny transformer encoders used in transformer.html. Needs numpy and tensorflow.
#   python3 train.py page            -> the two 1-layer networks that run in the page (with and without positions)
#   python3 train.py depth           -> deep stacks with and without skip connections
import os,sys,json,itertools
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np, tensorflow as tf
ANIMALS=['cat','dog','fox','owl','hen','pig','bat','cow']; VERBS=['chased','saw','fed','heard']; ASK=['who-did-it','to-whom']
WORDS=['the']+ANIMALS+VERBS+ASK; ix={w:i for i,w in enumerate(WORDS)}; V=len(WORDS); NA=len(ANIMALS); D=16; NH=2; DH=8; F=32; L=6
ALL=[(s,o,v,q) for s in range(NA) for o in range(NA) if s!=o for v in range(4) for q in range(2)]
TEST=[e for e in ALL if (e[0]*7+e[1]*3+e[2])%5==0]; TRAIN=[e for e in ALL if e not in TEST]
def enc(e): s,o,v,q=e; return [ix['the'],ix[ANIMALS[s]],ix[VERBS[v]],ix['the'],ix[ANIMALS[o]],ix[ASK[q]]], (s if q==0 else o)
def arr(es): X,Y=zip(*[enc(e) for e in es]); return np.array(X,'int32'),np.array(Y,'int32')
def posenc():
    p=np.arange(L)[:,None]; i=np.arange(D//2)[None,:]; a=p/np.power(10000,2*i/D); pe=np.zeros((L,D),'float32'); pe[:,0::2]=np.sin(a); pe[:,1::2]=np.cos(a); return pe
PE=posenc()
def make(seed,layers,pe=True,skip=True):
    g=tf.random.Generator.from_seed(seed)
    def var(*s,sc=None): return tf.Variable(g.normal(s,stddev=sc or (1/np.sqrt(s[-2] if len(s)>1 else s[0]))))
    P=dict(Emb=var(V,D,sc=.5),Wout=var(D,NA),bout=tf.Variable(tf.zeros(NA)),gf=tf.Variable(tf.ones(D)),cf=tf.Variable(tf.zeros(D)),layers=[])
    for _ in range(layers):
        P['layers'].append(dict(Wq=var(NH,D,DH),Wk=var(NH,D,DH),Wv=var(NH,D,DH),Wo=var(NH*DH,D),g1=tf.Variable(tf.ones(D)),c1=tf.Variable(tf.zeros(D)),W1=var(D,F),b1=tf.Variable(tf.zeros(F)),W2=var(F,D),b2=tf.Variable(tf.zeros(D)),g2=tf.Variable(tf.ones(D)),c2=tf.Variable(tf.zeros(D))))
    def ln(x,g_,c_): m=tf.reduce_mean(x,-1,keepdims=True); s=tf.math.reduce_std(x,-1,keepdims=True); return (x-m)/(s+1e-5)*g_+c_
    def fwd(X):
        x=tf.gather(P['Emb'],X)+(PE[None] if pe else 0.); ws=[]
        def attn(x,Lr):
            q=tf.einsum('bld,hde->bhle',x,Lr['Wq']); k=tf.einsum('bld,hde->bhle',x,Lr['Wk']); v=tf.einsum('bld,hde->bhle',x,Lr['Wv'])
            w=tf.nn.softmax(tf.matmul(q,k,transpose_b=True)/np.sqrt(DH),-1); ws.append(w)
            return tf.reshape(tf.transpose(tf.matmul(w,v),[0,2,1,3]),(-1,L,NH*DH))@Lr['Wo']
        ffn=lambda x,Lr: tf.nn.relu(x@Lr['W1']+Lr['b1'])@Lr['W2']+Lr['b2']
        for Lr in P['layers']:
            if skip=='pre':                      # normalise first, then add: the layout most models use today
                x=x+attn(ln(x,Lr['g1'],Lr['c1']),Lr); x=x+ffn(ln(x,Lr['g2'],Lr['c2']),Lr)
            else:                                # the 2017 layout: add (or not), then normalise
                o=attn(x,Lr); x=ln((x+o) if skip else o,Lr['g1'],Lr['c1'])
                f=ffn(x,Lr); x=ln((x+f) if skip else f,Lr['g2'],Lr['c2'])
        if skip=='pre': x=ln(x,P['gf'],P['cf'])
        return x[:,-1]@P['Wout']+P['bout'],ws
    flat=[P['Emb'],P['Wout'],P['bout']]+([P['gf'],P['cf']] if skip=='pre' else [])+[t for Lr in P['layers'] for t in Lr.values()]
    return P,fwd,flat
def train(seed,layers,pe=True,skip=True,steps=1500,lr=3e-3,log=50,everything=False):
    tf.random.set_seed(seed); P,fwd,flat=make(seed,layers,pe,skip); opt=tf.keras.optimizers.Adam(lr); rng=np.random.default_rng(seed)
    Xtr,Ytr=arr(ALL if everything else TRAIN); Xte,Yte=arr(ALL if everything else TEST)   # 'everything': train and test on every sentence there is
    @tf.function
    def step(X,Y):
        with tf.GradientTape() as t: z,_=fwd(X); loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(Y,z))
        opt.apply_gradients(zip(t.gradient(loss,flat),flat)); return loss
    curve=[]
    for it in range(1,steps+1):
        i=rng.integers(0,len(Xtr),64); l=float(step(Xtr[i],Ytr[i]))
        if it%log==0: curve.append(round(float(np.mean(fwd(Xte)[0].numpy().argmax(1)==Yte)),4))
    acc=curve[-1]; n=sum(int(np.prod(t.shape)) for t in flat)
    return P,fwd,acc,curve,n,l
r=lambda t: np.round(t.numpy(),4).tolist()
if sys.argv[1]=='page':
    out={}
    for name,pe in (('pos',True),('nopos',False)):
        best=None
        for seed in range(3):
            P,fwd,acc,curve,n,l=train(seed,1,pe=pe,steps=2000,everything=not pe); print(name,'seed',seed,'test acc',acc,'loss',round(l,4),'params',n,flush=True)
            if best is None or acc>best[2]: best=(P,fwd,acc,curve,n,seed)
        P,fwd,acc,curve,n,seed=best; Lr=P['layers'][0]
        out[name]=dict(acc=acc,curve=curve,params=n,seed=seed,Emb=r(P['Emb']),Wout=r(P['Wout']),bout=r(P['bout']),**{k:r(v) for k,v in Lr.items()})
        for e in [(0,1,0,0),(1,0,0,0),(0,1,0,1),(1,0,0,1)]:
            X,Y=arr([e]); z,ws=fwd(X); p=tf.nn.softmax(z[0]).numpy(); print('  ',[WORDS[i] for i in X[0]],'->',[(ANIMALS[i],round(float(p[i]),3)) for i in np.argsort(-p)[:2]],'attn from ask',np.round(ws[0].numpy()[0,:,-1,:],2).tolist())
    out.update(words=WORDS,animals=ANIMALS,verbs=VERBS,D=D,NH=NH,DH=DH,F=F,PE=np.round(PE,4).tolist(),ntrain=len(TRAIN),ntest=len(TEST))
    json.dump(out,open('page.json','w'),separators=(',',':'))
else:
    res=json.load(open('depth.json')) if os.path.exists('depth.json') else {}
    for skip in ((True,False,'pre') if len(sys.argv)<3 else ('pre',)):
        for layers in (1,2,4,6,8,12):
            accs=[];curves=[]
            for seed in range(3):
                _,_,acc,curve,n,l=train(seed,layers,skip=skip,steps=1500); accs.append(acc); curves.append(curve)
            name='pre' if skip=='pre' else ('skip' if skip else 'noskip'); res[f"{name}_{layers}"]=dict(accs=accs,curves=curves,params=n); print(name,layers,accs,n,flush=True)
            json.dump(res,open('depth.json','w'),separators=(',',':'))
