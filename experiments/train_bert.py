# Trains the tiny masked-word models used in bert.html. Needs numpy and tensorflow.
#   python3 train.py page       -> two fill-in-the-blank models: one reads both ways (BERT style), one only left to right
#   python3 train.py finetune   -> pretrain then fine-tune, against training from scratch, by number of labelled examples
import os,sys,json
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np, tensorflow as tf
ANIMALS=['cat','dog','hen','owl','pig','bat','fox','bee','cow','ant','eel','yak']
PLACES =['mat','log','nest','tree','mud','cave','den','hive','barn','hill','reef','peak']      # each animal has one home, same index
PREPS  =['on','on','in','in','in','in','in','in','in','on','on','on']                       # the preposition goes with the place
VERBS=['sat','slept','hid','played','waited','rested']; ADJS=['big','small','old','young','quiet','happy']
WORDS=['[CLS]','[MASK]','the','on','in']+ADJS+ANIMALS+VERBS+PLACES; ix={w:i for i,w in enumerate(WORDS)}; V=len(WORDS)
T=8; D=24; NH=2; DH=12; F=48; LAYERS=2
def sent(a,v,j,p=None): p=a if p is None else p; return ['[CLS]','the',ADJS[j],ANIMALS[a],VERBS[v],PREPS[p],'the',PLACES[p]]
enc=lambda s:[ix[w] for w in s]
COMBOS=[(a,v,j) for a in range(12) for v in range(6) for j in range(6)]
HELD=[c for c in COMBOS if (c[0]*5+c[1]*3+c[2]*7)%5==0]; SEEN=[c for c in COMBOS if c not in HELD]
def make(seed,causal=False):
    g=tf.random.Generator.from_seed(seed); var=lambda *s,sc=None: tf.Variable(g.normal(s,stddev=sc or 1/np.sqrt(s[0])))
    P=dict(Emb=var(V,D,sc=.3),Pos=var(T,D,sc=.3),gf=tf.Variable(tf.ones(D)),cf=tf.Variable(tf.zeros(D)),Wout=var(D,V),bout=tf.Variable(tf.zeros(V)),Wc=var(D,2),bc=tf.Variable(tf.zeros(2)),layers=[])
    for _ in range(LAYERS): P['layers'].append(dict(g1=tf.Variable(tf.ones(D)),c1=tf.Variable(tf.zeros(D)),Wq=var(D,D),Wk=var(D,D),Wv=var(D,D),Wo=var(D,D),g2=tf.Variable(tf.ones(D)),c2=tf.Variable(tf.zeros(D)),W1=var(D,F),b1=tf.Variable(tf.zeros(F)),W2=var(F,D),b2=tf.Variable(tf.zeros(D))))
    ln=lambda x,g_,c_: (x-tf.reduce_mean(x,-1,keepdims=True))/(tf.math.reduce_std(x,-1,keepdims=True)+1e-5)*g_+c_
    split=lambda t: tf.transpose(tf.reshape(t,(-1,T,NH,DH)),[0,2,1,3])
    MASK=(np.triu(np.ones((T,T),'float32'),1)*-1e9)[None,None] if causal else 0.          # causal: a word may not look to its right
    def body(X):
        x=tf.gather(P['Emb'],X)+P['Pos'][None]; ws=[]
        for Lr in P['layers']:
            h=ln(x,Lr['g1'],Lr['c1']); w=tf.nn.softmax(tf.matmul(split(h@Lr['Wq']),split(h@Lr['Wk']),transpose_b=True)/np.sqrt(DH)+MASK,-1); ws.append(w)
            x=x+tf.reshape(tf.transpose(tf.matmul(w,split(h@Lr['Wv'])),[0,2,1,3]),(-1,T,D))@Lr['Wo']
            h=ln(x,Lr['g2'],Lr['c2']); x=x+tf.nn.relu(h@Lr['W1']+Lr['b1'])@Lr['W2']+Lr['b2']
        return ln(x,P['gf'],P['cf']),ws
    words=lambda X: body(X)[0]@P['Wout']+P['bout']            # a guess for every position
    label=lambda X: body(X)[0][:,0]@P['Wc']+P['bc']           # one yes/no answer, read from [CLS]
    core=[P['Emb'],P['Pos'],P['gf'],P['cf']]+[t for Lr in P['layers'] for t in Lr.values()]
    return P,words,label,core,body
def pretrain(seed,causal=False,steps=3000):
    tf.random.set_seed(seed); rng=np.random.default_rng(seed); P,words,label,core,body=make(seed,causal); flat=core+[P['Wout'],P['bout']]; opt=tf.keras.optimizers.Adam(2e-3)
    S=np.array([enc(sent(*c)) for c in SEEN],'int32')
    @tf.function
    def step(X,pos,Y):
        with tf.GradientTape() as t: z=tf.gather(words(X),pos,batch_dims=1); loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(Y,z))
        opt.apply_gradients(zip(t.gradient(loss,flat),flat)); return loss
    for it in range(steps):
        X=S[rng.integers(0,len(S),64)].copy(); pos=rng.integers(1,T,64).astype('int32'); Y=X[np.arange(64),pos].copy(); r=rng.random(64)
        X[np.arange(64),pos]=np.where(r<.8,ix['[MASK]'],np.where(r<.9,rng.integers(2,V,64),Y))     # the 80 / 10 / 10 recipe
        step(X,pos,Y)
    H=np.array([enc(sent(*c)) for c in HELD],'int32'); acc={}
    for name,pos in (('adjective',2),('animal',3),('verb',4),('place',7)):
        X=H.copy(); X[:,pos]=ix['[MASK]']; acc[name]=round(float(np.mean(words(X).numpy()[:,pos].argmax(1)==H[:,pos])),4)
    return P,words,label,core,acc
def labelled(rng,n,combos):
    X=[];Y=[]
    for i in range(n):
        a,v,j=combos[rng.integers(0,len(combos))]; ok=i%2
        p=a if ok else int((a+rng.integers(1,12))%12); X.append(enc(sent(a,v,j,p))); Y.append(ok)
    return np.array(X,'int32'),np.array(Y,'int32')
def finetune(seed,n,pre):
    rng=np.random.default_rng(100+seed)
    if pre: P,words,label,core,_=pretrain(seed)
    else: tf.random.set_seed(seed); P,words,label,core,_=make(seed)
    flat=core+[P['Wc'],P['bc']]; opt=tf.keras.optimizers.Adam(1e-3); Xtr,Ytr=labelled(rng,n,SEEN); Xte,Yte=labelled(np.random.default_rng(999),1000,HELD)
    @tf.function
    def step(X,Y):
        with tf.GradientTape() as t: loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(Y,label(X)))
        opt.apply_gradients(zip(t.gradient(loss,flat),flat)); return loss
    for it in range(400):
        i=rng.integers(0,n,min(n,32)); step(Xtr[i],Ytr[i])
    return round(float(np.mean(label(Xte).numpy().argmax(1)==Yte)),4)
r=lambda t: np.round(t.numpy(),3).tolist()
if sys.argv[1]=='page':
    out=dict(words=WORDS,animals=ANIMALS,places=PLACES,preps=PREPS,verbs=VERBS,adjs=ADJS,nseen=len(SEEN),nheld=len(HELD))
    for name,causal in (('both',False),('left',True)):
        P,words,label,core,acc=pretrain(0,causal); n=sum(int(np.prod(t.shape)) for t in core+[P['Wout'],P['bout']]); print(name,acc,n,flush=True)
        out[name]=dict(acc=acc,params=n,**{k:r(v) for k,v in P.items() if k not in('layers','Wc','bc')},layers=[{k:r(v) for k,v in Lr.items()} for Lr in P['layers']])
    json.dump(out,open('page.json','w'),separators=(',',':'))
else:
    res={}
    for n in (8,32,128,512):
        for pre in (True,False):
            accs=[finetune(s,n,pre) for s in range(3)]; res[f"{'pre' if pre else 'scratch'}_{n}"]=accs; print('pretrained' if pre else 'scratch',n,accs,flush=True); json.dump(res,open('finetune.json','w'))
# ---- python3 train.py surprise : use the pretrained model with no new layer at all.
# Hide the place, ask the model how likely the word that was really there is, and call the sentence wrong if that is unlikely.
if sys.argv[1]=='surprise':
    res=json.load(open('finetune.json')); zero=[]
    for n in (8,32,128,512): res[f'surprise_{n}']=[]
    for seed in range(3):
        P,words,label,core,_=pretrain(seed)
        def score(X): Xm=X.copy(); Xm[:,7]=ix['[MASK]']; p=tf.nn.softmax(words(Xm)[:,7]).numpy(); return p[np.arange(len(X)),X[:,7]]
        Xte,Yte=labelled(np.random.default_rng(999),1000,HELD); ste=score(Xte); zero.append(round(float(np.mean((ste>.5)==Yte)),4))
        for n in (8,32,128,512):
            Xtr,Ytr=labelled(np.random.default_rng(100+seed),n,SEEN); s=score(Xtr); th=(s[Ytr==1].min()+s[Ytr==0].max())/2      # the labels only choose where to draw the line
            res[f'surprise_{n}'].append(round(float(np.mean((ste>th)==Yte)),4))
    res['surprise_0']=zero; json.dump(res,open('finetune.json','w')); print({k:v for k,v in res.items() if 'surprise' in k})
