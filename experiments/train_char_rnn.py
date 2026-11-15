# Trains the tiny character-level RNN that runs inside rnn.html. Needs only numpy. Writes rnn.json.
import numpy as np, json
rng=np.random.default_rng(0)
S=["the cat sat on the mat. ","the dog sat on the log. "]
chars=sorted(set(''.join(S))); V=len(chars); ix={c:i for i,c in enumerate(chars)}; H=24
def stream(n):
    return ''.join(S[i] for i in rng.integers(0,2,n))
Wxh=rng.normal(0,.3,(H,V)); Whh=rng.normal(0,.3,(H,H)); bh=np.zeros(H); Why=rng.normal(0,.3,(V,H)); by=np.zeros(V)
P=[Wxh,Whh,bh,Why,by]; m=[np.zeros_like(p) for p in P]; v=[np.zeros_like(p) for p in P]
def run(text,h,train=True):
    xs=[ix[c] for c in text[:-1]]; ys=[ix[c] for c in text[1:]]; hs=[h]; ps=[]; loss=0
    for x,y in zip(xs,ys):
        h=np.tanh(Wxh[:,x]+Whh@h+bh); z=Why@h+by; p=np.exp(z-z.max()); p/=p.sum(); hs.append(h); ps.append(p); loss-=np.log(p[y])
    if not train: return loss/len(xs),hs,ps
    g=[np.zeros_like(p) for p in P]; dh=np.zeros(H)
    for t in reversed(range(len(xs))):
        dz=ps[t].copy(); dz[ys[t]]-=1; g[3]+=np.outer(dz,hs[t+1]); g[4]+=dz
        d=(Why.T@dz+dh)*(1-hs[t+1]**2); g[0][:,xs[t]]+=d; g[1]+=np.outer(d,hs[t]); g[2]+=d; dh=Whh.T@d
    return loss/len(xs),hs[-1],g
h=np.zeros(H); hist=[]
for it in range(1,4001):
    text=stream(5)  # 120 chars
    loss,h,g=run(text,np.zeros(H))
    gn=np.sqrt(sum((x**2).sum() for x in g)); 
    if gn>5: g=[x*5/gn for x in g]
    for p,gi,mi,vi in zip(P,g,m,v):
        mi[:]=.9*mi+.1*gi; vi[:]=.999*vi+.001*gi*gi; p-=0.003*(mi/(1-.9**it))/(np.sqrt(vi/(1-.999**it))+1e-8)
    if it in (1,10,30,100,300,1000,2000,4000): hist.append((it,float(np.exp(loss)))); print(it,'perplexity',round(float(np.exp(loss)),3))
# evaluate
test=stream(400); L,hs,ps=run(test,np.zeros(H),False); print('eval perplexity',np.exp(L),'V',V,chars)
for pre in ["the cat sat on the ","the dog sat on the ","the "]:
    _,hs,ps=run(pre+'x' if False else pre+' ',np.zeros(H),False); p=ps[-1]; top=np.argsort(-p)[:3]; print(repr(pre),[(chars[i],round(float(p[i]),3)) for i in top])
# gradient through time: norm of d h_t / d h_{t-k}, averaged over positions in a long run
_,hs,_=run(stream(40),np.zeros(H),False); hs=hs[1:]
K=24; norms=[]
for k in range(K+1):
    vals=[]
    for t in range(60,len(hs)-1,7):
        J=np.eye(H)
        for j in range(k): J=J@(np.diag(1-hs[t-j]**2)@Whh)
        vals.append(np.linalg.norm(J,2))
    norms.append(float(np.mean(vals)))
print('grad norms',[round(x,4) for x in norms]); print('spectral radius Whh',max(abs(np.linalg.eigvals(Whh))))
json.dump(dict(chars=chars,H=H,Wxh=np.round(Wxh,4).tolist(),Whh=np.round(Whh,4).tolist(),bh=np.round(bh,4).tolist(),Why=np.round(Why,4).tolist(),by=np.round(by,4).tolist(),hist=hist,ppl=float(np.exp(L)),norms=norms),open('rnn.json','w'),separators=(',',':'))
