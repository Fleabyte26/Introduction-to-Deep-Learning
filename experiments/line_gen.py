# Renders the drawn fruit in headless Chrome and saves the pictures the small network trains and tests on.
import asyncio,base64,json,numpy as np
from playwright.async_api import async_playwright
JS=open('line_fruit.js').read()
GEN='''(a)=>{ const [n,seed0,mode,blur,frames]=a; const r=FRUIT.rng(seed0), PRI=[.70,.80,.86,.94,1.0]; const out=new Uint8Array(n*frames*3072), lab=[], sz=[];
 for(let i=0;i<n;i++){ let c; if(mode==='balanced') c=i%5; else { const u=r(); c=PRI.findIndex(p=>u<p); }
  const s={seed:seed0*100000+i, cond:FRUIT.CONDS[c], size:.72+r()*.58}; lab.push(c); sz.push(s.size);
  for(let f=0;f<frames;f++){ const im=FRUIT.crop(s,{blur:blur<0? r()*1.2 : blur, dx:(r()-.5)*8, dy:(r()-.5)*8, bright:.85+r()*.3, nseed:f+1}); const o=(i*frames+f)*3072; for(let k=0;k<3072;k++) out[o+k]=im[k]*255; } }
 let bin=''; for(let i=0;i<out.length;i+=32768) bin+=String.fromCharCode.apply(null,out.subarray(i,i+32768)); return [btoa(bin),lab,sz]; }'''
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(executable_path='/usr/bin/google-chrome',args=['--no-sandbox']); pg=await b.new_page(); await pg.set_content('<html><body></body></html>'); await pg.add_script_tag(content=JS)
        async def make(name,n,seed,mode,blur=-1,frames=1,chunk=1000):
            X=[];Y=[];S=[]
            for c in range(0,n,chunk):
                m=min(chunk,n-c); d,lab,sz=await pg.evaluate(GEN,[m,seed*1000+c//chunk,mode,blur,frames]); X.append(np.frombuffer(base64.b64decode(d),np.uint8).reshape(m,frames,32,32,3)); Y+=lab; S+=sz
            np.savez_compressed(name,X=np.concatenate(X),Y=np.array(Y,'int32'),S=np.array(S,'float32')); print(name,n,flush=True)
        await make('train',6000,1,'balanced'); await make('test',2000,2,'balanced')
        for i,bl in enumerate([0,0.3,1.4,2.8,5.6,10.5]): await make(f'blur{i}',1000,3,'balanced',blur=bl)
        await make('line',3000,4,'line',blur=0.3,frames=9,chunk=500)
        await pg.evaluate('''()=>{ const c=document.createElement('canvas'); c.width=900; c.height=380; document.body.append(c); const x=c.getContext('2d'); FRUIT.belt(x,900,380,0); FRUIT.CONDS.forEach((cd,i)=>{ for(let j=0;j<2;j++) FRUIT.draw(x,90+i*180,100+j*185,62,{seed:500+i*7+j,cond:cd,size:j?1.25:.8}); }); }''')
        await pg.locator('canvas').screenshot(path='sample.png'); await b.close()
asyncio.run(main())
