# Picks 16 fruit from the simulated line (8 the small network got wrong, 8 it got right), draws each as a large still,
# and writes the true conditions to a separate file that is not shown while the stills are being inspected.
import asyncio,json,numpy as np
from playwright.async_api import async_playwright
R=json.load(open('results.json'))['line']; ln=np.load('line.npz'); Y=ln['Y']; S=ln['S']; pred=np.array(R['pred5']); rng=np.random.default_rng(16)
wrong=list(np.where(pred!=Y)[0][:8]); right=[]
for c,k in ((0,2),(1,2),(2,1),(3,1),(4,2)): right+=list(rng.choice(np.where((pred==Y)&(Y==c))[0],k,replace=False))
pick=[int(i) for i in rng.permutation(wrong+right)]
CONDS=['good','bruised','rotten','underripe','cut']
specs=[dict(seed=(4000+j//500)*100000+j%500,cond=CONDS[Y[j]],size=float(S[j])) for j in pick]
json.dump(dict(pick=pick,specs=specs,small=[CONDS[pred[j]] for j in pick],pgood=[R['pgood5'][j]/255 for j in pick]),open('stills_truth.json','w'))
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(executable_path='/usr/bin/google-chrome',args=['--no-sandbox']); pg=await b.new_page(viewport={'width':1700,'height':900}); await pg.set_content('<html><body style="margin:0"></body></html>'); await pg.add_script_tag(content=open('line_fruit.js').read())
        for half in (0,1):
            await pg.evaluate('''([specs,half])=>{ document.body.innerHTML=''; const c=document.createElement('canvas'); c.width=1680; c.height=840; document.body.append(c); const x=c.getContext('2d');
              specs.slice(half*8,half*8+8).forEach((s,i)=>{ const ox=(i%4)*420, oy=Math.floor(i/4)*420; x.save(); x.beginPath(); x.rect(ox,oy,420,420); x.clip(); x.translate(ox,oy); FRUIT.belt(x,420,420,0); FRUIT.draw(x,210,215,130,s); x.restore();
                x.fillStyle='#fff'; x.font='bold 26px sans-serif'; x.fillText('#'+(half*8+i+1),ox+12,oy+32); x.strokeStyle='#fff'; x.strokeRect(ox+.5,oy+.5,419,419); }); }''',[specs,half])
            await pg.locator('canvas').screenshot(path=f'stills{half}.png')
        await b.close()
asyncio.run(main())
