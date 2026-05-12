import re

path = r'C:\Users\A\.gemini\antigravity\scratch\crypto-ai-bot\dashboard\static\index.html'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

new_html = """            <!-- TACTICAL GAUGE -->
            <div id="posGauge" class="mt-4 mb-6 hidden">
              <div class="flex items-center justify-between text-[9px] font-black uppercase tracking-[0.2em] mb-2 px-1">
                <span id="gaugeLabelL" class="text-rose-500 drop-shadow-[0_0_8px_rgba(244,63,94,0.6)]">SL</span>
                <span id="gaugeLabelM" class="text-slate-400 tracking-[0.3em]">Entry</span>
                <span id="gaugeLabelR" class="text-emerald-500 drop-shadow-[0_0_8px_rgba(16,185,129,0.6)]">TP</span>
              </div>
              <div class="relative w-full py-2">
                <div class="absolute inset-y-2 left-0 right-0 h-2 bg-slate-950 rounded-full border border-white/10 shadow-[inset_0_2px_4px_rgba(0,0,0,0.8)] overflow-hidden">
                  <div id="gaugeProgress" class="absolute top-0 bottom-0 transition-all duration-500 ease-out"></div>
                </div>
                <div id="gaugeEntryMark" class="absolute top-1 bottom-1 w-1 bg-slate-500 z-10 -translate-x-1/2 transition-all duration-500 shadow-[0_0_5px_rgba(0,0,0,0.8)]"></div>
                <div id="gaugeMarker" class="absolute top-0.5 w-2 h-5 bg-white rounded-[2px] shadow-[0_0_15px_rgba(255,255,255,1)] z-20 -translate-x-1/2 transition-all duration-300 ease-out border border-slate-300"></div>
              </div>
            </div>"""

new_js = """        // Gauge Update
        const posGauge = q('posGauge');
        if (pos && posGauge) {
            posGauge.classList.remove('hidden');
            const sl = parseFloat(pos.sl), tp = parseFloat(pos.tp_price), cp = curP;
            const entryP = parseFloat(pos.entry);
            const side = String(pos.side).toUpperCase();
            if (sl && tp && cp && entryP) {
                let pct = 0, entryPct = 0;
                if (side === 'LONG') {
                    pct = (cp - sl) / (tp - sl) * 100;
                    entryPct = (entryP - sl) / (tp - sl) * 100;
                } else {
                    pct = (sl - cp) / (sl - tp) * 100;
                    entryPct = (sl - entryP) / (sl - tp) * 100;
                }
                const bounded = Math.max(0, Math.min(100, pct));
                const entryBounded = Math.max(0, Math.min(100, entryPct));
                
                q('gaugeLabelL').textContent = 'SL'; 
                q('gaugeLabelR').textContent = 'TP';
                
                const prog = q('gaugeProgress');
                q('gaugeMarker').style.left = `${bounded}%`;
                if (q('gaugeEntryMark')) q('gaugeEntryMark').style.left = `${entryBounded}%`;
                
                if (bounded >= entryBounded) {
                    prog.style.left = `${entryBounded}%`;
                    prog.style.width = `${bounded - entryBounded}%`;
                    prog.className = 'absolute top-0 bottom-0 bg-gradient-to-r from-emerald-500/20 to-emerald-400 shadow-[0_0_12px_rgba(16,185,129,0.8)] transition-all duration-500 ease-out rounded-r-full';
                } else {
                    prog.style.left = `${bounded}%`;
                    prog.style.width = `${entryBounded - bounded}%`;
                    prog.className = 'absolute top-0 bottom-0 bg-gradient-to-l from-rose-500/20 to-rose-400 shadow-[0_0_12px_rgba(244,63,94,0.8)] transition-all duration-500 ease-out rounded-l-full';
                }
            }
        } else if (posGauge) {
            posGauge.classList.add('hidden');
        }"""

content = re.sub(r'            <!-- TACTICAL GAUGE -->.*?</div>\s*</div>\s*</div>', new_html, content, flags=re.DOTALL)
content = re.sub(r'        // Gauge Update.*?} else if \(posGauge\) {\s*posGauge\.classList\.add\(\'hidden\'\);\s*}', new_js, content, flags=re.DOTALL)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Replaced successfully!')
