"""Test re-snipe path through Phase 13b after patch."""
import numpy as np, pandas as pd, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['NO_UI'] = '1'

import indicators.signals.engine as eng

# Patch gate functions with tracing
_patches = []
for name, lbl in [('apply_range_reversal_sniper','Ph13'),
                   ('apply_midrange_policy','Ph14'),
                   ('_apply_rejection_confirmation_gate','Ph15'),
                   ('apply_exhaustion_divergence_gate','Ph16'),
                   ('apply_wall_rejection_rescue','Ph17')]:
    orig = getattr(eng, name)
    def _make_tracer(lbl, orig):
        def tracer(*args, **kwargs):
            ctx = args[0] if args and isinstance(args[0], dict) else {}
            sig = ctx.get('signal', {}) if isinstance(ctx, dict) else {}
            a = sig.get('action','?')
            _patches.append((f'{lbl} IN', a))
            try:
                if lbl == 'Ph15':
                    r = orig(*args, **kwargs)
                    oa = r[0].get('action','?') if r and isinstance(r, tuple) else '?'
                else:
                    orig(*args, **kwargs)
                    oa = ctx.get('signal',{}).get('action','?')
                _patches.append((f'{lbl} OUT', oa))
                return r if lbl == 'Ph15' else None
            except Exception as e:
                _patches.append((f'{lbl} ERR', str(e)[:30]))
                raise
        return tracer
    setattr(eng, name, _make_tracer(lbl, orig))

from indicators.signals.engine import generate_quant_signal

def mk_li(price, rsi, md, psar, **kw):
    return {'close':price,'ema_9':price*0.998,'ema_21':price*0.997,
            'rsi_14':rsi,'rsi':rsi,'adx':28,'atr_pct':0.5,
            'macd':0.001,'macd_diff':md,'psar':psar,'psar_streak':3,
            'vwap':100,'bb_low':90,'bb_high':110,'bb_mid':100,
            'obv':0,'obv_ema':0,'j':50,'z_score':0,'trend_bias':0,**kw}

cfg = {'range_action_zone_pct':0.20,'macd_noise_threshold':5e-5,
       'max_structural_sl_pct':0.012,'min_reward_risk':0.65,
       'sl_pct':0.0025,'tp_pct':0.006,'session_filter_enabled':False,
       'spread_max_pct':0.005,'low_vol_min_score':0.45,
       'chase_max_dist_pct':0.003,'chase_near_extreme_pct':0.005,
       'midrange_min_score':0.28,'session_block_min_score':0.35,
       'range_reversal_min_depth':0.20,'range_reversal_max_boost':0.45,
       'rsi_ob_entry_gate':72,'rsi_os_entry_gate':28,
       'range_position_veto_enabled':True,
       'range_veto_top_pct':0.75,'range_veto_bottom_pct':0.25,'range_veto_escape_score':0.90}
mtf_cfg = {'enabled':True,'timeframes':['1m','3m','5m','10m','15m']}
mtf_ctx = {'15m':{'macd':0.001,'structure':'NEUTRAL','rsi':50},'10m':{'macd':0.001,'rsi':50},
            '5m':{'macd':0.001,'rsi':50},'3m':{'macd':0.001,'rsi':50}}
pivot = {'classic':{'s1':91,'s2':89,'s3':87,'r1':109,'r2':111,'r3':113,'pp':100}}

def mk_df(price, lo=90, hi=110, side='ceiling'):
    n=100; s=np.random.seed(42)
    c = np.linspace(lo, hi, n) + np.random.randn(n)*0.3
    c[-1] = price
    hi_ext = hi * 1.005; lo_ext = lo * 0.995
    h = c + np.abs(np.random.randn(n))*0.3
    l = c - np.abs(np.random.randn(n))*0.3
    h[-2] = hi_ext; h[-3] = hi_ext
    l[-2] = lo_ext; l[-3] = lo_ext
    rsi = np.full(n, 65 if 'ceiling' in side else 35)
    md = np.full(n, 0.0001 if 'ceiling' in side else -0.0001)
    ps = np.full(n, price*1.02 if 'ceiling' in side else price*0.98)
    d = pd.DataFrame({'close':c,'open':c+np.random.randn(n)*0.1,'high':h,'low':l,
        'volume':np.random.randint(100,10000,n).astype(float),
        'ema_9':pd.Series(c).rolling(9,1).mean(),'ema_21':pd.Series(c).rolling(21,1).mean(),
        'rsi_14':rsi,'rsi':rsi,'adx':np.full(n,28.),'atr_pct':np.full(n,.5),'atr':np.full(n,.5),
        'macd':np.zeros(n),'macd_diff':md,'psar':ps,'psar_streak':np.full(n,3),
        'bb_low':np.full(n,lo),'bb_high':np.full(n,hi),'bb_mid':np.full(n,(lo+hi)/2),
        'obv':np.zeros(n),'obv_ema':np.zeros(n),'j':np.full(n,50.),'z_score':np.zeros(n),
        'vwap':np.full(n,(lo+hi)/2),'trend_bias':np.zeros(n),
        'stoch_k':np.full(n,50.),'stoch_d':np.full(n,50.),'bb_width':np.full(n,hi-lo),
        'session_open':np.full(n,100.),'previous_close':np.full(n,price),
        'previous_high':np.full(n,hi),'previous_low':np.full(n,lo),
        'ema_200':np.full(n,100.),'ema_200_pct_dist':np.full(n,0.)},
        index=pd.date_range('2025-01-01',periods=n,freq='1min'))
    return d

scenarios = [
    ("S1 BUY@ceil -> sniper SELL", 110.0, 'ceiling', 65, 0.0001, 112.0),
    ("S2 SELL@floor -> sniper BUY", 90.0, 'floor', 35, -0.0001, 88.0),
    ("S3 mid range", 100.0, 'mid', 50, 0.0, 99.5),
]

for label, price, side, rsi, md, psar in scenarios:
    _patches.clear()
    print(f"\n=== {label} ===")
    r = generate_quant_signal(
        state={'price':price,'spread_pct':0.0002,'bid':price*0.999,'ask':price*1.001},
        latest_indicators=mk_li(price,rsi,md,psar), strategy_config=cfg,
        df_indicators=mk_df(price,90,110,side),
        latest_macro={'regime':'NEUTRAL'}, mtf_context=mtf_ctx,
        mtf_config=mtf_cfg, pivot_data=pivot)
    tr = ' -> '.join(f'{p[0]}({p[1]})' for p in _patches)
    print(f"  trace: {tr}")
    print(f"  act={r['action']} hold_reason={str(r.get('hold_reason',''))[:70]}")
    # Check if re-snipe fired
    snipecnt = sum(1 for p in _patches if p[0]=='Ph13 IN')
    print(f"  Ph13 invocations: {snipecnt}")
