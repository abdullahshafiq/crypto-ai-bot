from __future__ import annotations

import copy
import csv
import json
import math
import numbers
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml

TRADE_LOG_FILE = "trade_log_futures.csv"


EDITABLE_FIELDS = [
    {"key": "execution.paused", "label": "Pause Bot", "type": "bool"},
    {"key": "execution.mode", "label": "Execution Mode", "type": "select", "options": ["paper", "live"]},
    {"key": "ai.enabled", "label": "Enable AI", "type": "bool"},
    {"key": "execution.market", "label": "Market", "type": "select", "options": ["spot", "usdm"]},
    {"key": "execution.leverage", "label": "Leverage", "type": "number", "step": "1"},
    {"key": "execution.spot_balance_pct", "label": "Spot Balance %", "type": "number", "step": "0.01"},
    {"key": "execution.spot_reserve_pct", "label": "Spot Reserve %", "type": "number", "step": "0.01"},
    {"key": "execution.spot_max_layers", "label": "Spot Max Layers", "type": "number", "step": "1"},
    {"key": "execution.dynamic_leverage", "label": "Dynamic Leverage", "type": "bool"},
    {"key": "execution.use_limit_orders", "label": "Limit Orders", "type": "bool"},
    {"key": "execution.use_native_trailing_stop", "label": "Native Trailing Stop", "type": "bool"},
    {"key": "execution.break_even_trigger_pct", "label": "Break-even Trigger", "type": "number", "step": "0.0001"},
    {"key": "execution.break_even_buffer_pct", "label": "Break-even Buffer", "type": "number", "step": "0.0001"},
    {"key": "execution.trailing_callback_pct", "label": "Trailing Callback %", "type": "number", "step": "0.01"},
    {"key": "execution.min_seconds_between_trades", "label": "Trade Cooldown", "type": "number", "step": "1"},
    {"key": "execution.min_seconds_before_reversal", "label": "Reversal Cooldown", "type": "number", "step": "1"},
    {"key": "strategy.min_conf", "label": "Min Confidence", "type": "number", "step": "0.01"},
    {"key": "strategy.tp_pct", "label": "TP %", "type": "number", "step": "0.0001"},
    {"key": "strategy.sl_pct", "label": "SL %", "type": "number", "step": "0.0001"},
    {"key": "strategy.max_spread", "label": "Max Spread", "type": "number", "step": "0.0001"},
    {"key": "strategy.max_structural_sl_pct", "label": "Max Structural SL %", "type": "number", "step": "0.0001"},
    {"key": "strategy.min_reward_risk", "label": "Min Reward/Risk", "type": "number", "step": "0.01"},
    {"key": "strategy.wick_sweep_enabled", "label": "Wick Sweep", "type": "bool"},
    {"key": "strategy.wick_sweep_buffer_pct", "label": "Wick Buffer", "type": "number", "step": "0.0001"},
    {"key": "strategy.wick_sweep_reclaim_pct", "label": "Wick Reclaim", "type": "number", "step": "0.0001"},
    {"key": "strategy.wick_sweep_wick_ratio", "label": "Wick Ratio", "type": "number", "step": "0.1"},
    {"key": "spot.mode", "label": "Spot Mode", "type": "select", "options": ["grid", "single"]},
    {"key": "spot.max_layers", "label": "Spot Layers", "type": "number", "step": "1"},
    {"key": "spot.layer_quote_pct", "label": "Spot Layer %", "type": "number", "step": "0.01"},
    {"key": "spot.reserve_quote_pct", "label": "Spot Reserve %", "type": "number", "step": "0.01"},
    {"key": "spot.buy_near_support_pct", "label": "Spot Buy Near S", "type": "number", "step": "0.0001"},
    {"key": "spot.sell_near_resistance_pct", "label": "Spot Sell Near R", "type": "number", "step": "0.0001"},
    {"key": "spot.layer_spacing_pct", "label": "Spot Layer Gap", "type": "number", "step": "0.0001"},
    {"key": "spot.emergency_break_pct", "label": "Spot Emergency", "type": "number", "step": "0.0001"},
    {"key": "spot.min_take_profit_pct", "label": "Spot Min TP", "type": "number", "step": "0.0001"},
    {"key": "mtf.min_agree", "label": "MTF Min Agree", "type": "number", "step": "1"},
    {"key": "mtf.veto_on_missing", "label": "MTF Veto On Missing", "type": "bool"},
    {"key": "mtf.sr_buffer_pct", "label": "MTF S/R Buffer", "type": "number", "step": "0.0001"},
    {"key": "risk.max_open_positions", "label": "Max Open Positions", "type": "number", "step": "1"},
    {"key": "risk.daily_loss_cap", "label": "Daily Loss Cap", "type": "number", "step": "0.01"},
    {"key": "risk.disable_loss_cap", "label": "Disable Loss Cap", "type": "bool"},
    {"key": "risk.min_balance_floor", "label": "Min Balance Floor", "type": "number", "step": "0.01"},
]


INDEX_HTML = r"""<!doctype html>
<html lang="en" class="dark">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Quantum Bot | Command Center</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://unpkg.com/lucide@latest"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Geist+Mono:wght@100..900&family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          fontFamily: { 
            sans: ['Inter', 'ui-sans-serif', 'system-ui'],
            mono: ['Geist Mono', 'monospace']
          },
          colors: {
            obsidian: '#020617',
            surface: '#0f172a',
            card: '#0b1120',
            panel: '#0d1526',
            accent: '#10b981',
          }
        }
      }
    }
  </script>
  <style>
    body { background-color: #020617; -webkit-font-smoothing: antialiased; letter-spacing: -0.01em; }
    .terminal-card { 
      background: linear-gradient(145deg, #0b1120 0%, #080d1a 100%);
      border: 1px solid rgba(71, 85, 105, 0.5); 
      box-shadow: 0 10px 30px -10px rgba(0,0,0,0.6);
    }
    .inner-glow { box-shadow: inset 0 1px 1px 0 rgba(255,255,255,0.1); }
    .custom-scrollbar::-webkit-scrollbar { width: 6px; height: 6px; }
    .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
    .custom-scrollbar::-webkit-scrollbar-thumb { background: #334155; border-radius: 10px; }
    #statusLines { scrollbar-width: none; font-variant-ligatures: none; }
    #statusLines::-webkit-scrollbar { display: none; }
    .animate-entry { animation: slideIn 0.4s ease-out forwards; opacity: 0; }
    @keyframes slideIn { from { transform: translateY(10px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
    .settings-drawer {
      transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1);
      transform: translateX(100%);
    }
    .settings-drawer.open { transform: translateX(0); }
    .drawer-overlay {
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.3s ease;
    }
    .drawer-overlay.open { opacity: 1; pointer-events: auto; }
    
    input[type="checkbox"] { filter: invert(100%) hue-rotate(150deg) brightness(1.5); cursor: pointer; }
    .text-glow { text-shadow: 0 0 10px rgba(255,255,255,0.1); }
  </style>
</head>
<body class="text-slate-300 font-sans min-h-screen selection:bg-emerald-500/30 overflow-y-auto">
  <div class="flex flex-col min-h-screen">
    <!-- Polished Header -->
    <header class="flex items-center justify-between px-4 sm:px-8 py-4 sm:py-5 border-b border-slate-700 bg-obsidian sticky top-0 z-50">
      <div class="flex items-center gap-3 sm:gap-6 min-w-0">
        <div class="flex items-center gap-2 sm:gap-3 shrink-0">
          <div class="w-8 h-8 sm:w-10 sm:h-10 bg-emerald-500/20 border border-emerald-500/40 rounded-lg sm:rounded-xl flex items-center justify-center inner-glow shrink-0">
            <i data-lucide="zap" class="w-5 h-5 sm:w-6 sm:h-6 text-emerald-400"></i>
          </div>
          <div class="truncate max-w-[140px] sm:max-w-none">
            <h1 class="text-sm sm:text-xl font-black text-white tracking-tight uppercase leading-none" id="botTitle">Quantum Command</h1>
            <p class="text-[8px] sm:text-xs font-bold text-slate-400 uppercase tracking-widest mt-1 hidden sm:block">Trading Protocol</p>
          </div>
        </div>
        <div class="h-6 w-px bg-slate-700 hidden md:block"></div>
        <div class="hidden md:flex items-center gap-3 shrink-0">
          <span id="modePill" class="text-[10px] font-black px-2 py-1 rounded-md bg-slate-800 border border-slate-600 text-slate-300 uppercase">Initializing</span>
          <span id="regime" class="text-[10px] font-bold text-slate-400 uppercase tracking-widest hidden xl:inline">-</span>
        </div>
      </div>
      <div class="flex items-center gap-2 sm:gap-6 shrink-0">
        <div class="flex items-center gap-2 sm:gap-4">
          <div class="hidden lg:flex flex-col items-end shrink-0">
            <p class="text-[9px] font-black text-slate-400 uppercase tracking-widest leading-none">Kernel Status</p>
            <p class="text-xs font-mono text-emerald-400 mt-1" id="apiStatus">CONNECTED</p>
          </div>
          <div class="relative flex h-2 w-2 sm:h-3 sm:w-3 shrink-0">
            <span id="connPulse" class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span id="connDot" class="relative inline-flex rounded-full h-2 w-2 sm:h-3 sm:w-3 bg-emerald-500 shadow-[0_0_12px_rgba(16,185,129,0.7)]"></span>
          </div>
          <span id="uptime" class="text-[10px] sm:text-sm font-mono font-bold text-emerald-400 uppercase tracking-tighter ml-0.5">UP: 00:00:00</span>
        </div>
        <button id="pauseToggleBtn" class="h-9 sm:h-11 px-3 sm:px-5 rounded-lg sm:rounded-xl bg-slate-900 border border-slate-700 flex items-center gap-2 hover:bg-slate-800 transition-all cursor-pointer">
          <div id="pauseIcon" class="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.6)]"></div>
          <span id="pauseText" class="text-[10px] sm:text-xs font-black text-slate-200 uppercase tracking-widest hidden xs:inline">Bot ON</span>
        </button>
        <button id="openSettings" class="w-9 h-9 sm:w-11 sm:h-11 rounded-lg sm:rounded-xl bg-slate-900 border border-slate-700 flex items-center justify-center hover:bg-slate-800 transition-all">
            <i data-lucide="sliders-vertical" class="w-4 h-4 sm:w-5 sm:h-5 text-slate-300"></i>
        </button>
      </div>
    </header>

    <main class="flex-1 p-6 bg-obsidian">
      <div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-8">
        
        <!-- Column 1: Financial Hub -->
        <div class="flex flex-col gap-8 lg:order-1 order-1">
          <div class="terminal-card rounded-3xl p-7 relative overflow-hidden inner-glow">
            <div class="absolute -top-4 -right-4 opacity-[0.05] rotate-12"><i data-lucide="wallet" class="w-32 h-32 text-white"></i></div>
            <p class="text-xs font-black text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-2">
               <i data-lucide="layout-dashboard" class="w-4 h-4"></i> Total Equity
            </p>
            <div class="flex items-baseline gap-3">
              <span class="text-5xl font-black text-white tracking-tighter" id="balance">0.00</span>
              <span class="text-base text-slate-500 font-mono font-bold" id="quoteAsset">USDC</span>
            </div>
            <div class="mt-6 pt-6 border-t border-slate-700/60 grid grid-cols-2 gap-4">
              <div class="flex flex-col">
                <span class="text-xs text-slate-400 font-black uppercase tracking-wider mb-1">Total Profit</span>
                <span class="text-sm font-black text-emerald-400" id="totalProfit">+$0.00</span>
              </div>
              <div class="flex flex-col items-end">
                <span class="text-xs text-slate-400 font-black uppercase tracking-wider mb-1">Total Loss</span>
                <span class="text-sm font-black text-rose-500" id="totalLoss">-$0.00</span>
              </div>
            </div>
            <div class="mt-4 flex justify-between items-center bg-slate-900/60 p-4 rounded-xl border border-slate-700/60">
              <span class="text-xs text-slate-300 font-black uppercase tracking-wider">Unrealized PnL</span>
              <span class="text-lg font-black tracking-tight" id="pnl">0.00%</span>
            </div>
          </div>

          <div class="terminal-card rounded-3xl p-7 flex-1 flex flex-col inner-glow">
            <div class="flex items-center justify-between mb-6">
              <h3 class="text-xs font-black text-slate-400 uppercase tracking-widest flex items-center gap-2">
                <i data-lucide="history" class="w-5 h-5 text-slate-400"></i> Trade History <span class="text-[9px] lowercase text-slate-500 font-normal tracking-normal">(Current Session)</span>
              </h3>
              <span id="ordersCount" class="text-sm font-mono font-bold text-emerald-400 bg-emerald-500/20 px-3 py-1 rounded-lg border border-emerald-500/40">0</span>
            </div>
            <div id="tradesList" class="flex-1 space-y-3 overflow-y-auto max-h-[500px] pr-1 custom-scrollbar">
              <p class="text-xs text-slate-500 uppercase font-black italic tracking-widest text-center py-12">No history available</p>
            </div>
          </div>
        </div>

        <!-- Column 2: Tactical Intelligence (Priority on Mobile) -->
        <div class="flex flex-col gap-8 lg:order-2 order-2">
          <div class="terminal-card rounded-3xl p-7 relative group overflow-hidden border-emerald-500/40 inner-glow">
            <div class="absolute top-0 left-0 w-2 h-full bg-emerald-500 shadow-[0_0_20px_rgba(16,185,129,0.4)]"></div>
            <div class="flex items-center justify-between mb-8">
              <h3 class="text-xs font-black text-white uppercase tracking-widest flex items-center gap-2.5">
                <i data-lucide="crosshair" class="w-5 h-5 text-emerald-400"></i> Tactical Position
              </h3>
              <span id="posSide" class="text-xs font-black px-4 py-1.5 rounded-xl bg-slate-800 border border-slate-600 text-slate-300 uppercase">-</span>
            </div>
            <div class="space-y-8">
              <div class="grid grid-cols-3 gap-4">
                <div class="space-y-2">
                  <p class="text-sm font-black text-slate-300 uppercase tracking-widest">Entry</p>
                  <p class="text-lg font-mono font-bold text-white tracking-tighter" id="posEntry">-</p>
                </div>
                <div class="space-y-2">
                  <p class="text-sm font-black text-slate-300 uppercase tracking-widest text-rose-400">Risk SL</p>
                  <p class="text-lg font-mono font-bold text-rose-500 tracking-tighter" id="posSl">-</p>
                </div>
                <div class="space-y-2">
                  <p class="text-sm font-black text-slate-300 uppercase tracking-widest text-emerald-400">Target TP</p>
                  <p class="text-lg font-mono font-bold text-emerald-400 tracking-tighter" id="posTp">-</p>
                </div>
              </div>
              <div class="grid grid-cols-2 gap-4 mt-6 pt-6 border-t border-slate-700/60 bg-white/5 p-4 rounded-2xl border border-white/5">
                <div class="flex flex-col">
                  <span class="text-[10px] font-black text-emerald-400 uppercase tracking-widest mb-1">Bull Strike (Resistance - Short Here)</span>
                  <span id="bullStrike" class="text-base font-mono font-bold text-white tracking-tighter">-</span>
                </div>
                <div class="flex flex-col items-end">
                  <span class="text-[10px] font-black text-rose-400 uppercase tracking-widest mb-1">Bear Strike (Support - Long Here)</span>
                  <span id="bearStrike" class="text-base font-mono font-bold text-white tracking-tighter">-</span>
                </div>
              </div>
              <div class="pt-6 mt-2">
                <p class="text-xs font-black text-slate-300 uppercase tracking-[0.2em] mb-4">MTF Structural Data</p>
                <div id="mtfContainer" class="space-y-4 text-sm font-mono">
                </div>
              </div>
            </div>
          </div>

          <div class="terminal-card rounded-3xl p-7 shadow-2xl inner-glow flex-1 flex flex-col min-h-[350px]">
            <div class="flex items-center justify-between mb-8">
              <h3 class="text-xs font-black text-white uppercase tracking-widest flex items-center gap-2.5">
                <i data-lucide="brain" class="w-5 h-5 text-indigo-400"></i> Decision Matrix
              </h3>
              <span id="price" class="text-sm font-mono font-bold text-slate-400">-</span>
            </div>
            <div class="space-y-8 flex-1 flex flex-col">
              <div class="flex items-center justify-between">
                <span class="text-sm font-black text-slate-300 uppercase tracking-widest">Signal Confidence</span>
                <span id="signalText" class="text-base font-black text-white uppercase tracking-tight">-</span>
              </div>
              <div class="w-full bg-slate-900 h-4 rounded-full overflow-hidden inner-glow border border-slate-800">
                <div id="signalConf" class="h-full bg-indigo-500 shadow-[0_0_15px_rgba(99,102,241,0.6)] transition-all duration-1000 w-0"></div>
              </div>
              <div class="flex-1 bg-black/60 rounded-2xl border border-slate-700 p-6 font-mono text-sm sm:text-base text-slate-200 leading-relaxed overflow-y-auto custom-scrollbar" id="aiRationale">
                Awaiting market vector...
              </div>
            </div>
          </div>
        </div>

        <!-- Column 3: Telemetry Kernel -->
        <div class="lg:col-span-1 xl:col-span-1 flex flex-col lg:order-3 order-3">
          <div class="terminal-card rounded-3xl p-0 flex flex-col shadow-2xl overflow-hidden bg-black/20 inner-glow h-full min-h-[500px] max-h-[750px]">
            <div class="flex items-center justify-between px-7 py-6 border-b border-slate-700 bg-obsidian/60">
              <h3 class="text-xs font-black text-slate-300 uppercase tracking-widest flex items-center gap-3">
                <i data-lucide="terminal" class="w-5 h-5"></i> Telemetry Kernel
              </h3>
              <div class="flex items-center gap-6">
                <div id="indicatorsHUD" class="flex gap-4 border-r border-white/10 pr-4"></div>
                <div id="pivotHUD" class="flex gap-4 text-xs font-mono font-bold text-slate-400"></div>
              </div>
            </div>
            <div id="statusLines" class="flex-1 p-7 font-mono text-sm text-slate-200 overflow-y-auto whitespace-pre-wrap leading-relaxed custom-scrollbar bg-black/60 selection:bg-indigo-500/40"></div>
          </div>
        </div>

      </div>
    </main>
  </div>

  <!-- Settings Drawer Overlay -->
  <div id="drawerOverlay" class="drawer-overlay fixed inset-0 bg-obsidian/80 backdrop-blur-sm z-[60]"></div>

  <!-- Settings Drawer -->
  <div id="settingsDrawer" class="settings-drawer fixed top-0 right-0 h-full w-[400px] bg-card border-l border-slate-800 z-[70] shadow-2xl flex flex-col">
      <div class="p-8 border-b border-slate-800 flex items-center justify-between bg-obsidian/40">
          <div>
              <h2 class="text-lg font-black text-white uppercase tracking-tight">System Tuning</h2>
              <p class="text-[10px] font-bold text-slate-600 uppercase tracking-widest mt-1">Parameter Overrides</p>
          </div>
          <button id="closeSettings" class="w-10 h-10 rounded-full hover:bg-white/5 flex items-center justify-center transition-colors">
              <i data-lucide="x" class="w-5 h-5 text-slate-500"></i>
          </button>
      </div>
      
      <div id="settingsForm" class="flex-1 overflow-y-auto p-8 space-y-5 custom-scrollbar bg-black/20">
          <!-- Dynamic Form -->
      </div>
      
      <div class="p-8 border-t border-slate-800 bg-obsidian/40 space-y-4">
          <button id="saveBtn" class="w-full py-5 rounded-2xl bg-emerald-500 text-obsidian text-xs font-black uppercase tracking-[0.2em] transition-all hover:bg-emerald-400 active:scale-95 shadow-[0_10px_20px_-5px_rgba(16,185,129,0.3)]">
              Commit Protocol
          </button>
          <button id="reloadBtn" class="w-full py-4 rounded-2xl text-[10px] font-bold text-slate-600 hover:text-white uppercase tracking-widest transition-colors">
              Abort & Reload
          </button>
          <p id="saveMsg" class="text-[10px] text-center text-emerald-500 font-black uppercase tracking-widest min-h-[1em]"></p>
      </div>
  </div>

  <script>
    const SCHEMA = [];
    let STATE = null;
    let CONFIG = null;
    const fmt = (n, d=2) => {
      const v = Number(n);
      return Number.isFinite(v) ? v.toLocaleString(undefined, {minimumFractionDigits: d, maximumFractionDigits: d}) : '-';
    };
    const q = (id) => document.getElementById(id);
    const safeText = (v) => (v === null || v === undefined || v === '') ? '-' : String(v);

    function setConnection(ok) {
      const dot = q('connDot');
      const pulse = q('connPulse');
      const text = q('apiStatus');
      if (dot) dot.className = ok ? 'relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500' : 'relative inline-flex rounded-full h-2.5 w-2.5 bg-rose-500';
      if (text) text.textContent = ok ? 'CONNECTED' : 'DISCONNECTED';
    }

    function toggleDrawer(open) {
        q('settingsDrawer').classList.toggle('open', open);
        q('drawerOverlay').classList.toggle('open', open);
    }
    q('openSettings').onclick = () => toggleDrawer(true);
    q('closeSettings').onclick = () => toggleDrawer(false);
    q('drawerOverlay').onclick = () => toggleDrawer(false);

    function getNested(obj, path) { return path.split('.').reduce((acc, k) => acc && acc[k], obj); }

    function buildSettingsForm(config) {
      const box = q('settingsForm');
      box.innerHTML = '';
      const sections = [
        { id: 'execution', label: 'Core Execution & Mode' },
        { id: 'strategy', label: 'Strategy Configuration' },
        { id: 'mtf', label: 'Multi-Timeframe Trend' },
        { id: 'risk', label: 'Risk & AI Management' },
        { id: 'spot', label: 'Spot Bot Settings' }
      ];

      sections.forEach(sec => {
        const fields = SCHEMA.filter(f => f.key.startsWith(sec.id) || (sec.id === 'risk' && f.key.startsWith('ai')));
        if (!fields.length) return;

        const secDiv = document.createElement('div');
        secDiv.className = 'mb-6 p-4 bg-slate-900/20 rounded-2xl border border-slate-800/40 select-none';

        const h = document.createElement('h4');
        h.className = 'text-xs font-black text-indigo-400 uppercase tracking-wider mb-4 border-b border-slate-800/40 pb-2 flex items-center justify-between';
        h.innerHTML = `<span>${sec.label}</span><span class="text-[9px] font-mono bg-slate-800 text-slate-400 px-2 py-0.5 rounded-full">${fields.length} fields</span>`;
        secDiv.appendChild(h);

        const fieldsBox = document.createElement('div');
        fieldsBox.className = 'grid grid-cols-1 gap-3';

        fields.forEach(f => {
          const v = getNested(config, f.key);
          const div = document.createElement('div');
          let input;
          if (f.type === 'bool') {
            div.className = 'flex items-center justify-between py-2.5 px-3.5 bg-slate-950/40 hover:bg-slate-900/40 rounded-xl border border-slate-800/40 cursor-pointer h-[50px] transition-colors';
            const label = document.createElement('label');
            label.className = 'text-xs font-semibold text-slate-300 tracking-tight';
            label.textContent = f.label;
            input = document.createElement('input');
            input.type = 'checkbox'; input.className = 'w-4 h-4 accent-emerald-500 rounded border-slate-700 bg-slate-800'; input.checked = !!v;
            div.onclick = (e) => { if(e.target !== input) input.click(); };
            div.appendChild(label); div.appendChild(input);
          } else {
            div.className = 'flex flex-col gap-1 justify-between bg-slate-950/20 p-2.5 rounded-xl border border-slate-800/20 h-[64px]';
            const label = document.createElement('label');
            label.className = 'text-[9px] font-bold text-slate-400 uppercase tracking-wider';
            label.textContent = f.label;
            input = (f.type === 'select') ? document.createElement('select') : document.createElement('input');
            if(f.type === 'select') { 
              f.options.forEach(opt => { const o = document.createElement('option'); o.value=opt; o.textContent=opt; if(String(opt)===String(v)) o.selected=true; input.appendChild(o); }); 
            } else { 
              input.type='number'; input.step=f.step||'0.0001'; input.value=v??''; 
            }
            input.className = 'w-full bg-slate-950/80 border border-slate-800/60 hover:border-slate-700/80 focus:border-indigo-500/50 rounded-lg px-3 py-1.5 text-xs text-white font-mono focus:outline-none transition-all';
            div.appendChild(label); div.appendChild(input);
          }
          input.id = f.key; input.dataset.key = f.key; input.dataset.type = f.type;
          fieldsBox.appendChild(div);
        });

        secDiv.appendChild(fieldsBox);
        box.appendChild(secDiv);
      });
    }

    function renderState(s) {
      if (!s) return;
      try {
        const isPaused = !!(s.config && s.config.execution && s.config.execution.paused);
        q('pauseIcon').className = isPaused ? 'w-2.5 h-2.5 rounded-full bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.4)]' : 'w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.4)]';
        q('pauseText').textContent = isPaused ? 'Bot OFF' : 'Bot ON';
        q('pauseToggleBtn').classList.toggle('border-emerald-500/30', !isPaused);
        q('pauseToggleBtn').classList.toggle('border-rose-500/30', isPaused);

        q('balance').textContent = fmt(s.balance, 2);
        q('regime').textContent = safeText(s.regime);
        q('price').textContent = fmt(s.price, 5);
        const symbolParts = (s.symbol || '').split(':');
        const displaySymbol = symbolParts[0] || 'Quantum';
        q('botTitle').textContent = s.symbol ? `${displaySymbol} COMMAND` : 'Quantum Command';
        
        // Update quote asset label
        const quote = (s.symbol || '').split('/')[1]?.split(':')[0] || 'USDT';
        if (q('quoteAsset')) q('quoteAsset').textContent = quote;
        
        const pnlVal = s.unrealized_pnl_pct || 0;
        const pnlAmt = s.unrealized_pnl || 0;
        q('pnl').textContent = `${pnlVal >= 0 ? '+' : ''}${fmt(pnlVal, 2)}% (${pnlAmt >= 0 ? '+' : ''}$${fmt(pnlAmt, 2)})`;
        q('pnl').className = `text-base font-black tracking-tight ${pnlVal >= 0 ? 'text-emerald-400' : 'text-rose-400'}`;
        
        const mode = safeText(s.mode);
        q('modePill').textContent = mode;
        q('modePill').className = `text-[10px] font-black px-2.5 py-1 rounded-lg border ${mode === 'LIVE' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : 'bg-slate-800 text-slate-500 border-slate-700'}`;
        
        const up = s.uptime_sec || 0;
        const h = Math.floor(up / 3600);
        const m = Math.floor((up % 3600) / 60);
        const sec = Math.floor(up % 60);
        q('uptime').textContent = `UP: ${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
        
        const sig = s.signal || {};
        q('signalText').textContent = sig.action || 'HOLD';
        const conf = (sig.confidence || 0) * 100;
        q('signalConf').style.width = `${conf}%`;
        q('signalConf').className = `h-full transition-all duration-1000 ${sig.action === 'BUY' ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]' : sig.action === 'SELL' ? 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.5)]' : 'bg-slate-700'}`;
        
        const rationale = s.ai_overlay?.rationale || sig.reason || 'Awaiting market vector...';
        const holdReason = sig.hold_reason ? `<div class="mt-4 p-3 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-400 font-black text-xs uppercase animate-pulse">GATE: ${sig.hold_reason}</div>` : '';
        q('aiRationale').innerHTML = `<div>${rationale}</div>${holdReason}`;

        // Strike Zones
        const m5 = s.mtf_context?.['5m'] || {};
        const allLvls = [...(m5.support_levels || []), ...(m5.resistance_levels || [])].map(l => parseFloat(l)).sort((a,b)=>a-b);
        const m5s = allLvls.filter(l => l < s.price).reverse()[0] || null;
        const m5r = allLvls.filter(l => l > s.price)[0] || null;
        
        q('bullStrike').textContent = m5r ? `> $${fmt(m5r, 5)}` : 'WAITING';
        q('bullStrike').className = `text-base font-mono font-bold ${s.price >= m5r && m5r ? 'text-emerald-400 animate-pulse' : 'text-white'}`;
        q('bearStrike').textContent = m5s ? `< $${fmt(m5s, 5)}` : 'WAITING';
        q('bearStrike').className = `text-base font-mono font-bold ${s.price <= m5s && m5s ? 'text-rose-400 animate-pulse' : 'text-white'}`;

        const pos = (s.positions && s.positions[0]) || null;
        q('posSide').textContent = pos ? pos.side : 'FLAT';
        q('posSide').className = `text-[11px] font-black px-3 py-1 rounded-xl border ${pos?.side === 'LONG' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : pos?.side === 'SHORT' ? 'bg-rose-500/10 text-rose-400 border-rose-500/20' : 'bg-slate-800 text-slate-500 border-slate-700'}`;
        q('posEntry').textContent = pos ? fmt(pos.entry, 5) : '-';
        q('posSl').textContent = pos ? fmt(pos.sl, 5) : '-';
        q('posTp').textContent = pos ? fmt(pos.tp_price, 5) : '-';

        // MTF Data
        const mtfBox = q('mtfContainer');
        const mtfCtx = s.mtf_context || {};
        mtfBox.innerHTML = Object.entries(mtfCtx).map(([tf, data]) => `
            <div class="flex items-center justify-between border-b border-white/10 pb-3">
                <span class="text-slate-300 font-bold">${tf}:</span>
                <span class="${String(data.trend).includes('BULL') ? 'text-emerald-400' : 'text-rose-400'} font-black uppercase">${String(data.trend).toUpperCase()}</span>
                <span class="text-xs text-slate-400 font-mono">S:${fmt(data.support_levels?.[0], 4)} R:${fmt(data.resistance_levels?.[0], 4)}</span>
            </div>
        `).join('');

        // Pivots
        const pvt = s.pivot_data?.classic || {};
        q('pivotHUD').innerHTML = `<span>P:${fmt(pvt.pp, 4)}</span> <span class="text-emerald-500">R1:${fmt(pvt.r1, 4)}</span> <span class="text-rose-500">S1:${fmt(pvt.s1, 4)}</span>`;

        // Indicators
        q('indicatorsHUD').innerHTML = `
            <div class="flex flex-col"><span class="text-[10px] text-slate-400 font-black uppercase tracking-tighter">ADX</span><span class="text-xs text-white font-mono font-bold">${fmt(s.latest_indicators?.adx, 1)}</span></div>
            <div class="flex flex-col"><span class="text-[10px] text-slate-400 font-black uppercase tracking-tighter">VWΔ</span><span class="text-xs text-white font-mono font-bold">${fmt(s.latest_indicators?.vwap_dist_pct, 2)}%</span></div>
        `;

        // Logs with coloring
        const logLines = s.status_lines || [];
        q('statusLines').innerHTML = logLines.map(line => {
            let colorClass = 'text-slate-300';
            if (line.includes('Exec:')) colorClass = 'text-emerald-400 font-black';
            if (line.includes('Signal:')) colorClass = 'text-indigo-400 font-black';
            if (line.includes('Reason:')) colorClass = 'text-slate-400 italic';
            if (line.includes('MTF:')) colorClass = 'text-sky-300 font-bold';
            return `<div class="mb-3 py-1.5 border-b border-white/5 ${colorClass}">${line}</div>`;
        }).join('');
        q('statusLines').scrollTop = q('statusLines').scrollHeight;

        // In-session trade count badge (trade panel is populated by refreshTrades())
        if (s.closed_trades && s.closed_trades.length) {
            q('ordersCount').textContent = s.closed_trades.length;
        }

        lucide.createIcons();
      } catch (err) { console.error(err); }
    }

    async function fetchJson(p) { 
      const r = await fetch(p, {cache:'no-store'});
      if(!r.ok) throw new Error(r.status);
      const t = await r.text();
      return JSON.parse(t.replace(/:\s*(NaN|Infinity|-Infinity)/g, ':null'));
    }

    async function load() {
      const [sch, cfg] = await Promise.all([fetchJson('/api/schema'), fetchJson('/api/config')]);
      SCHEMA.splice(0, SCHEMA.length, ...sch); CONFIG = cfg;
      buildSettingsForm(cfg); refresh(); refreshTrades();
    }

    async function refresh() {
      try { STATE = await fetchJson('/api/state'); renderState(STATE); setConnection(true); }
      catch (e) { setConnection(false); }
    }

    async function refreshTrades() {
        try {
        const trades = await fetchJson('/api/trades');
        const tradeBox = q('tradesList');
        
        // Filter trades for current session only
        const sessionStart = STATE?.session_start || 0;
        const filteredTrades = (trades || []).filter(t => {
            if (!t.timestamp) return false;
            // Parse timestamp string e.g. "2026-05-01 12:34:56"
            const tradeDate = new Date(t.timestamp.replace(/-/g, '/')); 
            return (tradeDate.getTime() / 1000) >= sessionStart;
        });

        if (!filteredTrades.length) {
          tradeBox.innerHTML = '<p class="text-xs text-slate-500 uppercase font-black italic tracking-widest text-center py-12">No session history</p>';
          q('ordersCount').textContent = '0';
          q('totalProfit').textContent = '+$0.00';
          q('totalLoss').textContent = '-$0.00';
          return;
        }
        
        q('ordersCount').textContent = filteredTrades.length;
        let totalProfit = 0, totalLoss = 0;
        filteredTrades.forEach(t => { const p = parseFloat(t.pnl||0); if(p>0) totalProfit+=p; else totalLoss+=p; });
        q('totalProfit').textContent = `+$${fmt(totalProfit, 2)}`;
        q('totalLoss').textContent = `-$${fmt(Math.abs(totalLoss), 2)}`;
        
        tradeBox.innerHTML = filteredTrades.slice().reverse().map(t => {
          const pnl = parseFloat(t.pnl || 0);
          const pnlPct = parseFloat(t.pnl_pct || 0);
          const isWin = pnl >= 0;
          const ts = t.timestamp ? String(t.timestamp).substring(11, 19) : '';
          const entry = parseFloat(t.entry || 0);
          const exitP = parseFloat(t.exit || 0);
          const side = String(t.side || '').toUpperCase();
          const label = t.type || t.event || 'TRADE';
          return `
            <div class="p-4 bg-slate-900/50 rounded-xl border-2 ${isWin ? 'border-emerald-500/30' : 'border-rose-500/30'} space-y-2">
              <div class="flex items-center justify-between">
                <div class="flex items-center gap-2">
                  <span class="text-xs font-black px-2 py-1 rounded ${side==='BUY'||label.includes('LONG') ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'}">${side||label}</span>
                  <span class="text-xs text-slate-300 uppercase tracking-wider font-bold">${label}</span>
                </div>
                <span class="text-base font-mono font-black ${isWin ? 'text-emerald-400' : 'text-rose-400'}">${isWin?'+':''}$${fmt(pnl,3)}</span>
              </div>
              <div class="flex items-center justify-between text-xs font-mono">
                <span class="text-slate-400">$${fmt(entry,5)} → $${fmt(exitP,5)}</span>
                <span class="${isWin ? 'text-emerald-500' : 'text-rose-500'} font-bold">${isWin?'+':''}${fmt(pnlPct,2)}%</span>
              </div>
              <div class="text-xs text-slate-500 font-mono">${ts}</div>
            </div>`;
        }).join('');
      } catch(e) { console.warn('Trade history fetch failed:', e); }
    }

    q('saveBtn').onclick = async () => {
      const v = {};
      SCHEMA.forEach(f => { const el = q(f.key); if(!el) return; v[f.key] = f.type === 'bool' ? el.checked : (f.type === 'select' ? el.value : (el.value===''?null:Number(el.value))); });
      const r = await fetch('/api/settings', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({values:v}) });
      const d = await r.json();
      q('saveMsg').textContent = d.ok ? 'SYSTEM CONFIGURED' : 'REJECTED';
      setTimeout(() => { q('saveMsg').textContent = ''; toggleDrawer(false); }, 2000); load();
    };

    q('pauseToggleBtn').onclick = async () => {
      try {
        const r = await fetch('/api/toggle-pause', { method:'POST' });
        const d = await r.json();
        if(d.ok) { refresh(); }
      } catch (err) { console.error(err); }
    };

    load(); setInterval(refresh, 2000); setInterval(refreshTrades, 5000); lucide.createIcons();
  </script>
</body>
</html>
"""


def _deep_get(data: dict, path: str):
    cur = data
    for part in str(path).split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _deep_set(data: dict, path: str, value: Any):
    parts = str(path).split(".")
    cur = data
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        try:
            numeric_value = float(value)
        except Exception:
            return None
        if not math.isfinite(numeric_value):
            return None
        return numeric_value
    if isinstance(value, str) and value.lower() in {"nan", "inf", "-inf", "infinity", "-infinity"}:
        return None
    return value


def _coerce_value(field: dict, raw_value: Any, current_value: Any):
    field_type = str(field.get("type", "")).lower()
    if field_type == "bool":
        if isinstance(raw_value, bool):
            return raw_value
        return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}
    if field_type == "select":
        return str(raw_value)

    if raw_value in ("", None):
        return current_value
    try:
        if field_type == "number":
            if isinstance(current_value, int) and not isinstance(current_value, bool):
                return int(float(raw_value))
            return float(raw_value)
    except Exception:
        return current_value

    if isinstance(current_value, bool):
        return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(float(raw_value))
    if isinstance(current_value, float):
        return float(raw_value)
    return raw_value


class DashboardRuntime:
    def __init__(self, cfg: dict, config_path: str = "config.yaml", overrides_path: str = "ui_state.json"):
        self.cfg = cfg
        self.config_path = Path(config_path)
        self.overrides_path = Path(overrides_path)
        self.ui_overrides: dict = {}
        self.lock = threading.RLock()
        self.state: dict = {}
        self.httpd: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.host: str | None = None
        self.port: int | None = None
        self._last_start_attempt_ts: float = 0.0

    def update_state(self, state: dict):
        with self.lock:
            self.state = copy.deepcopy(state or {})

    def get_state(self) -> dict:
        with self.lock:
            return copy.deepcopy(self.state)

    def get_config(self) -> dict:
        with self.lock:
            return copy.deepcopy(self.cfg)

    def save_overrides(self):
        with self.lock:
            try:
                self.overrides_path.write_text(json.dumps(self.ui_overrides, indent=2), encoding="utf-8")
            except Exception:
                pass

    def apply_settings(self, values: dict) -> dict:
        changed: dict[str, Any] = {}
        with self.lock:
            for field in EDITABLE_FIELDS:
                key = field["key"]
                if key not in values:
                    continue
                current = _deep_get(self.cfg, key)
                coerced = _coerce_value(field, values.get(key), current)
                _deep_set(self.cfg, key, coerced)
                self.ui_overrides[key] = coerced
                changed[key] = coerced
            self.save_overrides()
        return changed

    def start(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = str(host)
        requested_port = int(port)
        last_error: Exception | None = None
        server = None
        for candidate_port in range(requested_port, requested_port + 10):
            try:
                server = ThreadingHTTPServer((self.host, candidate_port), _DashboardHandler)
                self.port = int(candidate_port)
                break
            except OSError as exc:
                last_error = exc
                continue
        if server is None:
            raise last_error or OSError(f"Unable to bind dashboard on {self.host}:{requested_port}")
        server.runtime = self  # type: ignore[attr-defined]
        self.httpd = server
        self.thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.5}, daemon=True)
        self.thread.start()
        self._last_start_attempt_ts = time.time()
        return server

    def ensure_running(self, host: str | None = None, port: int | None = None) -> bool:
        host = str(host or self.host or "127.0.0.1")
        port = int(port or self.port or 8765)
        if self.thread is not None and self.thread.is_alive() and self.httpd is not None:
            return True
        now = time.time()
        if (now - float(getattr(self, "_last_start_attempt_ts", 0.0) or 0.0)) < 10.0:
            return False
        try:
            self.start(host=host, port=port)
            return True
        except Exception:
            self._last_start_attempt_ts = now
            return False


class _DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    @property
    def runtime(self) -> DashboardRuntime:
        return getattr(self.server, "runtime")  # type: ignore[no-any-return]

    def _send(self, payload: Any, status: int = 200, content_type: str = "application/json"):
        body = payload if isinstance(payload, (bytes, bytearray)) else str(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in {"/", "/index.html"}:
            self._send(INDEX_HTML, content_type="text/html; charset=utf-8")
            return
        if self.path == "/api/state":
            self._send(json.dumps(_json_safe(self.runtime.get_state()), default=str, allow_nan=False).encode("utf-8"))
            return
        if self.path == "/api/config":
            self._send(json.dumps(_json_safe(self.runtime.get_config()), default=str, allow_nan=False).encode("utf-8"))
            return
        if self.path == "/api/schema":
            self._send(json.dumps(_json_safe(EDITABLE_FIELDS), allow_nan=False).encode("utf-8"))
            return
        if self.path == "/api/trades":
            try:
                state = self.runtime.get_state()
                trades = state.get("closed_trades", [])
                self._send(json.dumps(_json_safe(trades), allow_nan=False).encode("utf-8"))
                return
            except Exception as e:
                self._send(json.dumps({"error": str(e)}), status=500)
                return
                self._send(json.dumps(_json_safe(trades), allow_nan=False).encode("utf-8"))
            except Exception as exc:
                self._send(json.dumps([]).encode("utf-8"))
            return
        self._send(json.dumps({"error": "not found"}).encode("utf-8"), status=404)

    def do_POST(self):
        if self.path == "/api/toggle-pause":
            try:
                with self.runtime.lock:
                    is_paused = not bool(self.runtime.cfg.get("execution", {}).get("paused", False))
                    if "execution" not in self.runtime.cfg:
                        self.runtime.cfg["execution"] = {}
                    self.runtime.cfg["execution"]["paused"] = is_paused
                    self.runtime.save_overrides()
                self._send(json.dumps({"ok": True, "paused": is_paused}).encode("utf-8"))
            except Exception as exc:
                self._send(json.dumps({"ok": False, "error": str(exc)}).encode("utf-8"), status=500)
            return

        if self.path != "/api/settings":
            self._send(json.dumps({"error": "not found"}).encode("utf-8"), status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            values = payload.get("values", {}) if isinstance(payload, dict) else {}
            changed = self.runtime.apply_settings(values if isinstance(values, dict) else {})
            self._send(json.dumps({"ok": True, "changed": changed}).encode("utf-8"))
        except Exception as exc:
            self._send(json.dumps({"ok": False, "error": str(exc)}).encode("utf-8"), status=500)


def start_dashboard_server(runtime: DashboardRuntime, host: str = "127.0.0.1", port: int = 8765):
    return runtime.start(host=host, port=port)
