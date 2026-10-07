import optuna
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.interpolate import interp1d
from pycalphad import Database, binplot, calculate, equilibrium, variables as v
from espei.datasets import load_datasets, recursive_glob
from tinydb import where
from tqdm import tqdm
import os
import warnings
import re
from collections import OrderedDict
import concurrent.futures
import multiprocessing

warnings.filterwarnings("ignore")

# ==========================================
# 🎨 1. ARCTDE 学术期刊格式全局配置 (严格对齐)
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 600,
    'savefig.dpi': 600
    
})

def _worker_topology(args):
    params, tdb_file, comps, comp2, test_temperatures, test_x = args
    warnings.filterwarnings("ignore")
    try:
        dbf = Database(tdb_file)
        for sym, val in params.items():
            dbf.symbols[sym] = float(val)
        for t_val in test_temperatures:
            eq = equilibrium(dbf, comps, list(dbf.phases.keys()), {v.P: 101325, v.T: t_val, v.X(comp2): test_x})
            if 'FCC_A1' in list(set(eq.Phase.values.flatten())):
                return (params, False)
        return (params, True)
    except Exception:
        return (params, False)

def extract_ensemble_parameters(db_file, study_name="entropy_opt", min_score=-125000000000):
    if not os.path.exists(db_file):
        print(f"❌ 找不到数据库文件 {db_file}")
        return []

    print(f"▶️ 正在连接数据库 {db_file}...")
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_file}")
    
    ensemble_params = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            score = trial.value 
            if score is not None and score > min_score:
                ensemble_params.append(trial.params)
                
    print(f"✅ 成功提取了 {len(ensemble_params)} 组 Score 高于 {min_score} 的参数集合！")
    
    if len(ensemble_params) > 200:
        indices = np.linspace(0, len(ensemble_params)-1, 200, dtype=int)
        ensemble_params = [ensemble_params[i] for i in indices]
        
    return ensemble_params

def filter_spurious_topologies(tdb_file, ensemble, comp1, comp2, comps):
    print(f"\n▶️ 正在启动【高精度多点拓扑探针】(并行加速模式)...")
    valid_ensemble = []
    test_temperatures = [880, 950, 1050, 1150]
    test_x = 0.5 
    
    args_list = [(params, tdb_file, comps, comp2, test_temperatures, test_x) for params in ensemble]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(tqdm(executor.map(_worker_topology, args_list), total=len(args_list), desc="严格筛查拓扑缺陷"))
        
    for params, is_valid in results:
        if is_valid:
            valid_ensemble.append(params)
            
    removed_count = len(ensemble) - len(valid_ensemble)
    print(f"✅ 深度过滤完成！共剿灭 {removed_count} 组带有虚假共晶线的参数，保留 {len(valid_ensemble)} 组绝对纯净的参数。")
    return valid_ensemble

def plot_pduq_phase_diagram(tdb_file, ensemble, comp1, comp2, comps, phases):
    c2_str = comp2.capitalize()
    
    custom_phase_colors = {
        'HCP_A3': '#135D78',     # Mg 基体
        'DIAMOND_A4': '#243B4A', # Si
        'FCC_A1': '#B70D11',     
        'MG2SI': '#DC6F39',      # 常见中间相
        'LIQUID': '#744162',        
    }
    palette = ['#135D78', '#8E6D59', '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B']
    
    p_idx = 0
    for p in phases:
        if p not in custom_phase_colors and p not in ['VA', '/-']:
            custom_phase_colors[p] = palette[p_idx % len(palette)]
            p_idx += 1

    print(f"\n▶️ 正在计算相边界概率密度图 (PDUQ Phase Diagram)... 此步骤涉及 Matplotlib 轴域渲染，将采用安全串行模式叠加线条。")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    conds = {v.P: 101325, v.T: (300, 1800, 10), v.X(comp2): (0, 1, 0.05)}
    
    tie_line_color = '#EDE7D5'
    default_greens = ['#008000', '#00FF00', '#2CA02C']
    
    # 实例化一个供串行绘图专用的 dbf
    dbf = Database(tdb_file)

    for params in tqdm(ensemble, desc="绘制 Phase Diagram Ensemble (串行叠加)"):
        for sym, val in params.items(): 
            dbf.symbols[sym] = float(val)
        try:
            binplot(dbf, comps, phases, conds, plot_kwargs={'ax': ax, 'linewidth': 1.2})
        except Exception: 
            continue

    color_map = {}
    legend = ax.get_legend()
    if legend:
        handles = getattr(legend, 'legend_handles', getattr(legend, 'legendHandles', []))
        for text, handle in zip(legend.get_texts(), handles):
            label = text.get_text()
            if label in custom_phase_colors:
                try: orig_c = handle.get_facecolor()
                except AttributeError: orig_c = handle.get_color()
                if isinstance(orig_c, np.ndarray) and orig_c.size > 0: orig_c = orig_c[0]
                try:
                    orig_hex = mcolors.to_hex(orig_c, keep_alpha=False).upper()
                    color_map[orig_hex] = custom_phase_colors[label].upper()
                except Exception: pass

    PDUQ_ALPHA = 0.03
    LINE_WIDTH = 1.2

    for line in ax.get_lines():
        try:
            c_hex = mcolors.to_hex(line.get_color(), keep_alpha=False).upper()
            if c_hex in color_map: 
                line.set_color(mcolors.to_rgba(color_map[c_hex], alpha=PDUQ_ALPHA))
                line.set_linewidth(LINE_WIDTH)
            elif c_hex in default_greens: 
                line.set_color(mcolors.to_rgba(tie_line_color, alpha=0.01))
                line.set_linewidth(0.5)
        except Exception: pass

    for collection in ax.collections:
        try:
            edge_colors = collection.get_edgecolors()
            face_colors = collection.get_facecolors()
            c_hex = None
            
            if len(edge_colors) > 0 and tuple(edge_colors[0][:3]) != (0,0,0): 
                c_hex = mcolors.to_hex(edge_colors[0], keep_alpha=False).upper()
            elif len(face_colors) > 0: 
                c_hex = mcolors.to_hex(face_colors[0], keep_alpha=False).upper()
                
            if c_hex:
                if c_hex in color_map: 
                    collection.set_facecolor('none')
                    collection.set_edgecolor(mcolors.to_rgba(color_map[c_hex], alpha=PDUQ_ALPHA))
                    collection.set_linewidth(LINE_WIDTH)
                elif c_hex in default_greens:
                    collection.set_facecolor('none')
                    collection.set_edgecolor(mcolors.to_rgba(tie_line_color, alpha=0.01))
                    collection.set_linewidth(0.5)
        except Exception: pass

    if ax.get_legend() is not None: 
        ax.get_legend().remove()

    ax.set_title("")           
    fig.suptitle("")           
    for txt in list(ax.texts):
        txt.remove()

    legend_elements = [
        mpatches.Patch(facecolor=color, edgecolor='black', linewidth=1.0, label=phase)
        for phase, color in custom_phase_colors.items() if phase in phases
    ]
    # ⭐ 已修改图例位置为 'upper center'
    ax.legend(handles=legend_elements, loc='upper center', frameon=True, edgecolor='black', fontsize=11)

    ax.set_box_aspect(1)
    ax.set_xlabel(fr"Mole Fraction of {c2_str} ($X_{{\mathrm{{{c2_str}}}}}$)", fontsize=14, labelpad=10)
    ax.set_ylabel(r"Temperature (K)", fontsize=14, labelpad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(300, 1800)

    for spine in ax.spines.values(): 
        spine.set_linewidth(1.5)
    ax.tick_params(which='both', direction='in', top=True, right=True, width=1.5, length=6)
    
    plt.tight_layout()
    output_filename = "PDUQ_PhaseDiagram_Uncertainty.png"
    fig.savefig(output_filename, dpi=600, bbox_inches='tight')
    print(f"✅ PDUQ 纯线条叠加相图已保存: {output_filename}")
    plt.close(fig)

# ==========================================
# 🚀 主程序入口
# ==========================================
if __name__ == "__main__":
    multiprocessing.freeze_support()

    TDB_FILE = "B23-MG-SI_Final_Optimized.tdb"
    DB_FILE = "B23-MG-SI_Final_Optimized.db"

    print(f"▶️ 正在初始化系统配置 (基于模型: {TDB_FILE})")
    dbf_init = Database(TDB_FILE)

    active_elements = sorted([e for e in dbf_init.elements if e not in ['VA', '/-']])
    if len(active_elements) != 2:
        print(f"⚠️ 警告: 检测到 {len(active_elements)} 个活性组元，当前脚本专为二元系设计！")

    comp1, comp2 = active_elements[0], active_elements[1]
    comps = [comp1, comp2, 'VA']
    phases = list(dbf_init.phases.keys())

    print(f"✅ 成功解析系统: 目标活度组元 [{comp1}], 变量组元 [{comp2}] | 包含 {len(phases)} 个相")

    raw_ensemble = extract_ensemble_parameters(
        db_file=DB_FILE,
        min_score=-125000000000
    )

    if len(raw_ensemble) > 0:
        clean_ensemble = filter_spurious_topologies(
            TDB_FILE, raw_ensemble, comp1, comp2, comps
        )

        if len(clean_ensemble) > 0:
            plot_pduq_phase_diagram(TDB_FILE, clean_ensemble, comp1, comp2, comps, phases)
            print("\n🎉 PDUQ 相图不确定性量化完成！")
        else:
            print("\n❌ 过滤后没有剩下任何有效的参数组，请考虑放宽 min_score 或重新优化模型。")
    else:
        print("\n❌ 没有找到符合条件的参数，请检查 min_score。")
