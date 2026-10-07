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

def _worker_activity(args):
    params, tdb_file, comps, phase_name, T_calc, comp1, comp2, x_grid = args
    warnings.filterwarnings("ignore")
    try:
        dbf = Database(tdb_file)
        for sym, val in params.items():
            dbf.symbols[sym] = float(val)

        eq_pure = equilibrium(dbf, comps, [phase_name], {v.P: 101325, v.T: T_calc, v.X(comp2): 1e-9})
        mu_pure = float(eq_pure.MU.sel(component=comp1).values.flatten()[0])

        eq_curve = equilibrium(dbf, comps, [phase_name], {v.P: 101325, v.T: T_calc, v.X(comp2): x_grid})
        x_calc = eq_curve.X.sel(component=comp2, vertex=0).values.flatten()
        mu_calc = eq_curve.MU.sel(component=comp1).values.flatten()

        valid = ~np.isnan(mu_calc)
        if not np.any(valid): return None

        act_calc = np.exp((mu_calc[valid] - mu_pure) / (8.3145 * T_calc))
        x_calc_valid = x_calc[valid]

        sort_idx = np.argsort(x_calc_valid)
        f_interp = interp1d(x_calc_valid[sort_idx], act_calc[sort_idx], kind='linear', bounds_error=False, fill_value="extrapolate")
        return f_interp(x_grid)
    except Exception:
        return None

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

def plot_pduq_activity(tdb_file, ensemble, comp1, comp2, comps, phase_name='LIQUID', data_path="input-data", T_calc=1350):
    c1_str = comp1.capitalize()
    c2_str = comp2.capitalize()
    print(f"\n▶️ 正在并行计算 {phase_name} 相 {c1_str} 活度的 95% 置信区间 (T={T_calc}K)...")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    
    color_c1 = '#B70D11' # 红色系
    exp_data_list = []
    
    try:
        datasets = load_datasets(recursive_glob(data_path))
        # 只搜索 Comp1 (Al) 的活度数据
        pattern = fr'ACR_{comp1}|ACTIVITY_{comp1}'
        exp_matches = datasets.search(
            where('output').test(lambda x: bool(re.search(pattern, str(x), re.IGNORECASE)))
        )
        for d in exp_matches:
            ref = d.get('reference', 'Unknown')
            conds = d['conditions']
            data_t = np.mean(conds.get('T'))
            if abs(data_t - T_calc) > 50: continue 
            
            # 严格核对 output 确认是目标组元
            out_str = d.get('output', '').upper()
            if comp1 not in out_str: continue
                
            if f'X_{comp1}' in conds: 
                ex = 1 - np.array(conds[f'X_{comp1}']).flatten()
            elif f'X_{comp2}' in conds:
                ex = np.array(conds[f'X_{comp2}']).flatten()
            else: continue
                
            ey = np.array(d['values']).flatten()
            exp_data_list.append({'ref': ref, 'x': ex, 'y': ey, 'target': comp1})
    except Exception as e:
        print(f"⚠️ 读取实验数据失败或未找到: {e}")

    x_grid = np.linspace(0.01, 0.99, 100) 
    
    # 启动多进程计算活度曲线
    args_list = [(params, tdb_file, comps, phase_name, T_calc, comp1, comp2, x_grid) for params in ensemble]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(tqdm(executor.map(_worker_activity, args_list), total=len(args_list), desc=f"计算 {c1_str} Activity"))
    
    all_act = [res for res in results if res is not None]

    if all_act:
        all_act = np.array(all_act)
        act_mean = np.mean(all_act, axis=0)
        ax.fill_between(x_grid, np.percentile(all_act, 2.5, axis=0), np.percentile(all_act, 97.5, axis=0), color=color_c1, alpha=0.35, label='95% Confidence Interval')
        ax.plot(x_grid, act_mean, color=color_c1, linewidth=2.5, label=f'Calculated (Ensemble Mean) $a_{{{c1_str}}}$')

    # 绘制实验点
    color_cycle = plt.cm.tab10.colors
    marker_cycle = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
    ref_styles, m_idx = {}, 0
    
    for item in exp_data_list:
        target_str = item['target'].capitalize()
        label = f"{item['ref']} ($a_{{{target_str}}}$)"
        
        if label not in ref_styles:
            ref_styles[label] = {
                'color': color_cycle[m_idx % len(color_cycle)],
                'marker': marker_cycle[m_idx % len(marker_cycle)]
            }
            m_idx += 1
            
        style = ref_styles[label]
        ax.scatter(item['x'], item['y'], label=label, edgecolors=style['color'], facecolors='none', 
                   s=70, marker=style['marker'], linewidths=1.5, zorder=10)

    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict()
    for handle, label in zip(handles, labels):
        if label not in by_label: by_label[label] = handle
    
    exp_handles, exp_labels = [], []
    calc_handles, calc_labels = [], []
    for l, h in by_label.items():
        if 'Calculated' in l or '95%' in l:
            calc_handles.append(h); calc_labels.append(l)
        else:
            exp_handles.append(h); exp_labels.append(l)
            
    if (exp_handles + calc_handles):
        ax.legend(exp_handles + calc_handles, exp_labels + calc_labels, loc='upper center', 
                  frameon=True, edgecolor='black', 
                  facecolor='#F5F5F7', fontsize=11, ncol=1, labelspacing=0.4, borderpad=0.5)
    
    ax.set_box_aspect(1)
    ax.set_xlabel(fr"Mole Fraction of {c2_str} ($X_{{\mathrm{{{c2_str}}}}}$)", fontsize=14, labelpad=10)
    ax.set_ylabel(f"Activity of {c1_str} ($a_{{\mathrm{{{c1_str}}}}}$)", fontsize=14, labelpad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for spine in ax.spines.values(): spine.set_linewidth(1.5)
    ax.tick_params(which='both', direction='in', top=True, right=True, width=1.5, length=6)
    
    plt.tight_layout()
    fig.savefig(f"PDUQ_Activity_{c1_str}_{T_calc}K.png", dpi=600, bbox_inches='tight')
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
            plot_pduq_activity(TDB_FILE, clean_ensemble, comp1, comp2, comps, phase_name='LIQUID', data_path="input-data", T_calc=1350)
            print("\n🎉 PDUQ 活度不确定性量化完成！")
        else:
            print("\n❌ 过滤后没有剩下任何有效的参数组，请考虑放宽 min_score 或重新优化模型。")
    else:
        print("\n❌ 没有找到符合条件的参数，请检查 min_score。")
