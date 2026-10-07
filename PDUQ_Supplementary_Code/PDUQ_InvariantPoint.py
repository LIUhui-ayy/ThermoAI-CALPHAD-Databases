import os
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 🚀 0. 核心配置：强制无界面渲染
# ==========================================
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

# ==========================================
# 📚 依赖库导入
# ==========================================
import optuna
import numpy as np
import seaborn as sns
import matplotlib.lines as mlines
from pycalphad import Database, equilibrium, variables as v
from espei.datasets import load_datasets, recursive_glob
from espei.core_utils import ravel_zpf_values
from tinydb import where
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ==========================================
# 🎨 1. 顶级期刊纯净美学配置 (统一格式)
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

# 🌟 高级冷暖对比配色方案
THEME_COLORS = {
    'scatter': '#135D78', # 翠蓝色 (深海冷色，高辨识度)
    'ci_68': '#B70D11',   # 宫墙朱砂 (68% 核心区实线)
    'ci_95': '#DC6F39',   # 琥珀橙 (95% 边缘区虚线)
}

# 🌟 10 种高对比度实验数据样式
EXP_STYLES = [
    ('#FFC107', '*'), ('#4CAF50', 's'), ('#9C27B0', 'D'), ('#E91E63', '^'),
    ('#00BCD4', 'v'), ('#FF9800', 'p'), ('#3F51B5', 'h'), ('#8BC34A', '8'),
    ('#F44336', 'X'), ('#795548', 'd')
]

# ==========================================
# 🔍 2. 自动扫描 ZPF 实验数据 (单组元稳定+雷达诊断版)
# ==========================================
def extract_zpf_in_box(data_path, comp_x, comps, x_range, t_range):
    print(f"\n▶️ 正在扫描 {data_path} 目录，提取边界范围内的 ZPF 实验数据...")
    x_min, x_max = x_range[0], x_range[1]
    t_min, t_max = t_range[0], t_range[1]
    print(f"   --> 设定的探测边界: X = {x_min}~{x_max}, T = {t_min}~{t_max} K")

    extracted_points = []
    all_parsed_points = []

    try:
        datasets = load_datasets(recursive_glob(data_path))
        if len(datasets.all()) == 0:
            print("⚠️ 警告：未在目录中找到任何 JSON 实验数据。")
            return []

        zpf_data = datasets.search(
            (where('output') == 'ZPF') & 
            (where('components').test(lambda x: set(x).issubset(comps)))
        )
        if not zpf_data:
            return []

        # 严格遵守物理规律：二元系只请求 1 个独立组元
        eq_dict = ravel_zpf_values(zpf_data, [comp_x], conditions={'P': 101325})
        
        for num_phases, equilibria in eq_dict.items():
            for eq in equilibria:
                for phase_record in eq:
                    comp_dict = phase_record[1] if len(phase_record) > 1 else {}
                    ref_key = str(phase_record[2]) if len(phase_record) > 2 else "Unknown"

                    if not comp_dict:
                        continue

                    t_v = comp_dict.get('T')
                    x_v = comp_dict.get(comp_x)

                    if t_v is not None and x_v is not None:
                        all_parsed_points.append((float(x_v), float(t_v), ref_key))
                        if (x_min <= float(x_v) <= x_max) and (t_min <= float(t_v) <= t_max):
                            extracted_points.append((float(x_v), float(t_v), ref_key))

    except Exception as e:
        print(f"❌ 提取 ZPF 数据时发生错误: {e}")

    unique_points = list(set(extracted_points))
    
    # 智能诊断雷达
    if len(unique_points) == 0 and len(all_parsed_points) > 0:
        print(f"⚠️ 提示：在全图中解析到了 {len(all_parsed_points)} 个实验点，但全都不在您的设定边界内！")
        near_points = [p for p in all_parsed_points if (t_min - 50) <= p[1] <= (t_max + 50)]
        near_points.sort(key=lambda x: x[0])
        
        if near_points:
            print(f"   --> 🔍 雷达扫描：在温度 {t_min-50}K ~ {t_max+50}K 附近，找到以下真实存在的实验点：")
            for p in near_points[:10]: 
                print(f"       👀 文献 [{p[2]}]: T = {p[1]:.1f} K, X_{comp_x} = {p[0]:.4f}")
        else:
            print("   --> 🔍 雷达扫描：在目标温度区间附近，真的没有任何实验点。")

    print(f"✅ 成功提取到 {len(unique_points)} 个符合边界条件的实验点！")
    return unique_points

# ==========================================
# 🔍 3. 提取参数集合
# ==========================================
def extract_ensemble_parameters(db_file="B23-MG-SI_Final_Optimized.db", study_name="entropy_opt", min_score=-125000000000, n_samples=200):
    if not os.path.exists(db_file): 
        return []
    print(f"▶️ 正在连接数据库 {db_file} 提取参数...")
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_file}")
    ensemble_params = [t.params for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and t.value > min_score]
    
    if len(ensemble_params) > n_samples:
        indices = np.linspace(0, len(ensemble_params)-1, n_samples, dtype=int)
        ensemble_params = [ensemble_params[i] for i in indices]
        
    print(f"✅ 成功提取了 {len(ensemble_params)} 组优化参数。")
    return ensemble_params

# ==========================================
# ⚡ 4. 并行计算核心 Worker 函数 (物理逻辑固相修正版)
# ==========================================
def _worker_calc_invariant(args):
    params, dbf, comps, target_phases, search_conds = args
    
    # 🎯 设定您真正想要提取成分的目标相：最大固溶度边界相
    TARGET_EXTRACT_PHASE = 'LIQUID'  # 这里可以根据需要修改为 'FCC_A1' 或其他固相名称
    
    for sym, val in params.items(): 
        dbf.symbols[sym] = float(val)
        
    try:
        eq = equilibrium(dbf, comps, target_phases, search_conds)
        phase_array = eq.Phase.values
        
        # 1. 依然利用 LIQUID 的最低存在温度来判定不变反应线 (共晶温度)
        if 'LIQUID' in phase_array:
            liquid_indices = np.argwhere(phase_array == 'LIQUID')
            t_dim_idx = eq.Phase.dims.index('T')
            min_t_idx = np.min(liquid_indices[:, t_dim_idx])
            T_e = float(eq.T.values[min_t_idx])
            
            # 获取该最低温度对应的系统状态的索引
            min_t_records = liquid_indices[liquid_indices[:, t_dim_idx] == min_t_idx]
            first_record = min_t_records[0] 
            
            # 提取系统坐标索引 (去掉最后表示内部 Vertex 的一维)
            coord_indices = tuple(first_record[:-1])
            
            # 2. 核心修正：在该平衡坐标点上寻找共存的 FCC_A1 固相
            phases_at_point = phase_array[coord_indices]
            
            if TARGET_EXTRACT_PHASE in phases_at_point:
                # 找到 FCC_A1 对应的 Vertex
                v_idx_solid = np.where(phases_at_point == TARGET_EXTRACT_PHASE)[0][0]
                
                # 构建用于提取成分的字典，将 Vertex 指向固相
                isel_dict = {dim: first_record[i] for i, dim in enumerate(eq.Phase.dims[:-1])}
                isel_dict[eq.Phase.dims[-1]] = v_idx_solid 
                
                if 'X_SI' in eq.data_vars:
                    X_e = float(eq.X_SI.isel(**isel_dict).values)
                else:
                    X_e = float(eq.X.sel(component='SI').isel(**isel_dict).values)
                
                return [X_e, T_e]
    except Exception:
        return None
    return None

# ==========================================
# 📊 5. 2D 网格并行平衡计算与可视化渲染
# ==========================================
def plot_pduq_invariant_point(dbf, ensemble, comps, target_phases, search_conds, exp_data=None):
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"\n▶️ 启动多进程并行计算 2D 平衡相图 (调用核心数: {max_workers})...")
    
    fig, ax = plt.subplots(figsize=(6.5, 6)) 
    
    task_args = [(params, dbf, comps, target_phases, search_conds) for params in ensemble]
    invariant_points = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker_calc_invariant, arg): arg for arg in task_args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="🚀 并行处理进度"):
            result = future.result()
            if result is not None:
                invariant_points.append(result)
            
    points = np.array(invariant_points)
    
    if len(points) < 5:
        print("\n❌ 提取失败。请检查 search_conds 区间或参数。")
        plt.close(fig)
        return
        
    print(f"\n✅ 成功提取 {len(points)} 个 LIQUID 液相边界点样本！")

    # ==========================================
    # 🌟 计算并输出与实验值的定量偏差 (RMSE)
    # ==========================================
    if exp_data and len(exp_data) > 0:
        print("\n" + "="*55)
        print("📊 预测值与实验值偏差分析 (PDUQ RMSE)")
        print("="*55)
        mean_X = np.mean(points[:, 0])
        mean_T = np.mean(points[:, 1])
        print(f"🔮 模型预测均值: X = {mean_X:.4f}, T = {mean_T:.2f} K")
        
        for idx, (ex, et, ref) in enumerate(exp_data):
            rmse_T = np.sqrt(np.mean((points[:, 1] - et)**2))
            rmse_X = np.sqrt(np.mean((points[:, 0] - ex)**2))
            print(f"🎯 实验点 {idx+1} [{ref}]: X = {ex:.4f}, T = {et:.2f} K  (ΔT={rmse_T:.2f}K, ΔX={rmse_X:.4f})")
        print("="*55 + "\n")

    # ==========================================
    # 🌟 绘制概率密度等高线和散点
    # ==========================================
    sns.kdeplot(x=points[:, 0], y=points[:, 1], ax=ax, levels=[0.05], color=THEME_COLORS['ci_95'], 
                linestyles="dashed", linewidths=2.0, fill=False, bw_adjust=1.5) 
                
    sns.kdeplot(x=points[:, 0], y=points[:, 1], ax=ax, levels=[0.32], color=THEME_COLORS['ci_68'], 
                linestyles="solid", linewidths=2.5, fill=False, bw_adjust=1.5) 
                
    ax.scatter(points[:, 0], points[:, 1], color=THEME_COLORS['scatter'], s=25, alpha=0.85, edgecolors='none', zorder=10)

    # ==========================================
    # 🌟 叠加实验数据
    # ==========================================
    exp_legend_handles = []
    if exp_data and len(exp_data) > 0:
        exp_groups = {}
        for ex, et, ref in exp_data:
            if ref not in exp_groups:
                exp_groups[ref] = {'x': [], 't': []}
            exp_groups[ref]['x'].append(ex)
            exp_groups[ref]['t'].append(et)
            
        for idx, (ref, data) in enumerate(exp_groups.items()):
            color, marker = EXP_STYLES[idx % len(EXP_STYLES)]
            marker_size = 350 if marker in ['*', 'X'] else 120 
            
            ax.scatter(data['x'], data['t'], marker=marker, s=marker_size, 
                       facecolors=color, edgecolors='black', linewidths=1.5, zorder=20)
            
            exp_legend_handles.append(
                mlines.Line2D([], [], color='white', marker=marker, markerfacecolor=color, 
                              markeredgecolor='black', markeredgewidth=1.5, 
                              markersize=14 if marker in ['*', 'X'] else 10, label=ref)
            )

    # ==========================================
    # 🌟 极简模块化图例
    # ==========================================
    legend_handles = [
        mlines.Line2D([], [], color='white', marker='o', markerfacecolor=THEME_COLORS['scatter'], markersize=7, label='Calculated Samples'),
        mlines.Line2D([], [], color=THEME_COLORS['ci_95'], linestyle='dashed', linewidth=2.0, label='95% CI'),
        mlines.Line2D([], [], color=THEME_COLORS['ci_68'], linestyle='solid', linewidth=2.5, label='68% CI')
    ]
    if exp_legend_handles:
        legend_handles.extend(exp_legend_handles)
    
    ax.legend(handles=legend_handles, loc='best', frameon=True, edgecolor='black', fontsize=12)
    ax.margins(x=0.15, y=0.15)

    # ==========================================
    # 坐标轴与画板美化
    # ==========================================
    ax.set_box_aspect(1) 
    ax.set_xlabel(r"Mole Fraction of Si ($X_{\mathrm{Si}}$)", labelpad=10, fontsize=15)
    ax.set_ylabel(r"Invariant Temperature (K)", labelpad=10, fontsize=15)
    
    for spine in ax.spines.values(): 
        spine.set_linewidth(1.5)
        
    ax.tick_params(which='both', direction='in', top=True, right=True, width=1.5, length=6, colors='black')
    
    plt.tight_layout()
    output_name = "InvariantPoint_LIQUID_Accurate.png"
    fig.savefig(output_name, bbox_inches='tight')
    print(f"✅ 学术级 PDUQ 图像已完美保存: {output_name}")
    plt.close(fig)

# ==========================================
# 🚀 6. 主程序入口
# ==========================================
if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    # 文件路径配置
    TDB_FILE = "B23-MG-SI_Final_Optimized.tdb"  
    DB_FILE = "B23-MG-SI_Final_Optimized.db"
    INPUT_DATA_DIR = "input-data"  
    
    dbf = Database(TDB_FILE)
    comps = ['MG', 'SI', 'VA'] 
    
    # ---------------------------------------------------------
    # ⚠️ 核心参数配置区
    # ---------------------------------------------------------
    TARGET_PHASES = list(dbf.phases.keys())       
    TEMP_RANGE    = (1200, 1220, 0.1)               
    X_RANGE       = (0.50, 0.55, 0.005)           
    
    search_conds = {
        v.P: 101325, 
        v.T: TEMP_RANGE,            
        v.X('SI'): X_RANGE 
    }
    
    dynamic_exp_points = extract_zpf_in_box(
        data_path=INPUT_DATA_DIR, 
        comp_x='SI', 
        comps=comps, 
        x_range=(X_RANGE[0], X_RANGE[1]), 
        t_range=(TEMP_RANGE[0], TEMP_RANGE[1])
    )
    
    ensemble = extract_ensemble_parameters(db_file=DB_FILE, min_score=-125000000000, n_samples=200)
    
    if len(ensemble) > 0:
        plot_pduq_invariant_point(
            dbf=dbf, 
            ensemble=ensemble, 
            comps=comps,
            target_phases=TARGET_PHASES,
            search_conds=search_conds,
            exp_data=dynamic_exp_points
        )