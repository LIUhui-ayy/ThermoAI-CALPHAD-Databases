import os
import sys
import builtins
import numpy as np
import symengine
from pycalphad import variables as v
import espei.parameter_selection.fitting_steps as fs
from espei.parameter_selection.model_building import make_successive
import espei.parameter_selection.selection as sel
import inspect
import re

# ==========================================
# 🛡️ 智能 UTF-8 全局补丁
# ==========================================
_original_open = builtins.open
def _smart_utf8_open(*args, **kwargs):
    mode = kwargs.get('mode', args[1] if len(args) > 1 else 'r')
    if 'b' not in mode and 'encoding' not in kwargs:
        kwargs['encoding'] = 'utf-8'
    return _original_open(*args, **kwargs)
builtins.open = _smart_utf8_open 
os.environ["PYTHONUTF8"] = "1"

# ==========================================
# 🧠 零特征安全补丁：允许参数静音
# ==========================================
orig_fit = sel.fit_model
def safe_fit_model(feature_matrix, data_quantities, ridge_alpha, weights=None):
    if feature_matrix.shape[1] == 0:
        return np.array([])  
    return orig_fit(feature_matrix, data_quantities, ridge_alpha, weights)
sel.fit_model = safe_fit_model

orig_score = sel.score_model
def safe_score_model(feature_matrix, data_quantities, model_coefficients, feature_list, weights, aicc_factor=None):
    if len(feature_list) == 0:
        rss = np.sum(data_quantities**2) if weights is None else np.sum(weights * data_quantities**2)
        n = len(data_quantities)
        if n == 0 or rss == 0:
            return -np.inf
        return float(n * np.log(rss / n))
    return orig_score(feature_matrix, data_quantities, model_coefficients, feature_list, weights, aicc_factor)
sel.score_model = safe_score_model

# ==========================================
# 🧠 唯一纯净指数模型生成器
# ==========================================
original_hm_features = fs.StepHM.features
original_sm_features = fs.StepSM.features

def get_current_phase():
    """逆向探测调用栈，侦测当前正在拟合的相"""
    try:
        for frame_info in inspect.stack():
            if 'phase_name' in frame_info.frame.f_locals:
                return frame_info.frame.f_locals['phase_name']
    except Exception:
        pass
    return None

def custom_hm_features(cls):
    phase = get_current_phase()
    if phase == 'LIQUID':
        # 💡 核心改动 1：只提供单指数候选项
        # ESPEI 会从中选一个最贴合数据的，拟合出唯一的 VVXXXX
        return [

            [symengine.exp(-v.T / 2000.0)],
            [symengine.exp(-v.T / 3000.0)],
            [symengine.exp(-v.T / 5000.0)],
        ]
    return make_successive(original_hm_features)

def custom_sm_features(cls):
    phase = get_current_phase()
    if phase == 'LIQUID':
        # 💡 核心改动 2：熵步骤彻底静音！！！
        # 绝对不允许 SM 步骤再添加第二个 EXP 参数
        return [[]]
    return make_successive(original_sm_features)

fs.StepHM.get_feature_sets = classmethod(custom_hm_features)
fs.StepSM.get_feature_sets = classmethod(custom_sm_features)

# ==========================================
# 🚀 自动生成配置文件并启动 ESPEI
# ==========================================
yaml_config = """
system:
  phase_models: phase_models.json
  datasets: input-data
  tags:
    dft:
      excluded_model_contributions: ['idmix', 'mag']
    estimated-entropy:
      excluded_model_contributions: ['idmix', 'mag']
      weight: 0.1
generate_parameters:
  excess_model: linear
  ref_state: SGTE91
  aicc_penalty_factor:
    HCP_A3:
      HM: 2
      SM: 2
    LIQUID:
      HM: 0.5
      SM: 0.5
      
"""

with open("generate_kaptay.yaml", "w") as f:
    f.write(yaml_config)

if __name__ == "__main__":
    print("🚀 启动【纯净单指数】生成引擎...")
    
    if not os.path.exists("phase_models.json") or not os.path.exists("input-data"):
        print("❌ 错误: 找不到 phase_models.json 或 input-data。")
        sys.exit(1)

    print("   ➤ 已切断多余参数！液相将被强制约束为唯一的 VV*EXP 形式。")
    
    from espei.espei_script import main
    sys.argv = ['espei', '--input', 'generate_kaptay.yaml']
    
    try:
        main()
        
        # ==========================================
        # ✨ 后处理美化器：将小数转换回纯正 Kaptay 格式
        # ==========================================
        if os.path.exists("out.tdb"):
            print("\n✨ 正在执行数学倒数反演与美化重构...")
            with open("out.tdb", "r") as f:
                content = f.read()
                
            max_vv = 0
            for m in re.finditer(r'VV(\d+)', content):
                max_vv = max(max_vv, int(m.group(1)))
                
            new_funcs = []
            
            def replace_exp_decimals(match):
                global max_vv
                vv_before = match.group(1)
                val_str = match.group(2)  
                vv_after = match.group(3)
                
                # 安全获取前置系数 (修复带有负号的参数)
                vv_a = vv_before if vv_before else vv_after
                if not vv_a:
                    vv_a = "1.0" 
                    
                tau = round(-1.0 / float(val_str))
                
                max_vv += 1
                vv_tau = f"VV{max_vv:04d}"
                new_funcs.append(f"FUNCTION {vv_tau} 1 {tau:.4f}; 10000 N !\n")
                
                # 完美重构：唯一的 VV_A * EXP(-T / VV_TAU)
                return f"{vv_a}*EXP(-T/{vv_tau})"
                
            # 增强的正则：兼容 -VV0002*EXP(...)
            pattern = r'(?:([+-]?\s*VV\d+)\s*\*\s*)?EXP\(\s*(-?[0-9\.]+)\s*\*\s*T\s*\)(?:\s*\*\s*(VV\d+))?'
            content = re.sub(pattern, replace_exp_decimals, content)
            
            if new_funcs:
                last_func_idx = content.rfind("FUNCTION ")
                if last_func_idx != -1:
                    end_of_line = content.find("!", last_func_idx) + 1
                    content = content[:end_of_line] + "\n" + "".join(new_funcs) + content[end_of_line:]
            
            with open("Mg-Si-Pure-Kaptay.tdb", "w") as f:
                f.write(content)
            os.remove("out.tdb")
            print("✅ 拦截美化成功！纯纯的单指数形态模型已保存至: Mg-Si-Pure-Kaptay.tdb")
        else:
            print("\n⚠️ 未生成 out.tdb 输出文件。")
            
    except Exception as e:
        print(f"\n❌ 生成失败，详细报错: {e}")