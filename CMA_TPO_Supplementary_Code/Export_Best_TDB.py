import os
import optuna
from pycalphad import Database
from sympy import Float

# ==========================================
# ⚙️ 配置区 (精确匹配防死锁内存版主程序的数据库)
# ==========================================
DB_FILE = "B6-CU-MG_Final_Optimized.db"  # 🔥 必须与最新主程序的数据库名称完全一致
DB_URL = f"sqlite:///{DB_FILE}"
STUDY_NAME = "entropy_opt"              # 与最新优化的 study_name 一致
TDB_INPUT = "Cu-Mg-generated1.tdb"       # 您的基础模型文件
TDB_OUTPUT = "B6-CU-MG_Best_Final.tdb"     # 导出的终极最优 TDB 文件名

def extract_best_tdb():
    print("="*60)
    print(f"🔍 正在从 {DB_FILE} 提取【全局最优】参数...")
    print("="*60)
    
    if not os.path.exists(DB_FILE):
        print(f"❌ 错误: 找不到数据库文件 '{DB_FILE}'。")
        print("请确保优化程序已经生成了进度，并且该脚本与数据库在同一目录下。")
        return

    try:
        print(f"📂 正在连接数据库...")
        # 读取优化生成的单目标数据库
        study = optuna.load_study(study_name=STUDY_NAME, storage=DB_URL)
        best_trial = study.best_trial
    except ValueError:
        print("⚠️ 数据库中还没有任何完成的有效试探 (Trial)，请先让优化程序跑一会儿！")
        return
    except Exception as e:
        print(f"❌ 读取数据库失败 (可能数据库被锁定或名称错误): {e}")
        return

    print(f"\n🏆 当前全局最优解: Trial 编号 {best_trial.number:04d}")
    print(f"🌟 综合总分 (Score): {best_trial.value:,.0f} ")
    print(f"   (注：此分数越高，代表相图 ZPF 驱动力惩罚越小，且物理属性符合红线要求)")
    
    print("\n📊 寻优后的最佳参数详情:")
    for sym, val in best_trial.params.items():
        print(f"   ➤ {sym:<10}: {val:>15.4f}")
    
    print(f"\n⚙️ 正在将冠军参数注入并重构 TDB 文件...")
    try:
        dbf = Database(TDB_INPUT)
        for sym, val in best_trial.params.items():
            if sym in dbf.symbols:
                # 严格使用 Sympy 的 Float 保持计算材料学所需的精度
                dbf.symbols[sym] = Float(val)
        dbf.to_file(TDB_OUTPUT)
        print(f"\n✅ 提取大成功！")
        print(f"🎉 您的完美相图参数已保存至: 【{TDB_OUTPUT}】")
        print(f"🚀 现在您可以直接使用这个新文件去绘制相图了！")
    except Exception as e:
        print(f"\n❌ TDB 生成失败: {e}")

if __name__ == "__main__":
    extract_best_tdb()