import subprocess
import time
import sys

# ==========================================
# ⚙️ 守护脚本配置
# ==========================================
MAIN_SCRIPT = "CMA_TPO_Optimization.py"  # 替换为你的主程序文件名
MAX_RESTARTS = 10            # 最大批次数：20000 / 2000 = 10
SLEEP_BETWEEN_RUNS = 60      # 批次之间的冷却休眠时间（秒），确保系统彻底回收内存和端口

def run_optimization():
    print(f"🛡️ [守护进程] 启动自动化多批次调度器...")
    print(f"🛡️ [守护进程] 目标脚本: {MAIN_SCRIPT}")
    
    for i in range(1, MAX_RESTARTS + 1):
        print("\n" + "="*50)
        print(f"🚀 [守护进程] 正在拉起第 {i} 轮计算批次...")
        print("="*50)
        
        start_time = time.time()
        
        # 启动子进程（使用当前运行此脚本的同一个 Python 环境）
        process = subprocess.Popen([sys.executable, MAIN_SCRIPT])
        
        # 阻塞等待主程序跑完这 2000 次并触发 os._exit()
        process.wait()
        
        elapsed_time = time.time() - start_time
        
        # 1. 检查是否有严重报错（主程序异常退出）
        if process.returncode != 0:
            print(f"\n❌ [守护进程] 主程序发生错误异常退出 (退出码: {process.returncode})，终止自动循环！")
            break
            
        # 2. 核心逻辑：智能判断是否已达到 20000 次总上限
        # 如果主程序启动后不到 30 秒就退出了，说明它跳过了计算循环（completed >= TOTAL_TRIALS）
        if elapsed_time < 30:
            print(f"\n✅ [守护进程] 批次运行时间极短 ({elapsed_time:.1f}秒)，推测 {MAIN_SCRIPT} 已完成所有优化目标！")
            print("🎉 [守护进程] 全局优化任务圆满结束！")
            break
            
        print(f"\n⏳ [守护进程] 第 {i} 轮批次已物理斩杀，内存已交还操作系统。")
        print(f"⏳ [守护进程] 冷却休眠 {SLEEP_BETWEEN_RUNS} 秒，等待 Dask 端口和 SQLite 锁彻底释放...")
        time.sleep(SLEEP_BETWEEN_RUNS)

if __name__ == "__main__":
    try:
        run_optimization()
    except KeyboardInterrupt:
        print("\n🛑 [守护进程] 接收到手动中断信号 (Ctrl+C)，已安全停止调度。")