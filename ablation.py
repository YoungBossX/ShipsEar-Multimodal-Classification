# ablation.py
# 运行消融实验：依次关闭各模态分支，记录准确率

import subprocess
import re
from src.utils.visualization import plot_modality_contribution

def run_experiment(overrides: str) -> float:
    """运行一次训练+推理，从日志里解析最终准确率"""
    result = subprocess.run(
        f"python train.py {overrides}",
        shell=True, capture_output=True, text=True
    )
    # 从输出里解析 Overall accuracy
    match = re.search(r"Overall accuracy: (\d+\.\d+)%", result.stdout + result.stderr)
    return float(match.group(1)) if match else 0.0


if __name__ == "__main__":
    experiments = {
        "Time only":        "datasets.use_spectrogram_branch=false datasets.use_text_branch=false",
        "Spectrogram only": "datasets.use_time_branch=false datasets.use_text_branch=false",
        "Text only":        "datasets.use_time_branch=false datasets.use_spectrogram_branch=false",
        "Time + Spec":      "datasets.use_text_branch=false",
        "Time + Text":      "datasets.use_spectrogram_branch=false",
        "Spec + Text":      "datasets.use_time_branch=false",
        "All (Fusion)":     "",   # 默认全开
    }

    results = {}
    for name, overrides in experiments.items():
        print(f"Running: {name} ...")
        acc = run_experiment(overrides)
        results[name] = acc
        print(f"  → Accuracy: {acc:.1f}%")

    plot_modality_contribution(results, save_dir="saved/plots")
    print("Ablation study complete.")