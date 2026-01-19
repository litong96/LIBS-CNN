import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

# ================= 配置区域 =================
INPUT_DIR = r"D:\PycharmProject\pytorch\libs\all_datasets"
OUTPUT_DIR = r"D:\PycharmProject\pytorch\libs\baseline_corrected_datasets"
PDF_DIR = r"D:\PycharmProject\pytorch\libs\baseline_reports"

# 多项式拟合参数
# degree: 多项式阶数。LIBS背景通常是宽波段的轫致辐射，5-7阶通常比较合适。
# 阶数太高会导致基线“钻”到峰里面去；太低则拟合不了弯曲的背景。
POLY_DEGREE = 6
MAX_ITER = 10  # 迭代次数
TOLERANCE = 1e-3  # 收敛阈值


# ===========================================

def modified_poly_fit(y, degree, n_iter):
    """
    迭代多项式拟合算法 (ModPoly)
    自动寻找非峰区域（基线），通过迭代剔除峰值点。
    """
    n = len(y)
    x = np.arange(n)

    # 初始化：使用原始数据作为基准
    y_base = y.copy()

    for _ in range(n_iter):
        # 1. 拟合当前的数据点
        coeffs = np.polyfit(x, y_base, degree)
        baseline = np.polyval(coeffs, x)

        # 2. 关键步骤：比较 原始拟合值 和 真实值
        # 如果真实值 y 大于 拟合基线 baseline，说明它是峰，我们用基线值代替它
        # 这样下一次拟合时，这个位置的权重就被拉低了，逼近底部
        y_base = np.minimum(y, baseline)

    return baseline


def process_baseline_correction():
    # 1. 准备文件夹
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)

    # 2. 扫描文件
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]

    if not csv_files:
        print(f"在 {INPUT_DIR} 中没有找到 CSV 文件。")
        return

    print(f"找到 {len(csv_files)} 个数据集，开始处理...")

    for csv_file in csv_files:
        file_path = os.path.join(INPUT_DIR, csv_file)
        print(f"\n正在处理: {csv_file}")

        # 读取数据
        df = pd.read_csv(file_path)
        filenames = df.iloc[:, 0].values
        wavelengths = df.columns[1:].astype(float)
        raw_intensities = df.iloc[:, 1:].values

        n_samples = len(filenames)
        corrected_intensities = []

        # 准备 PDF
        pdf_name = f"Report_{csv_file.replace('.csv', '.pdf')}"
        pdf_path = os.path.join(PDF_DIR, pdf_name)

        # 处理该文件中的每一个样品
        with PdfPages(pdf_path) as pdf:
            for i in range(n_samples):
                y_raw = raw_intensities[i]

                # --- 核心算法：基线拟合 ---
                baseline = modified_poly_fit(y_raw, degree=POLY_DEGREE, n_iter=MAX_ITER)

                # --- 扣除基线 ---
                # 结果 = 原始 - 基线
                # 偶尔会出现微小的负值（拟合误差），通常置0或保留均可，这里保留
                y_corrected = y_raw - baseline

                corrected_intensities.append(y_corrected)

                # --- 绘图 (仅抽取部分样品生成PDF，避免文件过大，或者全部生成) ---
                # 这里为了演示，我们每处理完一个样品都画图。
                # 如果样品非常多(>100)，建议每隔 5 个画一张，或者只画前 20 张。
                if i < 20:  # [可选] 限制PDF页数，例如只看前20个样品的效果
                    plt.figure(figsize=(10, 8))

                    # 子图1: 原始光谱 + 拟合的基线
                    plt.subplot(2, 1, 1)
                    plt.plot(wavelengths, y_raw, 'k', label='Raw Spectrum', lw=0.8, alpha=0.7)
                    plt.plot(wavelengths, baseline, 'r--', label=f'Fitted Baseline (Poly={POLY_DEGREE})', lw=1.5)
                    plt.title(f"Sample: {filenames[i]} (Baseline Fitting)")
                    plt.legend()
                    plt.ylabel("Intensity")
                    plt.grid(True, linestyle=':', alpha=0.3)

                    # 子图2: 校正后的光谱
                    plt.subplot(2, 1, 2)
                    plt.plot(wavelengths, y_corrected, 'b', label='Corrected Spectrum', lw=0.8)
                    plt.title("Baseline Corrected")
                    plt.xlabel("Wavelength (nm)")
                    plt.ylabel("Intensity")
                    plt.legend()
                    plt.grid(True, linestyle=':', alpha=0.3)

                    plt.tight_layout()
                    pdf.savefig()  # 保存当前页
                    plt.close()

        # 保存处理后的 CSV
        # 转换为 DataFrame
        corrected_df = pd.DataFrame(corrected_intensities, columns=wavelengths)
        corrected_df.insert(0, "Filename", filenames)

        save_csv_path = os.path.join(OUTPUT_DIR, f"baseline_{csv_file}")
        corrected_df.to_csv(save_csv_path, index=False)

        print(f"  -> 数据已保存: {save_csv_path}")
        print(f"  -> 报告已生成: {pdf_path}")

    print("-" * 30)
    print("所有文件基线校正完成！")


if __name__ == '__main__':
    process_baseline_correction()