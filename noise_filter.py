import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter
import pywt
import os

# ================= 配置区域 =================
# 输入路径：基线校正后的数据
INPUT_DIR = r"D:\PycharmProject\pytorch\libs\baseline_corrected_datasets"

# 输出路径根目录
OUTPUT_ROOT = r"D:\PycharmProject\pytorch\libs\denoised_results"
REPORT_DIR = r"D:\PycharmProject\pytorch\libs\denoised_reports"

# 算法参数配置 (您可以根据效果微调)
# 1. 移动平均 (MA)
MA_WINDOW = 5

# 2. Savitzky-Golay (SG)
SG_WINDOW = 15
SG_ORDER = 3

# 3. 小波去噪 (Wavelet)
WT_WAVELET = 'db1'  # 选用 Daubechies 1 小波
WT_LEVEL = 1  # 分解层数


# ===========================================

# --- 算法定义 (基于您提供的代码) ---
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def savgol_filtering(data, window_size, order):
    return savgol_filter(data, window_size, order)


def wavelet_denoising(data, wavelet, level):
    # 小波分解
    coeff = pywt.wavedec(data, wavelet, mode='per', level=level)
    # 计算噪声标准差估计 (基于最细尺度系数)
    sigma = np.median(np.abs(coeff[-1])) / 0.6745
    # 计算通用阈值
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    # 阈值处理 (Soft Thresholding)
    # 注意：coeff[0] 是近似系数(低频)，通常不处理；只处理细节系数(高频)
    coeff[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:]]
    # 小波重构
    return pywt.waverec(coeff, wavelet, mode='per')


def process_denoising():
    # 1. 创建输出目录结构
    methods = ['MA', 'SG', 'WT']
    for m in methods:
        path = os.path.join(OUTPUT_ROOT, m)
        if not os.path.exists(path):
            os.makedirs(path)

    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)

    # 2. 扫描输入文件
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
    if not csv_files:
        print(f"在 {INPUT_DIR} 未找到 CSV 文件。")
        return

    print(f"找到 {len(csv_files)} 个数据集，开始降噪处理...")

    for csv_file in csv_files:
        file_path = os.path.join(INPUT_DIR, csv_file)
        print(f"\n正在处理文件: {csv_file}")

        # 读取数据
        df = pd.read_csv(file_path)
        filenames = df.iloc[:, 0].values
        wavelengths = df.columns[1:].astype(float)
        raw_intensities = df.iloc[:, 1:].values
        n_samples = len(filenames)

        # 容器初始化
        results_ma = []
        results_sg = []
        results_wt = []

        # 准备 PDF 报告路径
        pdf_name_base = csv_file.replace('.csv', '')
        pdf_path_ma = os.path.join(REPORT_DIR, f"Report_MA_{pdf_name_base}.pdf")
        pdf_path_sg = os.path.join(REPORT_DIR, f"Report_SG_{pdf_name_base}.pdf")
        pdf_path_wt = os.path.join(REPORT_DIR, f"Report_WT_{pdf_name_base}.pdf")

        # 使用三个 PdfPages 上下文
        with PdfPages(pdf_path_ma) as pdf_ma, \
                PdfPages(pdf_path_sg) as pdf_sg, \
                PdfPages(pdf_path_wt) as pdf_wt:

            # 遍历每个样品
            for i in range(n_samples):
                y_raw = raw_intensities[i]

                # --- 执行三种算法 ---
                y_ma = moving_average(y_raw, MA_WINDOW)
                y_sg = savgol_filtering(y_raw, SG_WINDOW, SG_ORDER)
                # 某些情况下小波重构长度可能差1位，截断处理
                y_wt = wavelet_denoising(y_raw, WT_WAVELET, WT_LEVEL)[:len(y_raw)]

                results_ma.append(y_ma)
                results_sg.append(y_sg)
                results_wt.append(y_wt)

                # --- 绘图 (限制前20个样品以节省时间，需要全部请注释 if) ---
                if i < 20:
                    # 定义一个通用的绘图函数
                    def plot_comparison(pdf_obj, y_denoised, method_name, color):
                        plt.figure(figsize=(10, 6))

                        # 上图：原始数据 (Baseline Corrected)
                        plt.subplot(2, 1, 1)
                        plt.plot(wavelengths, y_raw, color='black', alpha=0.6, lw=0.8,
                                 label='Input (Baseline Corrected)')
                        plt.legend(loc='upper right')
                        plt.title(f"Sample: {filenames[i]} | Original vs {method_name}")
                        plt.ylabel("Intensity")
                        plt.grid(linestyle=':', alpha=0.3)
                        # 去除X轴标签
                        plt.gca().set_xticklabels([])

                        # 下图：降噪后数据
                        plt.subplot(2, 1, 2)
                        plt.plot(wavelengths, y_denoised, color=color, lw=1.0, label=f'{method_name} Result')
                        plt.legend(loc='upper right')
                        plt.xlabel("Wavelength (nm)")
                        plt.ylabel("Intensity")
                        plt.grid(linestyle=':', alpha=0.3)

                        plt.tight_layout()
                        pdf_obj.savefig()
                        plt.close()

                    # 分别生成三张图存入不同的 PDF
                    plot_comparison(pdf_ma, y_ma, f'Moving Average (W={MA_WINDOW})', 'green')
                    plot_comparison(pdf_sg, y_sg, f'Savitzky-Golay (W={SG_WINDOW}, O={SG_ORDER})', 'red')
                    plot_comparison(pdf_wt, y_wt, f'Wavelet ({WT_WAVELET})', 'purple')

        # --- 保存数据到 CSV ---
        def save_csv(data_list, method_folder, suffix):
            res_df = pd.DataFrame(data_list, columns=wavelengths)
            res_df.insert(0, "Filename", filenames)
            path = os.path.join(OUTPUT_ROOT, method_folder, f"{suffix}_{csv_file}")
            res_df.to_csv(path, index=False)
            print(f"  -> {suffix} 数据已保存")

        save_csv(results_ma, 'MA', 'ma_denoised')
        save_csv(results_sg, 'SG', 'sg_denoised')
        save_csv(results_wt, 'WT', 'wt_denoised')

        print(f"  -> 报告已生成: {pdf_path_ma}")
        print(f"  -> 报告已生成: {pdf_path_sg}")
        print(f"  -> 报告已生成: {pdf_path_wt}")

    print("-" * 30)
    print("所有降噪处理完成！请查看 denoised_reports 文件夹进行对比。")


if __name__ == '__main__':
    process_denoising()