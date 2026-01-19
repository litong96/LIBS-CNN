import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold  # 保持使用 StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
from scipy.interpolate import interp1d
import shap
import copy
import shutil

# ================= 1. config =================
DATA_FILE = r"D:\PycharmProject\pytorch\libs\final_dataset\Final_merged_Dataset.csv"
OUTPUT_DIR = r"D:\PycharmProject\pytorch\libs\model_results_Detailed_v3"

NUM_CLASSES = 5
BATCH_SIZE = 64
EPOCHS = 1000
PATIENCE = 50
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOLDS = 5  # 5折

COUNTRY_TRANSLATION = {
    "China": "China",
    "India": "India",
    "Russia": "Russia",
    "Brazil": "Brazil",
    "South Africa": "South Africa"
}


# ==================================================

# --- earlystopping ---
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# --- shap ---
class LIBSAnalyzer:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = []

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def _register_hooks(self):
        h1 = self.target_layer.register_forward_hook(self._save_activation)
        h2 = self.target_layer.register_full_backward_hook(self._save_gradient)
        self.handlers = [h1, h2]

    def _remove_hooks(self):
        for h in self.handlers:
            h.remove()
        self.handlers = []

    def get_analysis(self, x_tensor, class_idx=None):
        self.model.eval()
        self._register_hooks()
        x_tensor.requires_grad_()
        output = self.model(x_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()
        saliency = x_tensor.grad.data.abs().cpu().numpy().flatten()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)
        grads = self.gradients.data.cpu().numpy()[0]
        fmaps = self.activations.data.cpu().numpy()[0]
        weights = np.mean(grads, axis=1)
        cam = np.zeros(fmaps.shape[1], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * fmaps[i]
        cam = np.maximum(cam, 0)
        x_old = np.linspace(0, 1, len(cam))
        x_new = np.linspace(0, 1, x_tensor.shape[2])
        f = interp1d(x_old, cam, kind='linear')
        cam_resized = f(x_new)
        cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-10)
        self._remove_hooks()
        return saliency, cam_resized, class_idx


# --- 数据集类 ---
class LIBSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(1)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- CNN 模型 ---
class SpectralCNN(nn.Module):
    def __init__(self, num_classes, input_length):
        super(SpectralCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 5, 1, 2), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, 1, 2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.flatten_dim = self._get_flatten_dim(input_length)
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _get_flatten_dim(self, length):
        dummy_input = torch.zeros(1, 1, length)
        with torch.no_grad():
            output = self.features(dummy_input)
        return output.view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# --- 绘图与报告函数 ---
def plot_training_history_robust(history, output_dir, fold_idx):
    min_len = min(len(v) for v in history.values())
    if min_len == 0: return
    clean_history = {k: v[:min_len] for k, v in history.items()}

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(clean_history['train_loss'], label='Train Loss')
    plt.plot(clean_history['val_loss'], label='Val Loss')
    plt.title(f'Loss Curve (Fold {fold_idx})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(clean_history['train_acc'], label='Train Acc')
    plt.plot(clean_history['val_acc'], label='Val Acc')
    plt.title(f'Accuracy Curve (Fold {fold_idx})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"training_curves_fold_{fold_idx}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

    # 保存该折的历史数据
    pd.DataFrame(clean_history).to_csv(os.path.join(output_dir, f"training_history_fold_{fold_idx}.csv"), index=False)
    return filename


def run_shap_analysis(model, train_loader, test_loader, device, wavelengths, class_names, output_dir, top_n=3):
    """SHAP 分析函数"""
    print(f"\n[SHAP] 正在初始化全量 SHAP 分析...")
    try:
        batch_X, _ = next(iter(train_loader))
        background = batch_X[:100].to(device)
        explainer = shap.DeepExplainer(model, background)

        candidates = {i: [] for i in range(len(class_names))}
        model.eval()

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                for i in range(len(labels)):
                    true_label = labels[i].item()
                    pred_label = preds[i].item()
                    prob = probs[i, pred_label].item()
                    if true_label == pred_label:
                        candidates[true_label].append({
                            'prob': prob,
                            'input': inputs[i:i + 1],
                            'batch_idx': batch_idx,
                            'in_batch_idx': i
                        })

        for cls_idx in range(len(class_names)):
            cls_name = class_names[cls_idx]
            cls_name_en = COUNTRY_TRANSLATION.get(cls_name, cls_name)
            sorted_candidates = sorted(candidates[cls_idx], key=lambda x: x['prob'], reverse=True)
            top_candidates = sorted_candidates[:top_n]
            if not top_candidates:
                continue

            print(f" -> 处理 {cls_name_en} 类 (Top {len(top_candidates)})...")
            for rank, item in enumerate(top_candidates):
                sample_tensor = item['input']
                prob = item['prob']
                try:
                    shap_values_raw = explainer.shap_values(sample_tensor)
                    target_shap = None
                    if isinstance(shap_values_raw, list):
                        target_shap = shap_values_raw[cls_idx]
                    else:
                        shap_values_raw = np.array(shap_values_raw)
                        if shap_values_raw.ndim == 4 and shap_values_raw.shape[-1] == NUM_CLASSES:
                            target_shap = shap_values_raw[0, 0, :, cls_idx]
                        elif shap_values_raw.shape[0] == NUM_CLASSES:
                            target_shap = shap_values_raw[cls_idx]
                        else:
                            flat = shap_values_raw.flatten()
                            reshaped = flat.reshape(-1, NUM_CLASSES)
                            target_shap = reshaped[:, cls_idx]

                    if target_shap is not None:
                        sv = np.array(target_shap).reshape(-1)
                        sv = np.maximum(sv, 0)  # 只保留正相关

                        original_spectrum = sample_tensor.cpu().numpy().reshape(-1)
                        plt.figure(figsize=(14, 6))
                        plt.plot(wavelengths, original_spectrum, 'k-', alpha=0.3, label='Original Spectrum',
                                 linewidth=1)
                        ax1 = plt.gca()
                        ax1.set_xlabel('Wavelength (nm)')
                        ax1.set_ylabel('Normalized Intensity')
                        ax2 = ax1.twinx()
                        ax2.bar(wavelengths, sv, color='red', alpha=0.6, width=(wavelengths[1] - wavelengths[0]),
                                label='SHAP')
                        ax2.set_ylabel('SHAP Contribution (Positive)', color='red')
                        plt.title(f'SHAP: {cls_name_en} | Rank-{rank + 1} (Conf: {prob:.4f})')
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f"SHAP_{cls_name_en}_Rank{rank + 1}.png"), dpi=300)
                        plt.close()
                except Exception as e:
                    print(f"    ❌ 解析失败: {e}")
                    continue
    except Exception as e:
        print(f"❌ SHAP 分析初始化失败: {e}")


# 🔥 新增：通用评估函数，用于计算精确的 Loss 和 Acc
def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


# ================= 6. 主程序 =================
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"使用设备: {DEVICE}")
    print(f"正在读取数据: {DATA_FILE} ...")

    # 1. 数据准备
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        print(f"❌ 读取数据失败: {e}")
        return

    y_raw = df.iloc[:, 0].values
    sample_names_raw = df.iloc[:, 1].values.astype(str)
    X_raw = df.iloc[:, 2:].values.astype(np.float32)
    wavelengths = df.columns[2:].astype(float)
    input_length = X_raw.shape[1]

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    class_names = le.classes_
    english_labels = [COUNTRY_TRANSLATION.get(name, name) for name in class_names]

    # 使用 Stratified K-Fold
    print(f"\n🚀 开始 {FOLDS}-折 分层交叉验证 (Stratified K-Fold)...")
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

    # 存储所有折的详细指标
    fold_metrics = {
        'Fold': [],
        'Train Acc': [],
        'Val Acc': [],
        'Train Loss': [],
        'Val Loss': []
    }

    # 用于记录最佳模型
    best_fold_idx = -1
    best_fold_val_acc = 0.0
    best_model_state = None
    best_train_indices = None
    best_val_indices = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y_enc)):
        print(f"\n=== Fold {fold + 1}/{FOLDS} ===")

        # 数据划分
        X_train, X_val = X_raw[train_idx], X_raw[val_idx]
        y_train, y_val = y_enc[train_idx], y_enc[val_idx]

        train_loader = DataLoader(LIBSDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        # 为了计算准确的Train Acc，这里再建一个不shuffle的train loader
        train_loader_eval = DataLoader(LIBSDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
        val_loader = DataLoader(LIBSDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

        # 初始化模型
        model = SpectralCNN(num_classes=NUM_CLASSES, input_length=input_length).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        # 早停
        fold_save_path = os.path.join(OUTPUT_DIR, f"model_fold_{fold + 1}.pth")
        early_stopping = EarlyStopping(patience=PATIENCE, verbose=False, path=fold_save_path)
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        # 训练循环
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # 验证
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            avg_train_loss = train_loss / train_total
            avg_train_acc = 100 * train_correct / train_total
            avg_val_loss = val_loss / val_total
            avg_val_acc = 100 * val_correct / val_total

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(avg_train_acc)
            history['val_acc'].append(avg_val_acc)

            if (epoch + 1) % 10 == 0:
                print(
                    f"  Epoch [{epoch + 1}/{EPOCHS}] Train Acc: {avg_train_acc:.2f}% | Val Acc: {avg_val_acc:.2f}% | Loss: {avg_val_loss:.4f}")

            scheduler.step(avg_val_loss)
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print(f"  🚀 Fold {fold + 1} 早停于 Epoch {epoch + 1}")
                break

        # 保存曲线
        curve_file = plot_training_history_robust(history, OUTPUT_DIR, fold + 1)

        # 🔥 加载该折的最佳权重，计算最终的精确指标
        model.load_state_dict(torch.load(fold_save_path))

        final_train_loss, final_train_acc = evaluate_model(model, train_loader_eval, criterion, DEVICE)
        final_val_loss, final_val_acc = evaluate_model(model, val_loader, criterion, DEVICE)

        print(f"  ✅ Fold {fold + 1} 结果: Val Acc: {final_val_acc:.2f}%, Train Acc: {final_train_acc:.2f}%")

        # 记录到列表
        fold_metrics['Fold'].append(fold + 1)
        fold_metrics['Train Acc'].append(final_train_acc)
        fold_metrics['Val Acc'].append(final_val_acc)
        fold_metrics['Train Loss'].append(final_train_loss)
        fold_metrics['Val Loss'].append(final_val_loss)

        # 更新全局最佳模型 (基于 Val Acc)
        if final_val_acc > best_fold_val_acc:
            best_fold_val_acc = final_val_acc
            best_fold_idx = fold
            best_model_state = copy.deepcopy(model.state_dict())
            best_train_indices = train_idx
            best_val_indices = val_idx

    # --- 汇总与输出 ---
    print("\n" + "=" * 50)
    print(f"📊 {FOLDS}-Fold Cross-Validation 详细报告")
    print("=" * 50)

    # 转换为 DataFrame
    df_metrics = pd.DataFrame(fold_metrics)

    # 计算平均值和标准差
    avg_metrics = df_metrics.iloc[:, 1:].mean()
    std_metrics = df_metrics.iloc[:, 1:].std()

    # 打印详细表格
    print(df_metrics.to_string(index=False))
    print("-" * 50)
    print(f"平均 Train Acc: {avg_metrics['Train Acc']:.2f}% (+/- {std_metrics['Train Acc']:.2f}%)")
    print(f"平均 Val Acc  : {avg_metrics['Val Acc']:.2f}% (+/- {std_metrics['Val Acc']:.2f}%)")
    print(f"平均 Train Loss: {avg_metrics['Train Loss']:.4f}")
    print(f"平均 Val Loss  : {avg_metrics['Val Loss']:.4f}")

    # 保存 CSV
    df_metrics.to_csv(os.path.join(OUTPUT_DIR, "kfold_detailed_summary.csv"), index=False)

    # --- 最佳模型处理 ---
    print("\n" + "=" * 50)
    print(f"🏆 最佳模型来自 Fold {best_fold_idx + 1}")
    print("=" * 50)
    print(f"最佳模型指标:")
    print(f"  Train Acc : {fold_metrics['Train Acc'][best_fold_idx]:.2f}%")
    print(f"  Val Acc   : {fold_metrics['Val Acc'][best_fold_idx]:.2f}%")
    print(f"  Train Loss: {fold_metrics['Train Loss'][best_fold_idx]:.4f}")
    print(f"  Val Loss  : {fold_metrics['Val Loss'][best_fold_idx]:.4f}")

    # 复制最佳模型的曲线图为特殊名称
    src_curve = os.path.join(OUTPUT_DIR, f"training_curves_fold_{best_fold_idx + 1}.png")
    dst_curve = os.path.join(OUTPUT_DIR, "best_model_training_curves.png")
    if os.path.exists(src_curve):
        shutil.copy(src_curve, dst_curve)
        print(f"  -> 最佳模型训练曲线已另存为: {dst_curve}")

    # ================= 详细分析 (基于最佳模型) =================
    print(f"\n🔍 正在使用最佳模型 (Fold {best_fold_idx + 1}) 进行混淆矩阵和SHAP分析...")

    final_model = SpectralCNN(num_classes=NUM_CLASSES, input_length=input_length).to(DEVICE)
    final_model.load_state_dict(best_model_state)
    final_model.eval()

    # 重建加载器
    X_best_train = X_raw[best_train_indices]
    y_best_train = y_enc[best_train_indices]
    X_best_val = X_raw[best_val_indices]
    y_best_val = y_enc[best_val_indices]
    names_best_val = sample_names_raw[best_val_indices]

    best_train_loader = DataLoader(LIBSDataset(X_best_train, y_best_train), batch_size=BATCH_SIZE, shuffle=True)
    best_val_loader = DataLoader(LIBSDataset(X_best_val, y_best_val), batch_size=BATCH_SIZE, shuffle=False)

    # 1. 生成预测
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in best_val_loader:
            inputs = inputs.to(DEVICE)
            outputs = final_model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    # 2. 混淆矩阵 (带 Recall 百分比 + 明确坐标轴)
    # 强制指定 labels 参数，确保顺序与 class_names (即 english_labels) 一致
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))

    # 计算行归一化矩阵 (即 Recall，用于显示百分比)
    # 加上 1e-10 防止除以 0
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

    # 准备格子里的文字： "数值\n(召回率%)"
    annot_labels = [
        [f"{value}\n({pct:.1%})" for value, pct in zip(row_counts, row_pcts)]
        for row_counts, row_pcts in zip(cm, cm_norm)
    ]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues',
                xticklabels=english_labels,
                yticklabels=english_labels,
                annot_kws={"size": 11})

    # 添加明确的坐标轴标签
    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    plt.ylabel('True Label', fontsize=13, fontweight='bold')

    plt.title(f'Confusion Matrix (Best Fold: {best_fold_idx + 1})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_best.png"), dpi=300)
    plt.close()

    # 3. 详细报告
    report_dict = classification_report(all_labels, all_preds, target_names=english_labels, output_dict=True)
    pd.DataFrame(report_dict).transpose().to_csv(os.path.join(OUTPUT_DIR, "classification_report_best.csv"))

    # 4. SHAP 分析
    run_shap_analysis(final_model, best_train_loader, best_val_loader, DEVICE, wavelengths, class_names, OUTPUT_DIR)

    print("-" * 30)
    print(f"全部完成！请查看: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()