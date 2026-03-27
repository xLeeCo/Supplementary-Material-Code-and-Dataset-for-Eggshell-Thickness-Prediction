import numpy as np
import pandas as pd
import matplotlib

# 强制使用 Agg 后端以确保稳定保存高分辨率图片
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
import os
import warnings

warnings.filterwarnings('ignore')

# --- 1. 全局路径配置 ---
BASE_DIR = r"D:\thickness\Supplementary_Material"
DATA_PATH = os.path.join(BASE_DIR, "Eggshell_Top50_Dataset.xlsx")
SAVE_PATH = os.path.join(BASE_DIR, "Final_Thickness_Prediction.png")


# --- 2. 预处理：SNV + MSC ---
def snv_msc_preprocess(X_train, X_test):
    def apply_snv(data):
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        return (data - mean) / std

    X_tr_snv = apply_snv(X_train)
    X_te_snv = apply_snv(X_test)
    ref = np.mean(X_tr_snv, axis=0)

    def apply_msc(data, reference):
        processed = np.zeros_like(data)
        for i in range(data.shape[0]):
            poly = np.polyfit(reference, data[i, :], 1)
            processed[i, :] = (data[i, :] - poly[1]) / poly[0]
        return processed

    return apply_msc(X_tr_snv, ref), apply_msc(X_te_snv, ref)


# --- 3. GBDT 特征选择 (Top-50) ---
def get_gbdt_features(X, y):
    gbdt = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gbdt.fit(X, y)
    return np.argsort(gbdt.feature_importances_)[::-1][:50]


# --- 4. 手动内层循环调参 (避开兼容性报错) ---
def find_best_params(X, y):
    best_p, min_err = None, float('inf')
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    for depth in [4, 6]:
        for lr in [0.05, 0.1]:
            for l2 in [3, 7]:
                params = {'depth': depth, 'learning_rate': lr, 'l2_leaf_reg': l2}
                errs = []
                for tr_idx, val_idx in inner_cv.split(X):
                    model = CatBoostRegressor(iterations=100, **params, bootstrap_type='Bernoulli',
                                              subsample=0.8, verbose=0, random_state=42)
                    model.fit(X[tr_idx], y[tr_idx])
                    errs.append(mean_squared_error(y[val_idx], model.predict(X[val_idx])))
                if np.mean(errs) < min_err:
                    min_err, best_p = np.mean(errs), params
    return best_p


# --- 5. 绘图：完美复刻目标风格 ---
def plot_style_replication(y_true, y_pred, results):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 14
    fig, ax = plt.subplots(figsize=(8, 8))

    # 回归线与预测带计算
    slope, intercept, _, _, _ = stats.linregress(y_true, y_pred)
    x_range = np.linspace(0.26, 0.46, 100)
    y_fit = slope * x_range + intercept
    n = len(y_true)
    mse = np.sum((y_pred - (slope * y_true + intercept)) ** 2) / (n - 2)
    t_val = stats.t.ppf(0.975, n - 2)
    x_mean = np.mean(y_true)
    sum_sq_x = np.sum((y_true - x_mean) ** 2)
    conf = t_val * np.sqrt(mse) * np.sqrt(1 + 1 / n + (x_range - x_mean) ** 2 / sum_sq_x)

    # 1. 95% 预测带
    ax.fill_between(x_range, y_fit - conf, y_fit + conf, color='#ffebee', alpha=0.8, label='95% Prediction Band')
    # 2. 参考线 (y=x)
    ax.plot([0.26, 0.46], [0.26, 0.46], '--', color='black', linewidth=1, label='Reference Line (y=x)')
    # 3. 拟合线
    ax.plot(x_range, y_fit, color='#d32f2f', linewidth=2, label='Fit Line')
    # 4. 空心散点
    ax.scatter(y_true, y_pred, facecolors='none', edgecolors='#3498db', s=45, linewidths=0.8,
               label=f'Test Samples (N={n})')

    # 统计信息框 (对标右图格式)
    stats_text = (
            r"$\bf{10-fold\ Nested\ CV}$" + "\n" +
            "--------------------------\n" +
            fr"$R^2_p = {np.mean(results['r2']):.3f}$" + "\n" +
            fr"$MAE = {np.mean(results['mae']):.3f}\ mm$" + "\n" +
            fr"$RMSEP = {np.mean(results['rmse']):.3f}\ mm$"
    )
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=1, edgecolor='gray'))

    ax.set_xlabel('Actual thickness (mm)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Predicted thickness (mm)', fontweight='bold', fontsize=14)
    ax.set_xlim(0.26, 0.45);
    ax.set_ylim(0.26, 0.45)
    ax.tick_params(direction='in', top=True, right=True)
    ax.legend(loc='lower right', frameon=True, edgecolor='black', fontsize=11)

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=600)
    print(f"✅ 图片已成功保存至: {SAVE_PATH}")


if __name__ == "__main__":
    try:
        print(f"📂 正在加载数据: {DATA_PATH}")
        df = pd.read_excel(DATA_PATH)
        X_all, y_all = df.iloc[:, :-1].values, df.iloc[:, -1].values

        outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
        res = {'r2': [], 'mae': [], 'rmse': []}
        all_true, all_pred = [], []

        print("🚀 开始 10 折嵌套交叉验证...")
        for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X_all, y_all)):
            X_tr_p, X_te_p = snv_msc_preprocess(X_all[tr_idx], X_all[te_idx])
            f_idx = get_gbdt_features(X_tr_p, y_all[tr_idx])
            X_tr_sel, X_te_sel = X_tr_p[:, f_idx], X_te_p[:, f_idx]

            model = CatBoostRegressor(iterations=150, **find_best_params(X_tr_sel, y_all[tr_idx]),
                                      bootstrap_type='Bernoulli', subsample=0.8, verbose=0, random_state=42)
            model.fit(X_tr_sel, y_all[tr_idx])
            fold_pred = model.predict(X_te_sel)

            all_true.extend(y_all[te_idx]);
            all_pred.extend(fold_pred)
            res['r2'].append(r2_score(y_all[te_idx], fold_pred))
            res['mae'].append(mean_absolute_error(y_all[te_idx], fold_pred))
            res['rmse'].append(np.sqrt(mean_squared_error(y_all[te_idx], fold_pred)))
            print(f"Fold {fold + 1}/10 - R2: {res['r2'][-1]:.3f}")

        plot_style_replication(np.array(all_true), np.array(all_pred), res)
    except Exception as e:
        print(f"❌ 运行失败: {e}")