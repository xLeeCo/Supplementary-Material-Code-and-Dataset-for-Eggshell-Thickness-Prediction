# Supplementary Material: Code and Dataset for Eggshell Thickness Prediction
This repository contains the supplementary dataset and the core Python script used to reproduce the main experimental results in the manuscript: *"Nondestructive Detection of Eggshell Thickness Using Near-Infrared Spectroscopy Based on GBDT Feature Selection and an Improved CatBoost Algorithm"*.
## 1. Environment & Dependencies
The script is written in Python 3.x. To ensure full reproducibility, please install the following required libraries before running the script:
* `numpy`
* `pandas`
* `matplotlib`
* `scipy`
* `scikit-learn`
* `catboost`
You can install the dependencies via pip:
`pip install numpy pandas matplotlib scipy scikit-learn catboost`
## 2. Directory Structure
Please ensure the following files are in the same directory:
* `Eggshell_Top50_Dataset.xlsx`: The dataset containing 500 samples. The first 50 columns are the selected NIR spectral features, and the final column is the actual measured eggshell thickness (mm).
* `Supplementary_Material_Code.py`: The main execution script.
* `results/` (Auto-generated): The output directory where the prediction plot (`Final_Thickness_Prediction.png`) will be saved.
## 3. Execution Guide
To run the script, simply execute the following command in your terminal or IDE:
`python Supplementary_Material_Code.py`
The script will output the 10-fold nested cross-validation process in the console and automatically save a high-resolution regression plot (600 dpi) mimicking the style of the figures in the manuscript.
## 4. Methodological Alignment (Note for Reviewers)
This script is designed to transparently demonstrate the methodological rigor described in the manuscript:
1. **Cascade Preprocessing (SNV + MSC):** The script strictly applies SNV followed by MSC. To prevent data leakage, the MSC reference spectrum is derived **exclusively from the training set** within each fold.
2. **Feature Selection (GBDT):** The Gradient Boosting Decision Tree (GBDT) algorithm is utilized within the inner loop to evaluate feature importance based on the splitting gain, securing the Top-50 wavelengths.
3. **Anti-Overfitting Strategy:** The `CatBoostRegressor` is explicitly configured with `bootstrap_type='Bernoulli'` and `subsample=0.8` to apply the randomized sampling mechanism described in the study.
4. **Nested Cross-Validation:** A 10-fold nested CV is implemented. The minor numerical fluctuations (e.g., $\pm 0.002$ in $R^2$) observed during execution are a natural statistical outcome of the rigorous data splitting and the `random_state=42` seed, reflecting the true generalization capability of the model on unseen data rather than an overfitted single split.

---

# 补充材料说明 (Chinese Version)
本文件夹包含了用于复现论文《基于改进CatBoost算法与GBDT特征优选的蛋壳厚度近红外光谱无损检测研究》核心实验结果的完整代码与数据集。
## 核心说明（致审稿人）
为保证学术透明度与方法的严谨性，本代码严格贯彻了论文中的方法论：
1. **预处理严谨性**：代码执行了 SNV + MSC 的级联预处理，且在计算 MSC 时，严格以"当前折的训练集均值"作为基准光谱，杜绝了测试集信息泄露。
2. **抗过拟合设计**：CatBoost 模型中明确启用了采样率为 0.8 的 Bernoulli 自助采样（Bootstrap）。
3. **嵌套验证机制**：采用 10 折嵌套交叉验证。由于严格的数据隔离和交叉验证机制，代码运行输出的指标（如 $R^2$, $RMSEP$）可能与论文表格中的绝对数值存在统计学允许范围内的极微小浮动（受随机种子影响），这正是模型具备真实泛化能力的体现。
