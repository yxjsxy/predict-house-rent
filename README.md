# 房源成交概率预测

根据地区、入住日期、星期、房型、床位、评分、图片质量和报价等特征，预测给定报价下房源被接受的概率。

## 文件

- `model.py`：特征工程、分类模型与评估实验。
- `train.csv`、`test.csv`：训练和测试数据。
- `output.csv`：历史预测输出，不代表重新训练后的结果。

目标字段为 `Accept`：1 表示被接受，0 表示未被接受。模型预测的是成交概率，并非直接给出最优房租。

## 使用方法

```bash
git clone https://github.com/yxjsxy/predict-house-rent.git
cd predict-house-rent
```

在独立 Python 环境中阅读 `model.py`，先完成以下准备，再运行脚本：

1. 根据导入语句安装兼容的 pandas、NumPy、scikit-learn、XGBoost、mlxtend、Matplotlib。
2. 将脚本里的开发机绝对路径改为自己的 `train.csv`、`test.csv` 和输出位置。
3. 核对历史 `sklearn.preprocessing.data` 等导入路径的兼容性；当前版本不一定支持。
4. 检查需要运行的实验段落，确保训练/验证切分及特征处理一致，再执行 `python model.py`。

## 限制

仓库没有锁定依赖，也没有开箱即用的预测服务。部分实验/输出代码位于注释中。历史 `output.csv` 不能当作当前模型验收；比较模型时需保留独立验证集并记录评价指标。
