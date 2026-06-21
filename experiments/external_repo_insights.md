# 外部天气分类仓库调研与后续策略

> 更新日期：2026-06-22
> 目的：从外部天气图像分类项目中提炼对本项目后续有用的信息。
> 原则：只吸收策略和实验启发，不复制外部实现代码。

## 调研来源

| 仓库 | 本次查看版本 | 主要参考内容 | 价值判断 |
|------|--------------|--------------|----------|
| [berkgulay/weather-prediction-from-image](https://github.com/berkgulay/weather-prediction-from-image) | `0696080` / `development` | README、手工天气特征、天空区域裁剪、传统分类器 | 适合作为错误分析和数据诊断参考，不适合作为主模型路线 |
| [Algolzw/WeatherClassification](https://github.com/Algolzw/WeatherClassification) | `bdb20be` / `master` | 比赛经验、预训练模型、label smoothing、mixup/cutout/cutmix、RandAugment、FixRes、TTA、融合 | 最有参考价值，尤其是天气图像增强和分辨率策略 |
| [ricgao/paper_list_weather_classification](https://github.com/ricgao/paper_list_weather_classification) | `08b2e97` / `master` | 单图天气识别论文和数据集列表 | 适合补充数据源、论文背景和官方数据发布后的对照方向 |
| [nicku-a/Weather_Classification](https://github.com/nicku-a/Weather_Classification) | `8d5120e` / `main` | Mendeley 四分类数据、VGG16 transfer learning、precision/recall 结果展示 | 可作小数据 sanity check，不能把高 accuracy 直接外推到比赛 |

## 总体结论

这些仓库给我们的核心启发不是“换一个现成模型”，而是进一步确认当前路线应该从大范围搜索转向更谨慎的泛化验证。

最值得吸收的方向：

- 天气分类对图像几何和颜色变化敏感，增强策略必须克制。
- 分辨率策略值得做小规模验证，尤其是先训练主干，再冻结大部分层做高分辨率 head/BN 微调。
- TTA 和 ensemble 可以提升上限，但必须受 CPU 推理预算约束。
- 手工天气特征不适合替代 CNN，但适合解释错误样本和检查数据分布。
- 论文/公开数据集适合做背景和预训练参考，不能替代官方数据集上的验证。

## 对各仓库的具体启发

### 1. Algolzw/WeatherClassification

这是四个仓库中对我们最有参考价值的一个。它来自天气识别比赛场景，README 明确提到：先用预训练 SOTA 模型找合适 backbone，再逐步试 label smoothing、mixup、cutout、cutmix、RandAugment、optimizer、scheduler，并排除无效 trick。

对我们有用的点：

- **增强要考虑天气任务特性。** 该仓库特别提醒天气图片不适合上下翻转和强旋转，因为这会改变天空、地面、雨雪等空间结构。我们当前默认增强里已有轻度 rotation，官方数据集到位后应验证“去掉 rotation”是否更稳。
- **Cutout/MixUp/CutMix 不能照搬 ImageNet 习惯。** 天气标签往往依赖全局色调、天空比例、地面积雪、湿润路面等线索。大块遮挡或强混合可能让标签语义变脏。若要试 cutout，应优先试多个小块、低强度版本。
- **FixRes 类策略值得小成本验证。** 该仓库提到先用较小分辨率训练，再冻结前面层，只用更大分辨率训练最后层和 BN。我们的当前主线已经在 320 上较稳，后续可以试一个受控的 `320 -> 384 head/BN fine-tune`，而不是直接大规模重训 384。
- **TTA 和 logits 融合要作为离线评估工具。** 该仓库实现了 TTA、投票和 logits 汇总。我们已经有 ensemble eval，后续可以补一个单模型 TTA 评估脚本，但最终是否提交取决于 CPU benchmark。
- **复杂 stacking 不优先。** 该仓库也提到 stacking 训练复杂且慢，最终只融合少量模型。我们的 `exp_052` 已经说明双模型带来的 macro F1 增益很小，继续做复杂 stacking 不划算。

对我们当前结果的对应关系：

- 我们已经验证 LabelSmoothing 更像 rainy 互补分支，而不是主模型。
- 我们已经验证 checkpoint averaging / EMA 比继续普通单 checkpoint 搜索更有效。
- 外部经验支持我们暂缓强增强和大规模 backbone 搜索。

### 2. berkgulay/weather-prediction-from-image

这个仓库更偏传统图像特征和早期 CNN。它提取了亮度、对比度、雾霾程度、锐度、颜色直方图、灰度强度比例等手工特征，并尝试 SVM、RandomForest、DecisionTree 和 CNN。

对我们有用的点不是模型本身，而是**错误分析维度**：

- rainy、cloudy、snowy 的混淆常常和亮度、对比度、白色区域比例、图像清晰度有关。
- 颜色直方图和强度直方图可以作为数据分布检查工具，帮助判断官方数据和当前本地数据是否同分布。
- 它的 sky-aware crop 思路说明天气分类有明显的区域依赖：天空区域、地面区域、道路湿润或积雪区域都可能影响预测。

建议转化为我们的低成本分析任务：

- 增加一个 `reports/` 级别的数据诊断脚本，统计每类图片和错误样本的亮度、饱和度、白色像素比例、边缘密度。
- 对比官方训练集、官方验证 split、本地旧数据的这些统计量，判断是否存在明显 domain shift。
- 对 rainy/snowy 错误样本按这些特征分桶，看错误是否集中在“低亮度但无雨滴”“高白色比例但非雪”等模式。

不建议做的事：

- 不建议重新走 SVM/RF/手工特征主模型路线。
- 不建议把 sky crop 直接接入训练预处理，除非错误分析证明背景区域严重干扰。

### 3. ricgao/paper_list_weather_classification

这个仓库是天气分类论文和数据集列表。它列出了 Image2Weather、RSCM、MWIDataset、SP-Weather、Multi-task Weather 等方向。

对我们有用的点：

- 官方数据发布后，可以用这些公开数据集的类别定义和图像来源来判断官方数据更接近哪种分布。
- 如果官方数据很小，可以考虑只把公开数据作为预训练或补充训练来源，但必须保留干净 holdout，防止数据源噪声拖累最终模型。
- 多任务天气识别、区域选择模型等论文说明：天气分类不总是纯全局分类，局部区域和天气属性可能很重要。

短期建议：

- 不新增论文复现任务。
- 只把这些论文和数据集作为官方数据发布后的背景对照。
- 若官方数据类别更细或混淆更严重，再考虑区域注意力、属性辅助任务或错误样本可视化。

### 4. nicku-a/Weather_Classification

这个仓库使用 Mendeley 四分类天气数据，基于 VGG16 卷积特征和全连接分类器，报告了很高的测试准确率。

对我们有用的点：

- 小数据集上的 transfer learning 可以很快得到高 accuracy，适合作为教学或 pipeline sanity check。
- 它展示了 precision、recall、错误预测和低置信度样本，这对我们当前 error sample 分析方向有参考价值。

需要警惕的点：

- Mendeley 数据只有约千张级别，且类别是 `Cloudy / Rain / Shine / Sunrise`，和我们当前 `cloudy / rainy / snowy / sunny` 不完全一致。
- 98% 级别 accuracy 不代表泛化到比赛数据也能达到同等效果。
- VGG16 特征路线对我们当前 ConvNeXt 主线没有明显替代价值。

建议：

- 保留它作为“低成本 sanity check”的参考，不作为后续主实验方向。
- 我们更应该补充 confidence 分析：看错误样本是否低置信度、ensemble 是否主要修正低置信度样本。

## 对我们后续工作的优先级建议

### P0：官方数据发布后必须先做

1. **重新做数据审计**
   - 类别分布。
   - 重复图和近重复图。
   - train/val 泄漏。
   - 图片尺寸、亮度、饱和度、白色比例等基础统计。

2. **重建验证策略**
   - 使用 stratified split。
   - 保留最终 holdout。
   - 不把所有调参都压在同一个 validation set 上。

3. **复验当前主候选**
   - 单模型 checkpoint averaging。
   - EMA。
   - 双模型 rainy 互补 ensemble。
   - CPU benchmark。

### P1：值得新增的小实验

| 方向 | 实验建议 | 目的 | 成本 |
|------|----------|------|------|
| 天气敏感增强 | 去掉 rotation，只保留 resize crop、horizontal flip、轻 color jitter | 验证几何增强是否伤害天气语义 | 低 |
| 小块 cutout | 多个小块、低遮挡比例，不试大块 cutmix | 验证局部遮挡能否提升泛化且不破坏标签 | 中 |
| FixRes 微调 | `320` 主模型基础上，冻结大部分 backbone，用 `384` 只微调 head/BN | 验证高分辨率是否能补细节 | 中 |
| 单模型 TTA | center crop + horizontal flip logits average | 评估推理时增强收益和 CPU 成本 | 低 |
| confidence 分析 | 输出正确/错误样本置信度分布 | 判断 ensemble 是否主要修正低置信度样本 | 低 |
| 手工特征诊断 | 亮度、饱和度、白色比例、边缘密度统计 | 判断 domain shift 和错误模式 | 低 |

### P2：有条件再做

- teacher / pseudo-label：只有在官方规则允许使用无标签测试集或额外数据时再考虑。
- EfficientNet / ResNeXt 补充分支：只有当官方数据上 ConvNeXt 单模型不稳，或 CPU 预算允许做互补 ensemble 时再跑。
- 区域注意力或 sky/ground crop：只有错误样本显示背景区域强干扰时再考虑。

### P3：不建议优先投入

- 大规模 backbone 搜索。
- 强 RandAugment、强 rotation、vertical flip。
- 大块 CutMix 或高强度 MixUp。
- 复杂 stacking。
- 用小数据集 VGG16 结果替代当前主线判断。
- 手工特征主模型。

## 可以加入队列的后续任务草案

这些任务不需要立刻跑，适合官方数据发布后再进入队列：

| ID 草案 | 任务 | 说明 |
|--------|------|------|
| `ext_001` | Conservative Augmentation Ablation | 对比当前增强 vs 去 rotation 的天气敏感增强 |
| `ext_002` | Small Cutout Ablation | 多小块 cutout，观察 rainy/snowy 是否改善 |
| `ext_003` | FixRes Head/BN Fine-tune | 从当前单模型候选出发做 `320 -> 384` 微调 |
| `ext_004` | Single-model TTA Eval | 只做离线评估和 CPU benchmark，不默认进入提交 |
| `ext_005` | Weather Feature Diagnostics | 生成亮度、饱和度、白色比例、边缘密度的分布和错误样本统计 |
| `ext_006` | Confidence Report | 分析正确/错误、单模型/ensemble 的置信度差异 |

## 最终建议

外部仓库的经验整体支持我们当前判断：这一阶段不应该继续盲目扩实验矩阵。更合理的后续工作是等官方数据集发布后，先做数据审计和验证集重建，再复验当前 EMA/checkpoint averaging 主线。

如果还有额外精力，优先补低成本分析工具，而不是继续增加训练任务。最值得补的是：

1. 官方数据分布和错误样本诊断。
2. 天气敏感增强消融。
3. FixRes 风格的高分辨率 head/BN 微调。
4. 单模型 TTA 和 ensemble 的 CPU 成本评估。

这样能把外部经验转化为可验证的实验，而不是把别的比赛中的 trick 直接迁移过来。
