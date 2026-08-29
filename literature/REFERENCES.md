# PA-STFed 文献依据与迁移边界

本文件只记录已核验 DOI 的论文，以及它们在 SmartDS 项目中可以借用的模块。论文中的数据集、指标和实验数值不能直接替换本项目结果。

## 1. SmartDS 数据来源

Mateo, C., Postigo, F., de Cuadra, F., Gómez San Román, T., Elgindy, T., Dueñas, P., Hodge, B.-M., Krishnan, V., and Palmintier, B., “Building Large-Scale U.S. Synthetic Electric Distribution System Models,” *IEEE Transactions on Smart Grid*, vol. 11, no. 6, pp. 5301--5313, 2020. DOI: [10.1109/TSG.2020.3001495](https://doi.org/10.1109/TSG.2020.3001495).

该工作是 SMART-DS 合成配电系统模型的正式学术来源。当前压缩包的可审计事实为 273 个节点、92 个非零负荷节点、57 个原始连通分量和 35040 个 15 分钟序列点。SmartDS 是“realistic but not real”的合成网络，不能写成真实台区实测结果。

## 2. 配电台区联邦依据

Feng, C., Shao, L., Wang, J., Zhang, Y., and Wen, F., “Short-term Load Forecasting of Distribution Transformer Supply Zones Based on Federated Model-Agnostic Meta Learning,” *IEEE Transactions on Power Systems*, vol. 40, no. 1, pp. 31--45, 2025 (Early Access 2024). DOI: [10.1109/TPWRS.2024.3393017](https://doi.org/10.1109/TPWRS.2024.3393017).

论文直接以 distribution-transformer supply zones 为客户端，研究联邦短期负荷预测、客户端漂移和个性化。它支持本项目把“配电网联邦”写成问题背景，并说明 Non-IID 下 FedAvg/FedProx 需要与个性化策略比较。论文采用 FMAML 与 stochastic controlled averaging；由于当前大创代码不实现 MAML，不能宣称复现其算法，只迁移问题定义和对照逻辑。

## 3. 多节点时空预测依据

Zhao, P., Hu, W., Cao, D., Zhang, Z., Liao, W., Chen, Z., and Huang, Q., “Enhancing Multivariate, Multi-Step Residential Load Forecasting with Spatiotemporal Graph Attention-Enabled Transformer,” *International Journal of Electrical Power & Energy Systems*, 2024, vol. 160, 110074. DOI: [10.1016/j.ijepes.2024.110074](https://doi.org/10.1016/j.ijepes.2024.110074).

论文在多住宅负荷节点上联合使用空间注意力、时间注意力、门控时空融合和多步预测。它支持 PA-STFed 的编码器设计、8:1:1 时间切分、Adam/早停和多随机种子复核。其住宅数据与 SmartDS 不是同一数据，不能直接比较百分比精度。

Zhao, H., Wu, Y., Ma, L., and Pan, S., “Spatial and Temporal Attention-Enabled Transformer Network for Multivariate Short-Term Residential Load Forecasting,” *IEEE Transactions on Instrumentation and Measurement*, vol. 72, pp. 1--11, 2023. DOI: [10.1109/TIM.2023.3305655](https://doi.org/10.1109/TIM.2023.3305655).

该文证明住宅/微电网负荷的空间相关性可以与时间注意力联合建模，并采用无先验地理信息的空间关系学习。它是“多节点负荷而非单节点序列”这一建模选择的补充依据。

## 4. 个性化联邦依据

Wu, H. and Xu, Z., “Multi-Energy Load Forecasting in Integrated Energy Systems: A Spatial-Temporal Adaptive Personalized Federated Learning Approach,” *IEEE Transactions on Industrial Informatics*, 2024, vol. 20, no. 10, pp. 12262--12274. DOI: [10.1109/TII.2024.3417297](https://doi.org/10.1109/TII.2024.3417297).

该文把空间-时间 Transformer 与自适应本地聚合结合，说明个性化联邦可以缓解负荷客户端异构。研究对象是综合能源系统，不应写成 SmartDS 配电网的直接复现。

Bose, S., Zhang, Y., and Kim, K., “Addressing Heterogeneity in Federated Load Forecasting with Personalization Layers,” arXiv:2404.01517, 2024. [arXiv](https://arxiv.org/abs/2404.01517).

其核心对照是共享 LSTM 表征、私有预测层和完全本地模型。本项目的 `personalized_head` 实验只借用这一可解释的参数分离思想，不声称提出新的联邦优化算法。

## 5. 联邦优化基线

McMahan, B., Moore, E., Ramage, D., Hampson, S., and y Arcas, B. A., “Communication-Efficient Learning of Deep Networks from Decentralized Data,” *Proceedings of AISTATS*, vol. 54, pp. 1273--1282, 2017. [PMLR](https://proceedings.mlr.press/v54/mcmahan17a.html).

FedAvg 是本项目的基础聚合对照。

Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., and Smith, V., “Federated Optimization in Heterogeneous Networks,” *Proceedings of Machine Learning and Systems*, vol. 2, pp. 429--450, 2020. [MLSys](https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html).

FedProx 通过近端项抑制本地更新偏离。本项目固定 `mu=0.005` 作为预注册对照，不在测试集上选择它。

## 6. 如何组合而不夸大

最终论文将 SmartDS 作为合成配电网案例，将 Zhao 等的多节点时空编码、Feng 等的配电台区联邦问题、Wu--Xu/Bose 的个性化参数分离组合成一个可复现工程框架。贡献表述应是“在 SmartDS 上的案例实现与系统比较”，而不是“首次提出新的联邦算法”或“优于所有现有方法”。
