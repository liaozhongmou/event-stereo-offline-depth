# 已读取输入的语义与来源

尺寸统一为 346×260；坐标 x 为列、y 为行。所有 NPZ 的实际键、dtype、shape 和文件 SHA-256 见 ARTIFACT_INVENTORY.json，不根据后缀自动转换。

| 输入 | 时间/空间/数值定义 | 本轮角色与来源边界 |
|---|---|---|
| `indoor{1,2,3}_left1.npz/right1.npz` | 双目事件整数像素，记录时间；校正状态为作者确认 | 历史开发输入；本轮未重跑匹配。原始校正脚本及当时输出链待恢复 |
| 同目录 `left2/right2` | 已保存 LIF 输出事件 | 开发中间输出；精确历史命令/源版本待恢复，未默认当作可复现基准 |
| `*_gt_disp.npz` | `disp[frame,y,x]`；`ts` 是 epoch 秒；Indoor2/3 另有相同的 timestamps | 开发 GT；按稿件记录乘 0.8819476 一次。该约定与源文件哈希已冻结；转换起源待原脚本核实 |
| `*_gt_depth.npz` | `depth[frame,y,x]`，相对秒 ts，跨度与 disparity 档不同 | 清点并散列；未用于本轮评分或替换时间轴 |
| `matches9/10/11.npz` | x_L/y_L，epoch 秒 t，px disparity；score/resid/support 先于行中值输出计算 | 开发原匹配的置信度诊断；meta_json 参数相同，保存输出有 row_bimed_k=10。resid 是 px² 的加权均方残差 |
| `matches*_1.npz` | 像素/秒/px 视差，无 confidence | 已保存趋势输出；Indoor2 行数及起点与当前 raw 不对应，不能默认同次流水线 |
| `matches*_2.npz` | 原行及离线填充值；重复记录，H/W 明确 | Indoor1 用于历史核对；先按固定 GT 帧像素去重。补全行可能替换某单位的最近时间关联 |
| `tnnls_2012/matches_tnnls2012.npz` | x_L/y_L，epoch 秒 t，px disparity，并含匹配配对字段 | 本地早期实现输出；Neural2016 缺失，未将它与本方法组成新的优势比较 |
| 根 `_tmp_scaled.npz` | 95,833 行，t=0..343.9；含 frame 字段 | 缺序列/变换来源，未强行认作 EMatch 或 Neural2016；不参与评分 |
| `tmp/lif_verify/left_filtered.npz/right_filtered.npz` | 事件中间结果 | 缺原运行记录；只列入全工作区检索清单，未参与评分 |
| `camchain-imucam-indoor_flying.yaml` | equidistant；左投影 fx=199.6530123165822 px；右 tx=-19.941771812941038 px·m | 当前标定快照已保存和散列；未再次校正输入，未把当前文件自动认定为当时处理版本 |
| `original/*.py` | 当前可见实现的修改前快照 | 本轮源码基线 01f39a2；不是已恢复的原历史运行源码版本 |

本轮没有训练或参数学习。Indoor1 提供无 GT 的置信度阈值分位数；Indoor1–3 全部属于历史开发数据。没有新的独立验证数据或已恢复的 Indoor4/EMatch 模型输入。
