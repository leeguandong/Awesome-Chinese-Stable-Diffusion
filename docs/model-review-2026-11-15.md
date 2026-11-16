# 图像模型进展复核记录 · 2026-11-15

本轮针对 README 中的近期图像生成、编辑、分层与评测资源查漏。以官方发布、作者代码仓库、模型卡和实际文件目录为依据；检查日期、发布日、仓库修改日分别处理。较早发布但此前缺失的资源标为补录，不作为 11 月新品。

## 本轮补齐与更正

| 范围 | 复核结果 | 主要依据 |
| --- | --- | --- |
| OpenAI | 补入 11 月 4 日 Images 2.6；官方 SDK 已列出 Flare / Sunburst。完整规格未核实，不填最大分辨率或架构 | [官方 RSS](https://openai.com/news/rss.xml)、[SDK 型号定义](https://github.com/openai/openai-python/blob/main/src/openai/types/image_model.py) |
| ByteDance Seed | 补入 11 月 9 日 Seedream 5.5：多轮编辑、原生 RGBA、中英文小字；5.5 Pro 仍是发布预告 | [官方发布](https://seed.bytedance.com/en/blog/more-realistic-images-more-stable-editing-across-100-rounds-introducing-seedream-5-5) |
| MiniMax | 确认 H3-Image 为 11 月 11 日官宣，许可日期为 11 月 10 日；6 步、约 16MP、最多 16 张参考图与许可限制分开记录 | [发布](https://www.minimax.io/blog/minimax-h3-image)、[模型卡与文件](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) |
| Qwen | 补入 2.1-Pro 托管服务、两个独立 PE 模型、Qwen-Image-Bench 与 Q-Judger；开放 2.1/2.1-Turbo 与托管 Pro 不混用参数 | [官方代码](https://github.com/QwenLM/Qwen-Image-2.1)、[接口目录](https://help.aliyun.com/zh/model-studio/qwen-image-generation-and-editing-api-reference)、[评测仓库](https://github.com/QwenLM/Qwen-Image-Bench) |
| InternLumina | 更正“仅昇腾开放”：已提供 Ascend/NVIDIA 权重、SFT 训练代码、加速路径及报告；参数更新为 16.9B / 约 1.4B 激活 | [官方 README](https://github.com/InternLM/InternLumina-U2)、[报告](https://arxiv.org/abs/2610.20818) |
| Black Forest Labs | 补齐 FLUX.2 Klein 4B/9B、Base、9B KV；4B Apache-2.0 与 9B 非商业许可分别记录，避免和 FLUX 3 Klein 混淆 | [FLUX.2](https://github.com/black-forest-labs/flux2)、[官方模型目录](https://huggingface.co/black-forest-labs)、[发布列表](https://bfl.ai/blog) |
| BRIA | 补入 Fibo 2.0 Dev / Edit Turbo：8B、结构化 VGL、4 步编辑；Turbo 不作为文生图模型推荐 | [Dev](https://huggingface.co/briaai/Fibo-2.0-dev)、[Edit Turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo) |
| 美团 LongCat | 补齐独立 Edit 与 2 月 3 日发布的 8 步 Edit-Turbo；质量分数不直接继承到 Turbo | [官方发布记录](https://github.com/meituan-longcat/LongCat-Image)、[Turbo](https://huggingface.co/meituan-longcat/LongCat-Image-Edit-Turbo) |
| HiDream | 补齐 O1-Image 的 28 步 Dev 与文生图专用 Dev-2604；官方编辑推荐完整版 | [Dev 模型卡](https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev) |
| vivo | 确认 BluePixel-Turbo 11 月 3 日 SATP 检查点的中间版本限制；VINS-4KEval 509 例与 ++ 3,305 例不混用 | [模型卡](https://huggingface.co/openvivo/BluePixel-Turbo)、[项目页](https://tech.vivo.com/BluePixel/)、[数据](https://huggingface.co/datasets/openvivo/VINS-120K) |
| Krea / Linum | 补录 Krea 2 Raw/Turbo；P-JiT 21K 的 11 月 4 日发布、训练成本和不支持文字生成的限制已核实 | [Krea](https://github.com/krea-ai/krea-2)、[P-JiT 官方说明](https://www.linum.ai/field-notes/pjit-21k) |
| OpenMOSS | 官方模型目录仍为 Base/SFT；补充社区 8 步 DMD 和低显存模式，不把社区编辑模型当官方新版本 | [官方目录](https://huggingface.co/OpenMOSS-Team)、[官方 README](https://github.com/OpenMOSS/MOSS-Image) |
| inclusionAI | Ming-Image Design/Layer 与 LLaDA-Image 已收录；补入不同技术路线的 Ming-UniVision | [官方目录](https://huggingface.co/inclusionAI)、[Ming-UniVision](https://huggingface.co/inclusionAI/Ming-UniVision-16B-A3B) |
| 阶跃 / VectorSpaceLab / Alpha-VLLM | 补录 Step1X-Edit-v1p2、OmniGen2、Lumina-DiMOO；这些是较早发布的编辑或统一模型 | [Step1X](https://huggingface.co/stepfun-ai/Step1X-Edit-v1p2)、[OmniGen2](https://huggingface.co/OmniGen2/OmniGen2)、[DiMOO](https://huggingface.co/Alpha-VLLM/Lumina-DiMOO) |
| 百度 | 补入 ERNIE-Image-Aes 到评测工具；不归入生图模型汇总 | [官方模型卡](https://huggingface.co/baidu/ERNIE-Image-Aes) |

## 核查后保留的边界

- **Google**：检查了官方 Gemini 发布 RSS；11 月 5 日 Gemini Omni 1.2 Flash 的原文聚焦视频生成与视频编辑，不据其名称升级 Nano Banana 图像型号。来源：[官方文章](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemini-omni-1-2-flash/)。
- **万相**：百炼图像接口仍列 Wan2.7-Image / Pro；Pro 的 4K 条件限定于非组图文生图，其余编辑/组图场景不套用。来源：[接口文档](https://help.aliyun.com/zh/model-studio/wan-image-generation-and-editing-api-reference)。
- **混元**：HunyuanImage-3.0 Instruct / Distil 的 1 月 26 日发布已在原条目记录，当前官方 README 推荐 Distil 8 步；不再重复作为 11 月新品。来源：[官方 README](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0)。
- **Z-Image / Ovis**：公开目录检索未发现可据此替换现有主版本的新官方图像型号；社区量化、合并权重和同名仓库不自动算官方基模升级。来源：[Z-Image 目录](https://huggingface.co/Tongyi-MAI)、[Ovis 权重](https://huggingface.co/ATH-MaaS/Ovis-Image-7B)。
- **基模收录边界（2026-11-16 更正）**：TailSteer 是 SenseNova-U1.5 的专项后训练检查点，已从模型汇总、独立模型条目与近期基模进展中移除；完整微调权重或发布账号不能作为新基模的判据。Lightning LoRA 仅保留在 Qwen-Image-2.1 的部署说明中，Qwen-Image-Flash 并入 Qwen-Image 的衍生加速说明，FLUX.2 Klein KV 合并到同族汇总。MOSS-Image Base/SFT 已有基模条目；Add & Preserve 与 TailSteer-Source 分别按评测、数据集收录。
- **Swift-Image / InnoText**：本轮代码仓库与模型目录检索没有确认作者发布的可用权重，保持论文/待开放分类；搜索无结果并不能证明永久未开放。
- **JuZhou**：发现 JuZhou-V1 页面，但其 README 明确是占位，实际文件只有 README 与仓库配置，不能改成权重已开放。来源：[页面及文件](https://huggingface.co/HswAI2026/JuZhou-V1/tree/main)。

## 来源与验证口径

- Hugging Face 直连在本次环境不可用，通过 HF Mirror 读取对应公开模型卡、API 元数据和文件目录；README 对外链接保留官方地址。受许可访问限制的模型只读取公开卡片，未申请或接受下载授权。
- OpenAI 发布正文与文档请求被站点拦截，Images 2.6 的发布日期与能力概述依据官方 RSS，型号依据官方 SDK；未据第三方宣传补齐缺失规格。
- 官方榜单和时延为发布方测量，本轮没有运行模型或复现分数。P-JiT 成本为计算成本，低显存卸载数字不是完整流水线常驻显存。
- 本记录覆盖本轮实际检查的主要发布方和已收录模型族，并非全网论文、社区微调、量化衍生或视频模型的完整目录。
