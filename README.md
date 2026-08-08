<h1 align="center">
Awesome-Chinese-Stable-Diffusion
</h1>
<p align="center">
<b>Awesome Chinese Image Generation Resources / 中文图像生成与编辑资源精选</b>
</p>
<p align="center">
  <a href="https://github.com/leeguandong/Awesome-Chinese-Stable-Diffusion/stargazers"> <img src="https://img.shields.io/github/stars/leeguandong/Awesome-Chinese-Stable-Diffusion.svg?style=popout-square" alt="GitHub stars"></a>
  <a href="https://github.com/leeguandong/Awesome-Chinese-Stable-Diffusion/issues"> <img src="https://img.shields.io/github/issues/leeguandong/Awesome-Chinese-Stable-Diffusion.svg?style=popout-square" alt="GitHub issues"></a>
  <a href="https://github.com/leeguandong/Awesome-Chinese-Stable-Diffusion/forks"> <img src="https://img.shields.io/github/forks/leeguandong/Awesome-Chinese-Stable-Diffusion.svg?style=popout-square" alt="GitHub forks"></a>
</p>

> A curated list of Chinese image generation and editing resources, covering open and closed models, benchmarks, evaluation tools, and datasets.

本项目旨在收集和梳理中文图像生成与编辑相关资源。除 Stable Diffusion 生态外，也收录 DiT、自回归、MoE 和统一多模态模型，以及中文文字渲染相关的评测与数据集。

如果本项目能给您带来一点点帮助，麻烦点个⭐️吧～

同时也欢迎大家贡献本项目未收录的开源模型、应用、数据集等。提供新的仓库信息请发起PR，并按照本项目的格式提供仓库链接、star数，简介等相关信息，感谢~

## 目录

- [1. 中文图像生成与编辑模型](#1-中文图像生成与编辑模型)
  - [1.1 模型汇总](#11-模型汇总)
  - [1.2 开源模型](#12-开源模型)
  - [1.3 闭源模型](#13-闭源模型)
  - [1.4 论文与待开放模型](#14-论文与待开放模型)
- [2. 测评](#2-测评)
  - [2.1 评测基准](#21-评测基准)
  - [2.2 评测工具](#22-评测工具)
  - [2.3 排行榜](#23-排行榜)
- [3. 数据集](#3-数据集)
  - [3.1 开源训练数据集](#31-开源训练数据集)
  - [3.2 评测/标注数据集](#32-评测标注数据集)
- [Star History](#star-history)
- [License](#license)

## 1. 中文图像生成与编辑模型

### 1.1 模型汇总

| 模型 | 参数量 | 架构 | 文本编码器 | 最大分辨率 | 中文文字渲染 |
|------|--------|------|-----------|-----------|-------------|
| SkyPaint | - | UNet (SD) | CLIP (中英) | 512 | - |
| Pai-Diffusion | - | UNet (SD) | Chinese CLIP | - | - |
| 中文SD-通用 | - | UNet (SD 2.1) | Chinese CLIP ViT-H | - | - |
| 通义-文生图 | 5B | Multi-stage UNet | CLIP | 1024 | - |
| Taiyi | - | UNet (SD 1.4) | Chinese RoBERTa CLIP | 512 | - |
| Taiyi-XL-3.5B | 3.5B | SDXL | Bilingual CLIP | 1024 | - |
| AltDiffusion | - | UNet (SD 1.4) | AltCLIP | 512 | - |
| VisCPM-Paint | 10B+ | UNet (SD 2.1) | CPM-Bee (10B) | - | - |
| WuKong-HuaHua | - | UNet (SD) | Chinese CLIP | 768 | - |
| PanGu-Draw | 5B | UNet | Chinese CLIP | 1024 | - |
| MiaoBi | - | UNet (SD 1.5) | Chinese | 512 | - |
| 混元DiT | 1.5B | DiT | CLIP + T5 | 1024+ | - |
| Kolors | - | SDXL | ChatGLM3-6B | 1024 | 支持 |
| UniT2IXL | - | SDXL | Chinese CLIP + LM | 1024 | - |
| CogView4 | 6B | DiT (Share-param) | GLM-4-9B | 2048 | 支持 |
| HiDream-I1 | 17B | MoE DiT | CLIP + T5 + Llama-3.1-8B | 1024+ | - |
| Qwen-Image | 20B | MMDiT | Qwen2.5-VL-7B | 2048+ | 支持 |
| Wan2.1 | 14B | DiT | umT5 | 720p | - |
| Wan2.2 | 27B (14B active) | MoE DiT | - | 720p+ | - |
| Hunyuan2.1 | 17B | DiT | MLLM + ByT5 | 2K | 支持 |
| Hunyuan3 | 80B (13B active) | AR + MoE | GLM-based | 1024+ | 支持 |
| Z-Image | 6B | S3-DiT | - | 2048 | 支持 |
| Ovis-Image | 7B | MMDiT | - | 1024+ | 支持 |
| LongCat-Image | 6B | MM-DiT + Single-DiT | VLM | 1024+ | 支持 |
| FLUX.2 | 32B 主干（另用 24B TE） | DiT | Mistral-Small-3.2-24B | 2048 | - |
| GLM-Image | 16B（9B AR + 7B DiT） | AR + DiT | GLM-4-9B | 2K | 支持 |
| Qwen-Image-2512 | 20B | MMDiT | Qwen2.5-VL-7B | 2048+ | 支持 |
| Qwen-Image-2.0 | - | - | - | 2K | 支持 |
| Z-Image-Turbo | 6B | S3-DiT (8-step distill) | - | 2048 | 支持 |
| BAGEL-7B-MoT | 14B (7B active) | MoT | - | 1024+ | - |
| ERNIE-Image | 8B | DiT (single-stream) | ERNIE-based | 2K | 支持 |
| ERNIE-Image-Turbo | 8B | DiT (8-step distill) | ERNIE-based | 2K | 支持 |
| Wan2.7-Image / Pro | - | - | - | 2K / 4K（Pro 文生图） | 支持 |
| Qwen-Image-2.0-Pro | - | - | - | - | 支持 |
| Seedream 5.0 Pro | - | - | - | - | 支持多语言 |
| HiDream-O1-Image | 8B | UiT (Pixel-native) | 无外置文本编码器（Gemma-4-31B 提示词代理） | 2048 | - |
| NextStep-1 | 14B + 157M | AR + Flow Matching | 内置 LM head | 1024+ | - |
| Boogu-Image-0.1 | 10B | DiT | Qwen3-VL-8B | 2048 | 支持 |
| JoyAI-Image / Edit | 24B (8B MLLM + 16B MMDiT) | MLLM + MMDiT | 8B MLLM | 1024 | 支持 |
| JuZhou 1.0 | 0.387B | UNet + Rectified Flow | Chinese CLIP | 1024 | 支持 |
| Mage-Flow / Mage-Flow-Edit | 4B | Native-Resolution MMDiT | Qwen3-VL | 2048 | 支持 |
| Qwen-Image-Flash | 20.43B DiT（全流水线 28.85B） | MMDiT (4-step DMD2) | Qwen2.5-VL | 未公开（官方仅测 1024） | 未评测（英文蒸馏） |
| Qwen-Image-3.0 / Qwen-Image-3.0-Pro | - | - | - | 2K | 支持（12 种语言） |

### 1.2 开源模型

* **SkyPaint**：
  * 地址：https://github.com/SkyWorkAIGC/SkyPaint-AI-Diffusion ![](https://img.shields.io/github/stars/SkyWorkAIGC/SkyPaint-AI-Diffusion.svg)
  * 简介：SkyPaint文本生成图片模型主要由两大部分组成，即提示词文本编码器模型和扩散模型两大部分。因此我们的优化也分为两步： 首先，基于[OpenAI-CLIP](https://github.com/openai/CLIP)优化了提示词文本编码器模型使得SkyPaint具有中英文识别能力， 然后，优化了扩散模型，使得SkyPaint具有现代艺术能力可以产生高质量图片。
  
* **Pai-Diffusion**：
  * 地址：https://github.com/alibaba/EasyNLP ![](https://img.shields.io/github/stars/alibaba/EasyNLP.svg)
  * 简介：由于现有Diffusion模型主要使用英文数据进行训练，如果直接使用机器翻译将英文数据翻译成中文进行模型训练，因为中英文在文化和表达上具有很大的差异性，产出的模型通常无法建模中文特有的现象。此外，通用的StableDiffusion模型由于数据源的限制，很难用于生成特定领域、特定场景下的高清图片。PAI-Diffusion系列模型由阿里云机器学习（PAI）团队发布并开源，除了可以用于通用文图生成场景，还具有一系列特定场景的定制化中文Diffusion模型，包括古诗配图、二次元动漫、魔幻现实等。在下文中，我们首先介绍PAI-Diffusion的模型Pipeline架构，包括中文CLIP模型、Diffusion模型、图像超分模型等。

* **中文StableDiffusion-通用领域**：
  * 地址：https://modelscope.cn/models/damo/multi-modal_chinese_stable_diffusion_v1.0/summary
  * 简介：本模型采用的是[Stable Diffusion 2.1模型框架](https://github.com/Stability-AI/generative-models)，将原始英文领域的[OpenCLIP-ViT/H](https://github.com/mlfoundations/open_clip)文本编码器替换为中文CLIP文本编码器[chinese-clip-vit-huge-patch14](https://github.com/OFA-Sys/Chinese-CLIP)，并使用大规模中文图文pair数据进行训练。训练过程中，固定中文CLIP文本编码器，利用原始Stable Diffusion 2.1 权重对UNet网络参数进行初始化、利用64卡A100共训练35W steps。训练数据包括经中文翻译的公开数据集（LAION-400M、cc12m、Open Images）、以及互联网搜集数据，经过美学得分、图文相关性等预处理进行图像过滤，共计约4亿图文对。
  
* **文本到图像生成扩散模型-中英文-通用领域-tiny**：
  * 地址：https://modelscope.cn/models/damo/cv_diffusion_text-to-image-synthesis_tiny/summary
  * 简介：文本到图像生成模型由文本特征提取与扩散去噪模型两个子网络组成。文本特征提取子网络为StructBert结构，扩散去噪模型为unet结构。通过StructBert提取描述文本的语义特征后，送入扩散去噪unet子网络，通过迭代去噪的过程，逐步生成符合文本描述的图像。训练数据包括LAION400M公开数据集，以及互联网图文数据。文本截断到长度64 (有效长度62)，图像缩放到64x64进行处理。模型分为文本特征提取与扩散去噪模型两个子网络，训练也是分别进行。文本特征提取子网络StructBert在大规模中文文本数据上预训练得到。扩散去噪模型则使用预训练StructBert提取文本特征后，与图像一同训练文本到图像生成模型。
  
* **通义-文本生成图像大模型-中英文-通用领域**：
  * 地址：https://www.modelscope.cn/models/damo/cv_diffusion_text-to-image-synthesis/summary
  * 简介：本模型基于多阶段文本到图像生成扩散模型, 输入描述文本，返回符合文本描述的2D图像。支持中英双语输入。文本到图像生成扩散模型由特征提取、级联生成扩散模型等模块组成。整体模型参数约50亿，支持中英双语输入。通过知识重组与可变维度扩散模型加速收敛并提升最终生成效果。训练数据包括LAION5B, ImageNet, FFHQ, AFHQ, WikiArt等公开数据集。经过美学得分、水印得分、去重等预处理进行图像过滤。模型分为文本特征提取、文本特征到图像特征生成、级联扩散生成模型等子网络组成，训练也是分别进行。文本特征提取使用大规模图文样本对数据上训练的CLIP的文本分支得到。文本到图像特征生成部分采用GPT结构，是一个width为2048、32个heads、24个blocks的Transformer网络，利用causal attention mask实现GPT预测。64x64、256x256、1024x1024扩散模型均为UNet结构，在64x64、256x256生成模型中使用了Cross Attention嵌入image embedding条件。为降低计算复杂度，在256扩散模型训练过程中，随机64x64 crop、128x128 crop、256x256 crop进行了multi-grid训练，来提升生成质量；在1024扩散模型中，对输入图随机256x256 crop。

* **Taiyi**：

  * 地址：https://github.com/IDEA-CCNL/Fengshenbang-LM ![](https://img.shields.io/github/stars/IDEA-CCNL/Fengshenbang-LM.svg)

  * 简介：Taiyi-clip：我们遵循CLIP的实验设置，以获得强大的视觉-语言表征。在训练中文版的CLIP时，我们使用[chinese-roberta-wwm](https://link.zhihu.com/?target=https%3A//huggingface.co/hfl/chinese-roberta-wwm-ext)作为语言的编码器，并将[open_clip](https://link.zhihu.com/?target=https%3A//github.com/mlfoundations/open_clip)中的**ViT-L-14**应用于视觉的编码器。为了快速且稳定地进行预训练，我们**冻结了视觉编码器并且只微调语言编码器**。此外，我们将[Noah-Wukong](https://link.zhihu.com/?target=https%3A//wukong-dataset.github.io/wukong-dataset/)数据集(100M)和[Zero](https://link.zhihu.com/?target=https%3A//zero.so.com/)数据集(23M)用作预训练的数据集。在悟空数据集和zero数据集上预训练24轮,在A100x32上训练了6天。

    Taiyi-SD：我们将[Noah-Wukong](https://link.zhihu.com/?target=https%3A//wukong-dataset.github.io/wukong-dataset/)数据集(100M)和[Zero](https://link.zhihu.com/?target=https%3A//zero.so.com/)数据集(23M)用作预训练的数据集，先用[IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese](https://link.zhihu.com/?target=https%3A//huggingface.co/IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese)对这两个数据集的图文对相似性进行打分，取CLIP Score大于0.2的图文对作为我们的训练集。 我们使用[IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese](https://link.zhihu.com/?target=https%3A//huggingface.co/IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese)作为初始化的text encoder，冻住[stable-diffusion-v1-4](https://link.zhihu.com/?target=https%3A//huggingface.co/CompVis/stable-diffusion-v1-4)([论文](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2112.10752))模型的其他部分，**只训练text encoder**，以便保留原始模型的生成能力且实现中文概念的对齐。该模型目前在0.2亿图文对上训练了一个epoch。 我们在 32 x A100 训练了大约100小时。
    补充: clip和sd的微调阶段都只调text encoder部分

* **Taiyi-XL-3.5B**：

  * 地址：https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-XL-3.5B

  * 简介：文生图模型如谷歌的Imagen、OpenAI的DALL-E 3和Stability AI的Stable Diffusion引领了AIGC和数字艺术创作的新浪潮。然而，基于SD v1.5的中文文生图模型，如Taiyi-Diffusion-v0.1和Alt-Diffusion的效果仍然一般。中国的许多AI绘画平台仅支持英文，或依赖中译英的翻译工具。目前的开源文生图模型主要支持英文，双语支持有限。我们的工作，Taiyi-Diffusion-XL（Taiyi-XL），在这些发展的基础上，专注于保留英文理解能力的同时增强中文文生图生成能力，更好地支持双语文生图。

    Taiyi-Diffusion-XL文生图模型训练主要包括了3个阶段。首先，我们制作了一个高质量的图文对数据集，每张图片都配有详细的描述性文本。为了克服网络爬取数据的局限性，我们使用先进的视觉-语言大模型生成准确描述图片的caption。这种方法丰富了我们的数据集，确保了相关性和细节。然后，我们从预训练的英文CLIP模型开始，为了更好地支持中文和长文本我们扩展了模型的词表和位置编码，通过大规模双语数据集扩展其双语能力。训练涉及对比损失函数和内存高效的方法。最后，我们基于Stable-Diffusion-XL，替换了第二阶段获得的text encoder，在第一阶段获得的数据集上进行扩散模型的多分辨率、多宽高比训练。

    我们的机器评估包括了对不同模型的全面比较。评估指标包括CLIP相似度（CLIP Sim）、IS和FID，为每个模型在图像质量、多样性和与文本描述的对齐方面提供了全面的评估。在英文数据集（COCO）中，Taiyi-XL在所有指标上表现优异，获得了最好的CLIP Sim、IS和FID得分。这表明Taiyi-XL在生成与英文文本提示紧密对齐的图像方面非常有效，同时保持了高图像质量和多样性。同样，在中文数据集（COCO-CN）中，Taiyi-XL也超越了其他模型，展现了其强大的双语能力。

    尽管Taiyi-XL可能还未能与商业模型相媲美，但它比当前双语开源模型优越不少。我们认为我们模型与商业模型的差距主要归因于训练数据的数量、质量和多样性的差异。我们的模型仅使用学术数据集和符合版权要求的图文数据进行训练，未使用Midjourney和DALL-E 3等生成数据。XL版本模型，如SD-XL和Taiyi-XL，在1.5版本模型如SD-v1.5和Alt-Diffusion上显示出显著改进。DALL-E 3以其生动的色彩和prompt-following的能力而著称。Taiyi-XL模型偏向生成摄影风格的图片，与Midjourney较为类似，而且 Taiyi-XL 在中英文文生图方面表现更出色。
    
* **AltDiffusion**：

  * 地址：https://github.com/FlagAI-Open/FlagAI ![](https://img.shields.io/github/stars/FlagAI-Open/FlagAI.svg)

  * 简介：AltClip：AltCLIP基于 [OpenAI CLIP](https://link.zhihu.com/?target=https%3A//github.com/openai/CLIP) 训练，训练数据来自 [WuDao数据集](https://link.zhihu.com/?target=https%3A//data.baai.ac.cn/details/WuDaoCorporaText) 和 [LAION](https://link.zhihu.com/?target=https%3A//huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6plus)，训练共有两个阶段。 在平行知识蒸馏阶段，我们只是使用平行语料文本来进行蒸馏（平行语料相对于图文对更容易获取且数量更大）。在双语对比学习阶段，我们使用少量的中-英图像-文本对（一共约2百万）来训练我们的文本编码器以更好地适应图像编码器。

    AltSD：基于 stable-diffusion v1-4 作为初始化，并使用 AltCLIP 或 AltCLIPM9 作为text encoder。在微调过程中，除了跨注意力块的键和值投影层之外，我们冻结了扩散模型中的所有参数。训练数据来自 [WuDao数据集](https://link.zhihu.com/?target=https%3A//data.baai.ac.cn/details/WuDaoCorporaText) 和 [LAION](https://link.zhihu.com/?target=https%3A//huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6plus)。

* **VisCPM-Paint**：

  * 地址：https://github.com/OpenBMB/VisCPM ![](https://img.shields.io/github/stars/OpenBMB/VisCPM.svg)

  * 简介：VisCPM-Paint支持中英双语的文到图生成。该模型使用CPM-Bee（10B）作为文本编码器，使用UNet作为图像解码器，并通过扩散模型训练目标融合语言和视觉模型。在训练过程中，语言模型参数始终保持固定。我们使用[Stable Diffusion 2.1](https://github.com/Stability-AI/generative-models)的UNet参数初始化视觉解码器，并通过逐步解冻其中关键的桥接参数将其与语言模型融合。该模型在[LAION 2B](https://laion.ai/)英文图文对数据上进行了训练。

    与VisCPM-Chat类似，我们发现得益于CPM-Bee的双语能力，VisCPM-Paint可以仅通过英文图文对训练，泛化实现良好的中文文到图生成能力，达到中文开源模型的最佳效果。通过进一步加入20M清洗后的原生中文图文对数据，以及120M翻译到中文的图文对数据，模型的中文文到图生成能力可以获得进一步提升。我们在标准图像生成测试集MSCOCO上采样了3万张图片，计算了常用评估图像生成指标FID (Fréchet Inception Distance)评估生成图片的质量。我们同样提供了两个模型版本，分别为VisCPM-Paint-balance和VisCPM-Paint-zhplus，前者在英文和中文两种语言上的能力较为平衡，后者在中文能力上更加突出。VisCPM-Paint-balance只使用了英文图文对进行训练，VisCPM-Paint-zhplus在VisCPM-Paint-balance基础上增加了20M原生中文图文对数据和120M翻译到中文的图文对数据进行训练。

* **WuKong-HuaHua**：

  * 地址：https://github.com/JeffDing/WuKong-HuaHua ![](https://img.shields.io/github/stars/JeffDing/WuKong-HuaHua.svg)

  * 简介： Wukong-Huahua是基于扩散模型的中文文生图大模型，由华为诺亚团队携手中软分布式并行实验室、昇腾计算产品部联合开发，使用昇思框架(MindSpore)+昇腾(Ascend)软硬件解决方案实现。该模型基于悟空中文多模态数据集训练，具备中文文本-图像生成能力，能够生成多种场景和绘画风格的图像。

    在1.0的基础上Wukong-Huahua模型基于华为MindSpore平台+昇腾硬件910进行大规模多机多卡训练，在新数据集上进行训练升级到2.0版本。相比于原版本，新版本大幅提升画质、艺术性和推理速度，更新内容包括以下3点：1.提升输出分辨率，2.0模型目前可以支持更高分辨率图形输出，从1.0版本的512x512提升到768x768，大图更清晰。2.采用自研Multistep-SDE采样加速推理技术，采样步数从原先的50步采样降到20-30步，加速2-3倍。3.采用自研RLAIF算法，提升生成图片的画质以及艺术性表达。

    悟空画画模型分别由中文文本编码器以及Stable Diffusion生成模型组成。具体的训练方法如下：

    1. 预训练中文图文判别模型，得到一个具有中文图文对齐能力的文本编码器；

    2. 结合Stable Diffusion图像生成模型和第一步训练得到的文本编码器，在悟空中文多模态数据集上进行训练，得到中文文图生成模型——悟空画画模型。

    悟空画画模型的训练依赖于悟空数据集，它是当时已开源的最大规模的中文多模态数据集。我们首先在百度搜索引擎上利用一百万个中文高频文本作为关键词进行图片搜索，获得接近20亿的原始图文对数据，此时这部分数据中包含了大量的噪声。第二步我们对这些原始数据进行多种方式的筛选清洗，主要操作包括：

    - 对图片的尺寸进行过滤，去除边长小于200px或者长宽比超出1/3~3范围的样本
    - 去除文本为无意义的词如 “Image”, “图片”，“照片”等的样本
    - 过滤文本长度过短，文本出现频次过高（如“如下图所示”等描述文本）的样本
    - 过滤文本中包含隐私/敏感词的样本
    
    最终我们经过过滤得到了一亿较高质量中文图文对。进一步地，在训练悟空画画模型时，我们对悟空数据集的数据根据图文匹配分数、水印分数以及[艺术性分数](https://github.com/christophschuhmann/improved-aesthetic-predictor)再次进行筛选，最终获得25M左右的数据进行训练。该部分数据具有较高的图像质量，并对常见文本内容进行了良好的覆盖，使得训练得到的悟空画画模型对文本拥有广泛的识别能力，并能根据不同的提示词生成多样的图片风格。

* **PanGu-Draw**：

  * 地址：https://pangu-draw.github.io/

  * 简介： 

    * 网络结构扩容，参数量从1B扩大到5B，是发布时规模较大的中文文生图模型之一；
    *  支持**中英文双语**输入；
    *  提升输出分辨率，支持**原生1K输出**（v1->v2->v3: 512->768->1024）；
    *  多尺寸（16:9、4:3、2:1...）输出；
    *  **可量化的风格化调整**：动漫、艺术性、摄影控制；
    *  基于**昇腾硬件和昇思平台**进行大规模多机多卡训练、推理，全自研昇思MindSpore平台和昇腾Ascend硬件；
    *  采用**自研RLAIF**提升画质和艺术性表达。

* **MiaoBi**：

  * 地址：https://github.com/ShineChen1024/MiaoBi ![](https://img.shields.io/github/stars/ShineChen1024/MiaoBi.svg)

  * 简介： 妙笔的测试版本。妙笔，一个中文文生图模型，与经典的stable-diffusion 1.5版本拥有一致的结构，兼容现有的lora，controlnet，T2I-Adapter等主流插件及其权重。       

    妙笔的训练数据包含Laion-5B中的中文子集（经过清洗过滤），Midjourney相关的开源数据（将英文提示词翻译成中文），以及我们收集的一批数十万的caption数据。由于整个数据集大量缺少成语与古诗词数据，所以对成语与古诗词的理解可能存在偏差，对中国的名胜地标建筑数据的缺少以及大量的英译中数据，可能会导致出现一些对象的混乱。妙笔Beta0.9在8张4090显卡上完成训练，我们正在拓展我们的机器资源来训练SDXL来获得更优的结果。

* **腾讯混元DiT**：

  * 地址：https://github.com/Tencent-Hunyuan/HunyuanDiT ![](https://img.shields.io/github/stars/Tencent-Hunyuan/HunyuanDiT.svg)

  * 简介：混元DiT，一个基于Diffusion transformer的文本到图像生成模型，此模型具有中英文细粒度理解能力。为了构建混元DiT，我们精心设计了Transformer结构、文本编码器和位置编码。我们构建了完整的数据管道，用于更新和评估数据，为模型优化迭代提供帮助。为了实现细粒度的文本理解，我们训练了多模态大语言模型来优化图像的文本描述。最终，混元DiT能够与用户进行多轮对话，根据上下文生成并完善图像。
    
    Hunyuan-DiT是一个在潜在空间中的扩散模型。基于潜在扩散模型，使用预训练的变分自编码器（VAE）将图片压缩到低维度的潜在空间，并训练一个扩散模型来学习数据分布。我们的扩散模型是用transformer参数化的。为了编码文本提示，我们利用了预训练的双语（英语和中文）CLIP和多语言T5编码器的组合。混元DiT提供双语生成能力，中国元素理解具有优势。混元DiT能分析和理解长篇文本中的信息并生成相应艺术作品。混元DiT能捕捉文本中的细微之处，从而生成完美符合用户需要的图片。混元DiT可以在多轮对话中通过与用户持续协作，精炼并完善的创意构想。性能上超过SDXL，Playground 2.5等。
    
* **Kolors**：

  * 地址：https://github.com/Kwai-Kolors/Kolors ![](https://img.shields.io/github/stars/Kwai-Kolors/Kolors.svg)

  * 简介：在架构上，可图也是采用latent diffusion架构，基本沿用SDXL的模型设计，但是文本编码器采用了支持中英文双语的ChatGLM3-6B-Base，而且文本提示词的输入长度支持 256 tokens，这比77 tokens的CLIP要长得多。使用GLM也比采用CLIP有更强的文本理解能力，和DALL-E 3一样，可图也对训练数据集中的图像作了重打标来生成文本详细描述，这里采用的打标模型是开源模型中效果相对较好的CogVLM-1.1-chat，由于多模态大模型MLLMs无法识别图像中的特定的概念，所以训练过程中采用混合caption：50%用原始文本，50%用合成的文本，这和SD3的训练策略类似。此外，为了让可图支持写中文，这里也是专门构建了包含中文字的数据集，包括合成的数据以及通过OCR或者多模态大模型打标的数据集。在训练策略上，可图也是采用两阶段训练，首先是预训练阶段，技术报告里面叫concept learning，这个阶段就是从大量的文本图像对数据集上学习，让模型能够对文本有很强的理解能力。然后是微调阶段，通过构建高质量数据来提升图像质量和分辨率。

* **UniT2IXL联通元景**：

  * 地址：https://github.com/UnicomAI/UniT2IXL ![](https://img.shields.io/github/stars/UnicomAI/UniT2IXL.svg)

  * 简介：联通元景（UniT2IXL）是中国联通AI推出的中文原生文生图模型，完全在国产昇腾AI基础软硬件平台上实现训练和推理。该模型采用复合语言编码模块，优化中文长文本和特色词汇理解，提升图像生成质量。联通元景基于预训练海量中文图文数据，减少信息损失，准确生成高质量图片。元景文生图模型支持国产全栈训推，适配自定义数据集，实现跨平台平滑切换。已在多个行业如文创、服装等领域应用，助力企业提效降本。复合语言编码模块：在SDXL架构中融合复合语言编码模块，替换英文CLIP模型为中文CLIP，增强中文短文本的理解能力。encoder-decoder架构：引入基于encoder-decoder架构的语言模型到语言编码器部分，支持超过CLIP长度限制的长文本输入。
    昇腾AI算力集群：在昇腾AI大规模算力集群上实现模型的训练和推理，提供强大的计算支持。接口与Diffusers对齐：模型推理接口与Diffusers对齐，简化使用流程，支持单卡和多卡推理，单卡推理支持UNet Cache加速。

* **CogView4**：

  * 地址：https://github.com/zai-org/CogView4 ![](https://img.shields.io/github/stars/zai-org/CogView4.svg)

  * 简介：CogView4 在 DPG-Bench 基准测试中的综合评分为 85.13。它将文本编码器从纯英文的 T5 encoder 换为具备双语能力的 GLM-4 encoder（GLM-4-9B），并通过中英双语图文进行训练，使模型具备双语提示词输入与汉字生成能力。CogView4 延续 Share-param DiT 架构，Attention 和 FFN 参数由文本、图像共享，并为两种模态分别设计自适应 LayerNorm；DiT 参数量约 6.4B，采用 Flow Matching，支持 512～2048 分辨率和最长 1024 tokens。采用 4-bit 文本编码器时，约需 13GB 显存生成 1024 分辨率图像。

* **HiDream-I1**：

  * 地址：https://github.com/HiDream-ai/HiDream-I1 ![](https://img.shields.io/github/stars/HiDream-ai/HiDream-I1.svg)

  * 简介：HiDream-I1采用扩散模型技术，是一种先进的深度学习方法，通过逐步去除噪声来生成图像。使模型能在细节渲染和图像一致性方面表现出色，生成的图像在色彩还原、边缘处理和构图完整性上都具有高质量。混合专家架构（MoE）：HiDream-I1使用了混合专家架构（MoE）的DiT模型，结合了双流MMDiT block与单流DiT block。通过动态路由机制高效分配计算资源，使模型在处理复杂任务时能够更灵活地利用计算能力。多种文本编码器集成：为了提升语义理解能力，HiDream-I1集成了多种文本编码器，包括OpenCLIP ViT-bigG、OpenAI CLIP ViT-L、T5-XXL和Llama-3.1-8B-Instruct。能更准确地理解文本描述，生成更符合用户需求的图像。大规模预训练策略：开发团队采用了大规模预训练策略，使HiDream-I1在生成速度与质量之间找到了绝佳平衡点。通过这种方式，模型能在短时间内生成高质量的图像，同时保持较高的生成效率。优化机制：HiDream-I1采用了Flash Attention等优化机制，进一步提升了生成图像的速度和质量。使模型在实际应用中更加高效，能快速响应用户的生成请求。

* **Qwen-Image/Edit**：

  * 地址：https://github.com/QwenLM/Qwen-Image ![](https://img.shields.io/github/stars/QwenLM/Qwen-Image.svg)

  * 简介：Qwen-image扩散模型是一个20B的MMDiT，采用Flow Matching，patch size为2x2。但是这里的模型架构设计和SD3类似，transformer只包含60层的MMDiT block（文本和图像采用不同的参数），而不像Flux那样还包含单流的DiT Block（文本和图像共享参数）。Qwen-Image的MMDiT一个独特设计是位置编码，这里引入了一种新的位置编码方法：多模态可扩展旋转位置编码（Multimodal Scalable RoPE，简称 MSRoPE）。这种设计使得 MSRoPE 能够在图像端利用分辨率缩放的优势；而且在文本端保持与一维 RoPE 等效的行为，从而无需再为文本设计复杂的位置编码策略。VAE是采用同时支持编码图像和视频的3D VAE，以在未来支持视频。这里的3D VAE复用Wan 2.1 VAE，模型大小为127M，空间下采样8x，时序下采样4x，latent特征维度为16，比如对于输入为1024x1024的图像，VAE编码的latent特征维度16x128x128。另外，为了提升VAE的重建精度，尤其是针对小字体文本和细粒度细节的还原能力，这里还基于内部构建的富含文本的图像数据集上对VAE decoder进行了微调，这里仅组合使用重建损失和感知损失。text encoder采用Qwen2.5-VL（提取模型最后一层的隐含层特征），具体来说是Qwen2.5-VL-7B。Qwen2.5-VL 的语言与视觉空间已经对齐，而且保留了纯语言模型的语言建模能力，相较于仅语言模型更适用于文生图任务，而且Qwen2.5-VL 支持多模态输入，使得 Qwen-Image 能够更好地支持图像编辑这样的任务。Qwen-Image的训练数据规模是十亿级图文对，包含自然类（Nature）、设计类（Design）、人物类（People）以及合成类数据（Synthetic Data），其中合成数据占比约5%。训练数据采用七阶段的渐进式数据过滤流程，模型的后训练包括SFT和RL，在微调阶段，用高质量图像和人工标注优化模型，使其生成更真实、细节更丰富的内容。在强化学习阶段中，先用高效的DPO方法进行大规模偏好训练，再用GRPO做小范围精细调整，从而提升生成效果。
 
* **Wan2.1**：

  * 地址：https://github.com/Wan-Video/Wan2.1 ![](https://img.shields.io/github/stars/Wan-Video/Wan2.1.svg)

  * 简介：Wan2.1可以生图，采用交叉注意力来嵌入文本条件，为了进一步增强模型捕捉复杂动态的能力，加入了完整的时空注意力机制，有140亿参数，训练包括了数十亿图像和视频的大规模数据。训练数据构建遵循三个核心原则：高质量、高多样性和大规模。从内部版权来源和公开可访问的数据中采集并去重，预训练阶段，我们的目标是从这个庞大而噪杂的数据集中选择高质量和多样化的数据，以促进有效的训练，设计了一个四步数据清洗流程，重点关注基本维度、视觉质量和运动质量。Wan-VAE实现了仅127M参数的模型，遵循MagViT-v2，我们利用流匹配框架来建模图片和视频领域的统一去噪过程，首先在低分辨率图像上预训练，然后对图像和视频进行多阶段联合优化，umT5是文本嵌入序列，长度为512个标记。图像预训练：直接联合训练高分辨率图像和长时间视频序列的两个关键挑战：1.扩展序列长度，通常81帧的1280x720视频，显著降低了训练吞吐量；2.过高的GPU内存消耗迫使使用次优的bs，导致由于梯度方差的波动而引起的训练不稳定性；通过低分辨率256p的文本到图像预训练初始化14B模型训练，强制进行跨模态语义文本对齐和几何结构表征，然后逐步引入高分辨率视频模态。图像-视频联合训练：在大规模256p文本到图像预训练之后，通过分辨率渐进的方式实行图像和视频数据的分阶段联合训练，训练包括三个不同阶段，按分辨率区分：1.在第一阶段，使用256p的图像和5s的视频片段（192p,16fps）进行联合训练；2.在第二阶段，将图像和视频分辨率都升级到480p（会进行分辨率缩放），同时保持固定的5s视频时长，3.将图像和5s的视频片段的分辨率提升到720p。保持与预训练相同的模型架构和优化器配置，在480p和720p下，使用后训练数据集联合训练。

* **Wan2.2**：

  * 地址：https://github.com/Wan-Video/Wan2.2 ![](https://img.shields.io/github/stars/Wan-Video/Wan2.2.svg)

  * 简介：Wan2.2可以生图，且效果优于Wan2.1。混合专家（MoE）架构：引入MoE架构，将模型分为高噪声专家和低噪声专家。高噪声专家负责视频的整体布局，低噪声专家负责细节完善。两个专家各约14B参数，总计27B，但每步仅激活14B，在保持计算成本不变的情况下，大幅提升模型的参数量和生成质量。扩散模型（Diffusion Model）：基于扩散模型作为基础架构，通过逐步去除噪声来生成高质量的视频内容。MoE架构与扩散模型结合，能进一步优化生成效果。高压缩率3D VAE：为提高模型的效率，通义万相2.2基于高压缩率的3D变分自编码器（VAE）。架构实现了时间、空间的高压缩比，让模型能在消费级显卡上快速生成高清视频。大规模数据训练：模型在大规模数据集上进行训练，包括更多的图像和视频数据，提升模型在多种场景下的泛化能力和生成质量。美学数据标注：基于精心标注的美学数据（如光影、色彩、构图等），模型能生成具有专业电影质感的视频内容，满足用户对视频美学的定制需求。

* **Hunyuan2.1**：

  * 地址：https://github.com/Tencent-Hunyuan/HunyuanImage-2.1 ![](https://img.shields.io/github/stars/Tencent-Hunyuan/HunyuanImage-2.1.svg)

  * 简介：混元图像2.1在复杂语义理解和跨领域泛化能力上有了显著提升，它支持最长达1000个tokens的提示词，可精准生成场景细节、人物表情和动作，实现多物体的分别描述与控制。此外，混元图像2.1还能够对图像中的文字进行精细控制，使文字信息与画面自然融合。1、模型对复杂语义理解能力强，支持多主体分别描述与精确生成;2、模型对图像中的文字和场景细节的把控更为稳定;3、模型支持风格丰富，如真人、漫画与搪胶手办等，并具备较高美感。混元图像2.1模型不仅采用了海量训练数据，还利用结构化、不同长度、内容多样的caption，极大提升了对文本描述的理解能力。在caption模型中，引入了OCR和IP RAG专家模型，有效增强了对复杂文字识别和世界知识的响应能力。为大幅降低计算量、提升训练和推理效率，模型采用了32倍超高压缩倍率的VAE, 并使用dinov2对齐和repa loss来降低训练难度。因此，模型能高效原生生成2K图。在文本编码方面，混元图像2.1配备了双文本编码器：一个MLLM模块用于进一步提升图文对齐能力，另一个ByT5模型则增强了文字生成表现力。整体架构为17B参数的单/双流DiT模型。此外，混元图像2.1还在17B参数量级的模型上解决了平均流模型（meanflow）的训练稳定性问题，将模型推理步数由100步蒸馏到8 步，显著提升推理速度的同时保证了模型原有的效果。

* **Hunyuan3**：

  * 地址：https://github.com/Tencent-Hunyuan/HunyuanImage-3.0 ![](https://img.shields.io/github/stars/Tencent-Hunyuan/HunyuanImage-3.0.svg)

  * 简介：混元图像3.0是首个开源的工业级原生多模态图像生成模型，这里的工业级说的是效果能达到可用的地步，之前学术界其实也有很多开源的原生多模态生图模型，但是效果上其实都不算很好。此外，这个模型是80B的MoE模型（13B激活参数），也是目前参数量最大的开源生图模型。HunyuanImage-3.0 是一个突破性的原生多模态模型，它在一个自回归框架内统一了多模态理解和生成。我们的文本到图像模块实现了与领先闭源模型相当或超越的性能。统一多模态架构：超越流行的基于 DiT 的架构，HunyuanImage-3.0 采用统一的自回归框架。这种设计能够更直接和集成地建模文本和图像模态，从而实现令人惊讶的有效且具有丰富上下文的图像生成。最大图像生成 MoE 模型：这是迄今为止最大的开源图像生成专家混合（MoE）模型。它拥有 64 个专家，总参数量达 800 亿，每个 token 激活参数达 130 亿，显著提升了其容量和性能。卓越的图像生成性能：通过严格的数据集筛选和先进的强化学习后训练，我们在语义准确性和视觉卓越性之间实现了最佳平衡。该模型在遵循提示方面表现出色，同时提供具有惊人美学质量和细粒度细节的逼真图像。 智能世界知识推理：统一的跨模态架构赋予 HunyuanImage-3.0 强大的推理能力。它利用其丰富的世界知识智能地理解用户意图，自动用恰当的上下文细节补充稀疏提示，以生成更优质、更完整的视觉输出。

    **更新**：2026年1月26日，腾讯发布了HunyuanImage-3.0-Instruct及其蒸馏版本HunyuanImage-3.0-Instruct-Distil。Instruct版本引入了Chain-of-Thoughts推理能力，支持指令驱动的图像生成和编辑。截至2026年7月5日，Hunyuan Image 3.0在LM Arena Text-to-Image Overall排行榜上排名第24位（Elo 1151±3）。

* **Z-Image**：

  * 地址：https://github.com/Tongyi-MAI/Z-Image ![](https://img.shields.io/github/stars/Tongyi-MAI/Z-Image.svg)

  * 简介：采用了一种可扩展的单流数字图像处理 （S3-DiT）架构。在该架构中，文本、视觉语义标记和图像 VAE 标记在序列级别上连接起来，作为统一的输入流，与双流方法相比，最大限度地提高了参数效率。Decoupled-DMD：Z-Image背后的加速魔力，Decoupled-DMD 是赋能 8 步 Z-Image 模型的核心少步蒸馏算法。团队在 Decoupled-DMD 中的核心洞察是，现有分布匹配蒸馏（Distribution Matching Distillation，DMD）方法的成功来源于两个独立且协作的机制：CFG 增强：驱动蒸馏过程的主要引擎 ，这是以前工作中大多被忽视的因素。分布匹配：更像是一种正则化器 ，确保生成结果的稳定性和质量。通过识别并解耦这两个机制，能够独立地研究和优化它们。这最终促使团队开发出了一种改进的蒸馏流程，大幅提升了少步生成的性能。在Decoupled-DMD 基础上，8 步 Z-Image 模型已经展示了卓越的能力。为了在语义对齐、美学质量和结构一致性方面实现进一步提升，同时生成具有更丰富高频细节的图像，团队提出了 DMDR。DMDR 的核心洞见是，强化学习（RL）与分布匹配蒸馏（DMD）可以在少步模型的后训练阶段协同整合。团队展示了：1. RL 解锁了 DMD 的性能，2. DMD 有效规范了 RL。Z-Image-Turbo —— Z-Image 的蒸馏轻量版，仅使用 8 步即可达到或超越主流竞品性能。它在企业级 H800 GPU 上可实现亚秒级推理速度⚡️，并能轻松运行于 16G显存的消费级设备。该模型在照片级写实生成、中英双语文字渲染，以及指令遵循方面表现突出。

    **更新**：官方模型族包含Z-Image-Turbo、Z-Image、Z-Image-Omni-Base和Z-Image-Edit等版本。Z-Image-Turbo是8步蒸馏版本，面向快速高质量文生图；Z-Image是Turbo背后的基础生成模型，支持负提示词、创意生成、微调和下游开发；Z-Image-Omni-Base定位为兼具生成和编辑能力的原始基础模型；Z-Image-Edit是面向图像编辑的微调版本。截至2026年7月，官方Model Zoo中Z-Image与Z-Image-Turbo已提供权重，Z-Image-Omni-Base与Z-Image-Edit仍标注为待发布。

* **Ovis-Image**：

  * 地址：https://github.com/ATH-MaaS/Ovis-Image ![](https://img.shields.io/github/stars/ATH-MaaS/Ovis-Image.svg)

  * 简介：Ovis-Image 是基于 Ovis-U1 构建的 7B 文本到图像模型，专门针对高质量文本渲染进行了优化。模型采用 MMDiT 和以文本为核心的训练流程，后训练包括 SFT、DPO 与 GRPO。在官方报告的 LongText-Bench-ZH 中文长文本评测中表现突出，并面向海报、横幅、徽标、UI 模型和信息图等文本密集、布局敏感的场景优化。

* **LongCat-Image**：

  * 地址：https://github.com/meituan-longcat/LongCat-Image ![](https://img.shields.io/github/stars/meituan-longcat/LongCat-Image.svg)

  * 简介：LongCat-Image具备出色的跨语言图像编辑能力，通过共享 MM-DiT+Single-DiT 混合主干架构与VLM条件编码器，文生图与编辑能力相互辅助，继承文生图的出图质量并具备出色的指令遵循、一致性保持能力，在主流公开评测基准上达到第一梯队水平。文字生成专项能力上，覆盖全量通用规范汉字并在商业海报、自然场景文字上都展现出极强的适用性。此外，通过精细化模型设计及多阶段训练策略优化，极大提升生成真实度、合理性并可支持消费级显卡高效推理。图像编辑：ImgEdit 得分 4.50（开源 SOTA），GEdit 中/英 7.60 / 7.64，接近商业模型；文字渲染：ChineseWord 分数 90.7，超越所有竞品；文生图：GenEval 0.87、DPG 86.8，达到开源/闭源顶级模型水平。

* **FLUX.2**：

  * 地址：https://github.com/black-forest-labs/flux2 ![](https://img.shields.io/github/stars/black-forest-labs/flux2.svg)

  * 简介：参数量级：Text Encoder (TE) 激增至 23B，配合 32B 的 DiT (Diffusion Transformer) 主干，总参数量达到了恐怖的 55B 级别。模型性质：目前的 dev 版本是一个蒸馏模型 (Distilled)，这意味着它在保持高性能的同时，推理步数被压缩了。多参考图的“逻辑融合” 这一点非常 aggressive。Flux.2 支持高达 10 张的参考图输入 (Multi-Reference)。实际官方示例来看，其对 ID 保持、风格迁移和物体融合的理解能力还是不错的。对于需要处理复杂构图和一致性角色的工作流，这是质变。提示词工程：从 NLP 到 结构化指令 Text Encoder 的升级带来了全新的 Prompt 范式。结构化 (Structured)：模型对 JSON 格式的理解力大幅提升，支持分层定义主体、光影、构图。Hex 色值锚定：不再需要用 "dark blue" 这种模糊词，直接丢给它 #0033FF，模型能将其精确映射到 Latent Space 的颜色向量中。这意味着 AI 正在从“生成艺术”走向“工业设计”。视觉上限：4MP 原生分辨率 + 全新 VAE分辨率：原生支持生成 4MP（约 2048x2048）级别的图像，在这个分辨率下，细节的连贯性依然稳健。VAE 升级：配套了全新的 Flux.2 VAE。众所周知，高分图的崩坏往往始于解码阶段，新的 VAE 显然是为了应对 4K 级（4096px）纹理细节而特调的，极大地修复了边缘伪影和细节涂抹感。

* **GLM-Image**：

  * 地址：https://github.com/zai-org/GLM-Image ![](https://img.shields.io/github/stars/zai-org/GLM-Image.svg)

  * 简介：首个开源的，国产链路训练的工业表现级离散自回归图像生成模型，是面向认知型生成技术范式的一次尝试，在文字生成上取得 SOTA 成绩。整个模型可以拆成两大块： Autoregressive（AR）模块：负责生成离散视觉 token。AR 模块本质上是个多模态理解 + 生成系统，包含 GLM-4-0414-9B 结构的语言模型, X-Omni-En 的 ViT 和 VQVAE。在原词表前面新增了16512个视觉 token，这也是这个模型lm head的大小，AR的输出结果不是直接出图，而是先产出一串离散视觉 token，然后把它们交给后面的 DiT 当输入。 Diffusion Decoder 模块：负责把 token 变成高质量图像，包含 DiT（Diffusion Transformer）Glyph + VAE，这里和去年发布的 CogView4 相似，是经典的Diffusion结构。不管多少张图以及什么比例，AR 这边的 ViT 会先把参考图编码成 Image Token （参考 GLM-V 类似的处理），这些 token 不仅给AR 用，也会复用到 DiT 里作为 condition_token（条件输入），让生成更贴着参考图走。
    

* **Qwen-Image-2512**：

  * 地址：https://huggingface.co/Qwen/Qwen-Image-2512

  * 简介：Qwen-Image-2512是阿里Qwen-Image的更新版本（2025年12月发布），专注于提升人像生成的真实感。相比初版Qwen-Image，该模型在面部细节、皮肤纹理、光影表现等方面进行了优化，生成的人像更加自然，减少了"AI生成感"。截至2026年7月5日，在LM Arena Text-to-Image Overall排行榜上排名第34位（Elo 1127±4）。模型架构与Qwen-Image一致，为20B的MMDiT，采用Qwen2.5-VL-7B作为文本编码器。

* **BAGEL-7B-MoT**：

  * 地址：https://github.com/ByteDance-Seed/Bagel ![](https://img.shields.io/github/stars/ByteDance-Seed/Bagel.svg)

  * 简介：BAGEL-7B-MoT是字节跳动Seed团队于2025年5月发布的统一多模态基础模型，总参数量14B，激活参数7B。该模型采用Mixture-of-Transformer（MoT）架构，在单一模型中统一了文生图生成（效果可与SD3竞争）、图像编辑和多模态理解三大能力。在LM Arena文生图排行榜上有排名。BAGEL的创新之处在于将理解和生成能力融合在同一个模型中，无需为不同任务切换模型。


* **HiDream-O1-Image**：

  * 地址：https://github.com/HiDream-ai/HiDream-O1-Image ![](https://img.shields.io/github/stars/HiDream-ai/HiDream-O1-Image.svg)

  * 简介：HiDream-O1-Image是HiDream.ai于2026年5月8日发布的8B开源文生图模型（MIT许可证），采用全新的像素级统一Transformer（Pixel-level Unified Transformer, UiT）架构，彻底移除了传统的VAE模块，扩散Transformer直接在原始像素空间操作，将文本和任务条件统一在同一个token空间中。该模型在单一架构中统一了文生图生成、指令编辑、主体驱动个性化和分镜生成四大能力，支持最高2048×2048分辨率。HiDream-O1-Image内置了基于Gemma-4-31B-IT的推理驱动提示词代理（Reasoning-Driven Prompt Agent），在生成前进行O1风格的思考规划。发布时在Artificial Analysis Image Arena排名第8，是当时排名最高的开源模型，以7倍更小的参数量超越了FLUX.2 Dev（56B）。

* **NextStep-1 / NextStep-1.1**：

  * 地址：https://github.com/stepfun-ai/NextStep-1 ![](https://img.shields.io/github/stars/stepfun-ai/NextStep-1.svg)

  * 简介：NextStep-1是阶跃星辰（StepFun）于2025年8月发布的14B自回归文生图模型，获得**ICLR 2026 Oral**，训练代码和NextStep-1.1后训练版本于2026年2月16日开源。该模型采用自回归语言模型架构处理连续图像token，创新性地使用标准LM head处理离散文本token、轻量级Flow Matching head（157M）处理连续视觉token，通过新型自编码器将图像编码为patchwise连续latent token。NextStep-1在DPG-Bench上达到85.28分，支持文生图和图像编辑（物体增删、背景替换、风格迁移）。该模型代表了自回归范式在文生图领域的前沿探索。

* **Boogu-Image-0.1**：

  * 地址：https://github.com/boogu-project/Boogu-Image ![](https://img.shields.io/github/stars/boogu-project/Boogu-Image.svg)

  * 简介：Boogu-Image-0.1是Boogu Project于2026年6月16日发布的10B开源文生图与图像编辑统一模型家族（Apache-2.0许可证），包含Base（基础生成模型，25-50步推理，强调多样性和可控性）、Turbo（3-4步蒸馏版，基于Decoupled DMD加速，面向快速推理和照片级真实感）、Edit（图像编辑与变换）和Edit-Turbo（四步蒸馏编辑版）等变体。文本编码器采用Qwen3-VL-8B-Instruct，VAE复用开源FLUX.1 VAE，支持1K和2K分辨率输出。该模型支持中英双语文字渲染，擅长海报、印章、文档界面、品牌指南等场景的超密集文字生成，并提供FP8量化版本以降低部署门槛。2026年6月30日，官方发布Boogu-Image-0.1-Edit-Turbo；2026年7月8日发布Edit-Turbo hotfix，修复前一版本问题并提供1K/1.5K检查点。

    **技术报告与部署更新**：官方于 2026 年 7 月 16 日发布[技术报告](https://arxiv.org/abs/2607.13125)（arXiv 首次提交于 7 月 14 日），披露训练使用 208.62M 张去重图像，Base 的理论训练成本约 40 万美元，并以 Apache-2.0 开放代码、权重与训练 recipes。7 月 22 日，vLLM-Omni 主仓库新增[社区维护的 Boogu-Image recipe](https://github.com/vllm-project/vllm-omni/blob/ee33954dff27da317be597449a6c1b5a5df4052b/recipes/Boogu/Boogu-Image.md)；7 月 23 日，项目新增[昇腾 NPU 初步支持](https://github.com/boogu-project/Boogu-Image/blob/b40214e5c0f94579a932fdc8074c17330051d16f/NPU_INFERENCE_GUIDE.md)。

* **JoyAI-Image / Edit**：

  * 地址：https://github.com/jd-opensource/JoyAI-Image ![](https://img.shields.io/github/stars/jd-opensource/JoyAI-Image.svg) | [论文](https://arxiv.org/abs/2605.04128)

  * 简介：京东开源的统一视觉模型，采用 8B MLLM 与 16B MMDiT，覆盖图像理解、文生图和指令编辑，重点强化空间关系、视角与物体控制、长文本排版和多图编辑。目前已开放理解模型、单图 Edit 与多图 Edit-Plus 权重及推理代码（Apache-2.0）；2026 年 7 月 17 日，Edit 与 Edit-Plus 新增原生 ComfyUI 支持。

* **ERNIE-Image / 文心 ERNIE-Image**：

  * 地址：https://huggingface.co/baidu/ERNIE-Image

  * 简介：百度文心于 2026 年 4 月 15 日开源的 8B 单流 DiT 中文文生图模型，以 Apache-2.0 许可证发布。文本编码器基于 ERNIE LLM，并搭配轻量 Prompt Enhancer 与 iRAG 检索增强，实现强中英双语理解与 2K 高清生成。当前在 8B 量级开源模型中**中英双语图内文字渲染**与多面板漫画生成能力领先。同期还放出了 8 步采样的 **ERNIE-Image-Turbo** 蒸馏版（[huggingface.co/baidu/ERNIE-Image-Turbo](https://huggingface.co/baidu/ERNIE-Image-Turbo)），延迟优化场景的同源伴侣模型。

* **Mage-Flow / Mage-Flow-Edit**：

  * 地址：https://github.com/microsoft/Mage ![](https://img.shields.io/github/stars/microsoft/Mage.svg) | [论文](https://arxiv.org/abs/2607.19064) | [生成权重](https://huggingface.co/microsoft/Mage-Flow) | [编辑权重](https://huggingface.co/microsoft/Mage-Flow-Edit)

  * 简介：微软于 2026 年 7 月 22 日开放的 4B 图像生成与指令编辑模型族。模型采用 Native-Resolution MMDiT、Mage-VAE 与 Qwen3-VL 文本编码器，同一套 4B 主干分别提供 Base、RL 对齐和 4 步 Turbo 版本；支持 512～2048 分辨率、最高 4:1 宽高比及中文文字渲染。官方开放生成和编辑代码、权重，项目仓库采用 MIT 许可证；编辑版支持单图与多图参考。项目报告给出的结果中，Mage-Flow 在 GenEval 为 0.90、CVTG-2K 为 0.887；这些分数属于项目方自报结果，横向比较时仍应统一评测设置。

* **Qwen-Image-Flash**：

  * 地址：https://huggingface.co/nvidia/Qwen-Image-Flash

  * 简介：NVIDIA 于 2026 年 7 月 23 日发布的 Qwen-Image 四步蒸馏版本，使用 DMD2 保留原 20.43B MMDiT 架构，并提供 Diffusers、SGLang Diffusion、vLLM-Omni 和 TensorRT-LLM 推理路径。四步推理减少了 Transformer 前向次数，但不减少参数量或基础权重占用；官方仅在 1024×1024 上完成测试。该版本使用英文提示词蒸馏，模型卡明确说明继承的中文能力尚未评测，且用途限于文生图，不包含图像编辑。权重遵循 NVIDIA Open Model License。

### 1.3 闭源模型

* **Qwen-Image-3.0 / Qwen-Image-3.0-Pro**：

  * 地址：https://help.aliyun.com/zh/model-studio/qwen-image-generation-and-editing-api-reference | [官方首发](https://zhuanlan.zhihu.com/p/2062903884503330992) | [全量开放](https://zhuanlan.zhihu.com/p/2068284890467014567) | [Standard](https://help.aliyun.com/zh/model-studio/qwen-image-3-0) | [Pro](https://help.aliyun.com/zh/model-studio/qwen-image-3-0-pro)

  * 简介：阿里于 2026 年 7 月 21 日发布并开启 API 邀测、8 月 5 日全量开放的闭源图像生成与编辑模型族，包含兼顾质量与速度的 Standard 和强调复杂版面、真实细节的 Pro。两者均支持文生图，以及基于 1～3 张参考图的图生图/指令编辑；支持最长约 4.5K tokens 的中英文指令、12 种语言与多字体文字渲染、最多 6 张输出。文生图和图生图的输出像素面积均为 512×512～2048×2048，宽高比可在 1:8～8:1 之间。目前仅提供 API，未公开模型架构和权重。

* **Wan2.7-Image / Wan2.7-Image-Pro**：

  * 地址：https://help.aliyun.com/zh/model-studio/wan-image-generation-and-editing-api-reference

  * 简介：阿里云百炼提供的闭源图像生成与编辑模型，覆盖文生图、图像编辑、多图参考和组图生成。标准版支持最高 2K；Pro 版仅在单张文生图场景支持 4K（4096×4096），图像编辑与组图生成仍最高为 2K。

* **Qwen-Image-2.0**：

  * 地址：https://qwen.ai/blog?id=qwen-image-2.0

  * 简介：Qwen 团队于 2026 年 2 月 10 日发布的闭源图像生成与编辑模型，支持专业排版渲染（PPT、海报和漫画等）、最长 1000 tokens 的复杂输入和原生 2K 输出。目前官方未公开对应模型权重。

* **Qwen-Image-2.0-Pro**：

  * 地址：https://qwen.ai/blog?id=a6f483777144685d33cd3d2af95136fcbeb57652

  * 简介：Qwen-Image-2.0-Pro是阿里Qwen团队于2026年4月22日发布的Qwen-Image-2.0增强版本。截至2026年7月5日，qwen-image-2.0-pro-2026-06-22在LM Arena Text-to-Image Overall排行榜上排名第12位（Elo 1193±8）。相比基础版Qwen-Image-2.0，Pro版本在排版渲染、细节保真度和复杂提示理解上均有显著提升。

* **腾讯混元**：
  * 地址：https://mp.weixin.qq.com/s/hEqVR89qDyMckld-OikDPQ
  * 简介：大模型文生图的难点体现在对提示词的语义理解，生成内容的合理性以及生成图片的效果，针对这三个技术难点，腾讯进行了专项的技术研究，提出了一系列原创算法，来保证生成图片的可用性和画质。 

    1、在语义理解方面，腾讯混元采用了中英文双语细粒度的模型，模型同时建模中英文实现双语理解，而不是通过翻译，通过优化算法提升了模型对细节的感知能力与生成效果，有效避免多文化差异下的理解错误。

    2、在内容合理性方面，AI生成人体结构和手部经常容易变形。混元文生图通过增强算法模型的图像二维空间位置感知能力，并将人体骨架和人手结构等先验信息引入到生成过程中，让生成的图像结构更合理，减少错误率。

    3、在画面质感方面，混元文生图基于多模型融合的方法，提升生成质感。经过模型算法的优化之后，混元文生图的人像模型，包含发丝、皱纹等细节的效果提升了30%，场景模型，包含草木、波纹等细节的效果提升了25%。

* **美图MiracleVision**：
  * 地址：https://mp.weixin.qq.com/s/Hixjc6x-L-Zd5JLBjVXCZA
  * 简介：美图自研大模型名叫**MiracleVision**（奇想智能）。其最显著的特点是更懂美学。美图把长期积累的美学认知融入MiracleVision视觉大模型，并搭建了基于机器学习的美学评估系统，为模型生成结果打上“美学分数”，从而不断地提升模型对美学的理解。

* **网易丹青**：
  * 地址：https://zhuanlan.zhihu.com/p/648712812

  * 简介：丹青模型基于原生中文语料数据及网易自有高质量图片数据训练，与其他文生图模型相比，丹青模型的差异化优势在于对中文的理解能力更强，对中华传统美食、成语、俗语、诗句的理解和生成更为准确。比如，丹青模型生成的图片中，鱼香肉丝没有鱼，红烧狮子头没有狮子。基于对中文场景的理解，丹青模型生成的图片更具东方美学，能生成“飞流直下三千尺”的水墨画，也能生成符合东方审美的古典美人。

    基于数据集和理解模型，网易伏羲对图文生成算法进行重构，依托于扩散模型的原理，在广泛的（8 亿）图文数据上训练以达到较好的生成结果。具体来说，丹青模型侧重文本与图片的交互，强化了在文图引导部分的参数作用，能够让文本更好地引导图片的生成，因此生成的结果也更加贴近用户意图。同时，丹青模型进行了图片多尺度的训练，充分考虑图片的不同尺寸和清晰度问题，将不同尺寸和分辨率的图片进行分桶。在充分保证训练图片不失真的前提下，保留尽可能多的信息，适应不同分辨率的生成。

    在数据策略方面，丹青模型在初始阶段使用亿级别的广泛分布的数据，不仅在语义理解上具有广泛性，可以很好地理解一些成语、古文诗句，在生成的画风上也具有多样性，可以生成多种风格。在之后的阶段，丹青模型分别从图文关联度、图片清晰度、图片美观度等多个层面进行数据筛选，以优化生成能力，生成高质量图片。

    此外，丹青模型在训练和生成阶段还引入了人工反馈。在训练阶段，人工从多个维度进行评估，筛选出大批高质量图文匹配、高美观度数据，以补足自动流程缺失能力，帮助基础模型获得更好的效果；在生成阶段，人工对模型的语义生成能力和图片美观度进行评分，筛选出大批量优质生成结果，将其作为正反馈引入模型训练，实现数据闭环。

* **腾讯太极**：

  * 地址：https://zhuanlan.zhihu.com/p/590459240

  * 简介：1.太极-Imagen文生图模型：团队成员对Imagen模型进行了实现和改进，主要采用自研的中文文本编码器，优化模型训练过程，结合latent diffusion model优化超分辨率模型训练过程，在内部亿级别的中文场景数据上进行训练，获得了在中文场景下自研文生图模型。

    中文文本编码器：在训练Imagen模型的过程中，我们发现文本编码器对于生成模型的语义理解至关重要，在英文场景中Imagen采用了T5-XXL作为文本编码器并通过固定了文本编码器训练生成模型的方法使得模型具有强大的文本理解能力。在中文场景中，我们采用自研的混元sandwich模型作为文本编码器，该文本编码器在中文场景中强大的语义理解能力为中文文生图模型的训练奠定了良好的基础。同时，Imagen模型训练过程中，我们发现文本embedding和Imagen模型参数的匹配也对生成结果起了至关重要的作用。在模型训练的第一阶段，我们首先固定文本编码器，训练diffusion模型的参数，通过文本embedding来指导模型的生成结果。当第一阶段训练收敛后，我们发现模型对于中文场景的实体，物体关系等已经有了较好的理解，但是对于更难，更细粒度的语义提升困难。因此在第二阶段，我们通过放开文本编码器的参数，将其与diffusion model一起进行端到端的训练，能够进一步提升模型对于细粒度语义的理解。

    多阶段不同分辨率级联生成：Imagen通过级联的diffusion模型生成不同分辨率的图像，其中第一阶段的模型生成64x64分辨率的图像，第二阶段和第三阶段分别生成256x256分辨率和1024x1024分辨率的图像。通过多阶段级联的结构，可以使得第一阶段模型的文生图模型训练更加的高效。

    文生图大模型训练策略优化：最后，针对自研Imagen，我们训练了不同参数量和大小的模型。我们首先训练了u-net核心参数量为3亿的模型，已经能够获得中文场景下不错的效果，之后我们将模型规模扩大到核心参数量为13亿，基于团队在太极-DeepSpeed的大规模预训练加速优化技术，在亿级数据上，32*A100只需要2周时间即可收敛。经过实验对比，13亿参数的大模型比3亿参数模型在生成图像细节和语义捕获能力上都获得了更好的效果。

    2.太极-sd文生图模型：在中文场景的SD训练中，一方面对文本编码器进行了替换，将其从原生的CLIP替换为自研的中文太极-ImageCLIP图文匹配模型，并且在训练过程中，优先对文本encoder部分进行训练，以保留SD预训练模型的生成能力；另一方面，为了提升模型对于文本内语义，数量，实体等不同方面的捕捉能力，我们综合了太极-ImageCLIP和混元-Sandwich两类不同的中文encoder所生成的embedding，来指导图片的生成；最后，为了更好的捕捉长文本的信息，我们还将池化后的文本embedding也融合进u-net中，提升整体的生成效果。

* **阿里通义万相**：

  * 地址：https://www.jiqizhixin.com/articles/2023-07-07-6

  * 简介：通义万相基于阿里自研的组合式生成模型 Composer，它拥有 50 亿参数，并在数十亿个文本、图像对上进行训练。在业界都在考虑如何提升 AI 绘画模型的可控性这一点上，Composer 给出了它的创新性思路。

    通过一个基于扩散模型的「组合式生成」框架，Composer 能够对配色、布局、风格等图像设计元素进行拆解和组合，实现了高度可控性和极大自由度的图像生成效果。所谓拆解 - 组合，首先将图像分解为不同的设计元素，比如配色、草图、布局、风格、语义、材质等。然后使用 AI 模型将这些设计元素重新组合成新的图像。这里，拆解 - 组合过程中允许对用到的元素自由修改编辑，如此一来可控性大大增强。
    
    正是基于 Composer 框架，通义万相才能让我们体验到相似图生成和风格迁移这两种图生图功能。一边用图像理解模型将图像拆解为不同元素，一边用扩散模型将这些元素重新组合成新图像，双管齐下，图生图水到渠成。其中对于相似图生成，保持图像语义内容不变，仅仅改变图像中的局部细节，就能生成相似图片。过程中既可以较好地保持原图主体一致性，还提升了生成图的多样性和质量。对于风格迁移，一方面保留原图的基本形态、结构，另一方面将目标风格图片的风格、色彩、笔触等个性化信息，最终实现风格迁移。

* **快手可灵 Kling Image 3.0**：

  * 地址：https://klingai.com
  
  * 简介：快手于2026年2月5日发布的Kling Image 3.0，是可灵系列文生图模型的重大升级。该模型引入了视觉思维链（Visual Chain-of-Thought, VCoT）推理机制，在生成图像前先进行视觉规划推理，显著提升了复杂场景的语义理解和构图合理性。Kling Image 3.0原生支持2K和4K超高清分辨率输出，支持最多10张参考图输入，可生成具有一致风格和叙事连贯性的系列图像。该模型目前通过Kling AI平台以商业API形式提供。

* **字节Seedream系列**：

  * 地址：https://seed.bytedance.com/zh/seedream5_0_pro | [官方发布博客](https://seed.bytedance.com/zh/blog/beyond-generation-it-understands-design-introducing-seedream-5-0-pro)

  * 简介：Seedream是字节跳动的文生图大模型系列，已迭代多个重要版本：

    **Seedream 3.0**（2025年初）：双语文生图基础模型，原生2K分辨率，支持准确的小字生成。截至2026年7月5日，Seedream 3在LM Arena Text-to-Image Overall排行榜上排名第48位（Elo 1082±5）。DiT采用SD3的MMDiT架构，文本tokens和图像tokens统一进行attention但参数不共享，位置编码采用自研的Scaling RoPE（以图像中心编码位置，不同分辨率设置不同缩放因子）。文本编码器采用LLM（而非T5），并结合Glyph-ByT5提取文字字形特征以提升文字渲染准确性。

    **Seedream 4.0**（2025年中）：统一了文生图合成、图像编辑和多图组合能力。采用高效DiT主干+强大VAE，支持原生高分辨率最高4K输出，2K图像生成仅需1.8秒。技术报告：arXiv 2509.20427。截至2026年7月5日，Seedream-4-2k在LM Arena Text-to-Image Overall排行榜上排名第29位（Elo 1141±7）。

    **Seedream 4.5**（2025年底）：改进了主体一致性、参考细节保持和排版保真度，支持批量输入/输出。截至2026年7月5日，在LM Arena Text-to-Image Overall排行榜上排名第28位（Elo 1146±3）。

    **Seedream 5.0 Pro**（2026 年 7 月 8 日）：面向专业设计的闭源多模态图像创作模型，支持高密度信息图的逻辑推理与版面规划、点选/圈选/草图等精确局部编辑、图层拆分、多图融合和真实感增强，并支持十余种常用语言的输入与文字生成。

### 1.4 论文与待开放模型

* **Qwen-Image-2.0-RL**：

  * 地址：https://arxiv.org/abs/2606.27608

  * 简介：Qwen 团队于 2026 年 6 月 25 日公开的强化学习后训练技术报告，围绕 Qwen-Image-2.0 构建文生图与图像编辑奖励体系，并结合 GRPO、混合 CFG、提示词筛选和 on-policy distillation。当前仅有论文，未公开独立 checkpoint。

* **JuZhou 1.0**：

  * 地址：https://arxiv.org/abs/2606.28421

  * 简介：2026 年 6 月 25 日提交的端侧中文文生图技术报告。模型由 0.385B 去噪 UNet 和 1.90M 蒸馏解码器组成，采用 Rectified Flow 与 DMD2 蒸馏，将推理压缩到 4 步；中文对齐阶段使用 900 万精选图文对。目前公开来源为论文，权重仍待开放。

## 2. 测评

### 2.1 评测基准

* **GenEval**：

  * 地址：https://github.com/djghosh13/geneval ![](https://img.shields.io/github/stars/djghosh13/geneval.svg)

  * 简介：一个以物体为中心的文图对齐评估框架（NeurIPS 2023）。通过物体检测模型验证生成图像，测试物体共现、位置、数量、颜色和空间关系等组合式生成能力。被广泛用于评估Hunyuan-DiT、Kolors、FLUX、Seedream、Qwen-Image、SDXL、DALL-E 3、CogView等模型。LongCat-Image在GenEval上得分0.87，达到开源/闭源顶级水平。

* **DPG-Bench**：

  * 地址：https://github.com/TencentQQGYLab/ELLA ![](https://img.shields.io/github/stars/TencentQQGYLab/ELLA.svg)

  * 简介：Dense Prompt Graph Benchmark，由腾讯提出，包含1000条密集/复杂提示词，用于评估文生图模型对详细、多元素描述的指令跟随能力。CogView4在DPG-Bench上综合评分85.13，LongCat-Image得分86.8。该基准是评估文生图模型文本理解与复杂提示词跟随能力的重要指标。

* **T2I-CompBench**：

  * 地址：https://karine-h.github.io/T2I-CompBench/

  * 简介：一个全面的组合式文生图评测基准，包含6000条组合式文本提示，覆盖三大类别：属性绑定（颜色、形状、纹理）、物体关系（空间、非空间、复杂）和复杂组合。为每个类别提供专门的评估指标，被Hunyuan-DiT、Kolors、Seedream、SDXL等模型广泛采用。

* **WeGenBench**：

  * 地址：https://github.com/WeChatCV/WeGenBench ![](https://img.shields.io/github/stars/WeChatCV/WeGenBench.svg) | [论文](https://arxiv.org/abs/2606.20100)

  * 简介：微信视觉团队推出的中英双语文生图诊断基准，包含 4,000 条提示词，严格按 General / Text × 中文 / 英文各 1,000 条平衡，覆盖语义一致性、美学质量和视觉文字渲染，并为评分输出可解释理由。代码采用 Apache-2.0，提示数据采用 CC BY-NC 4.0。

* **CVTG-2K**：

  * 地址：https://github.com/NJU-PCALab/TextCrafter ![](https://img.shields.io/github/stars/NJU-PCALab/TextCrafter.svg)

  * 简介：Complex Visual Text Generation 2K，由 TextCrafter 团队构建，包含 2,000 条复杂视觉文字提示词，覆盖街景、广告和书籍封面等场景；每条提示包含 2～5 个文字区域，其中一半带尺寸、颜色和字体等风格属性。该基准用于评估复杂、多区域文字渲染，并非中文专项基准。

* **AnyText-benchmark**：

  * 地址：https://modelscope.cn/datasets/iic/AnyText-benchmark/summary

  * 简介：AnyText 系列的多语言视觉文字评测集。OCR 准确率部分包含中文 `wukong_word` 与英文 `laion_word` 各 1,000 张，FID 部分中英文各 40,000 张；2025 年 2 月更新了 AnyText2 使用的长描述版本。

* **ChineseWord**：

  * 简介：中文字符渲染准确性评测基准，覆盖全部8105个通用规范汉字，评估模型对中文字符的渲染准确率、稳定性以及对生僻字/复杂字形的支持能力。LongCat-Image（美团）在此基准上得分90.7，超越所有竞品，达到开源SOTA。该基准是衡量中文文生图模型文字渲染能力的核心指标。

* **C3 Benchmark**：

  * 地址：https://openreview.net/forum?id=7isO_QfcX55

  * 简介：Challenging Cross-Cultural Benchmark，由腾讯AI Lab提出，专门评估文生图模型生成中国及非西方文化场景图像的能力。包含500条挑战性提示词（C3），扩展版C3+包含9889条提示词，平均每条约40个词，远比标准基准复杂。是评估中文文化理解能力的重要基准。

* **OneIG-Bench**：

  * 地址：https://github.com/OneIG-Bench/OneIG-Benchmark ![](https://img.shields.io/github/stars/OneIG-Bench/OneIG-Benchmark.svg)

  * 简介：由StepFun提出的综合性文生图评测基准，从多个维度评估模型能力，包括文图对齐、文字渲染、推理能力、风格化和多样性等。Qwen-Image在OneIG-Bench上开源模型排名第一，提供了对图像生成质量的全面评估视角。

* **COCO-CN**：

  * 地址：https://github.com/li-xirong/coco-cn ![](https://img.shields.io/github/stars/li-xirong/coco-cn.svg)

  * 简介：最大的中英跨语言图文数据集，包含20342张图像，标注了27218条中文句子和70993个标签。虽然主要面向图像描述和检索任务设计，但可作为中文文生图模型计算FID和CLIP Score的参考数据集。Taiyi-XL在COCO-CN上的评测中超越了同类双语开源模型。

* **T2I-CoReBench**：

  * 地址：https://t2i-corebench.github.io/

  * 简介：由中国科学技术大学（USTC）提出的综合性评测基准，从12个维度评估文生图模型的组合能力和推理能力。超越简单的组合测试，进一步考察物理常识、时间理解等推理能力，是目前维度最全面的文生图评测基准之一。

* **UniGenBench++**：

  * 地址：https://arxiv.org/abs/2510.18701

  * 简介：由复旦大学提出的统一语义评测基准（2025），为每条提示词提供中英双语的短文本和长文本版本，采用层次化提示词结构评估文生图模型在多样场景和语言下的语义一致性。是少数同时覆盖中英文评测的基准之一，对评估双语文生图模型具有重要参考价值。

* **T2I-FactualBench**：

  * 地址：https://arxiv.org/abs/2412.04300

  * 简介：由阿里巴巴提出（ACL 2025），专门评测文生图模型对知识密集型概念的事实准确性。采用三层评估框架，包含8个领域的1600个概念和3000条提示词，考察模型是否能准确生成符合真实世界知识的图像（如历史建筑、科学概念、文化符号等）。

* **SpatialGenEval**：

  * 地址：https://arxiv.org/abs/2601.20354

  * 简介：2026年提出的空间智能评测基准，包含1230条长提示词，覆盖10个空间子领域（物体位置、布局、遮挡、因果关系等）。评估了20+文生图模型的空间理解和生成能力，是目前最全面的空间智能评测基准。

* **T2I-ConBench**：

  * 地址：https://openreview.net/forum?id=aR6QpqqIo9

  * 简介：2025年提出的持续后训练评测基准，聚焦于物品定制化和领域增强场景。从通用性保持、目标任务性能、灾难性遗忘和跨任务干扰四个维度分析文生图模型在持续训练过程中的表现。

* **T2ISafety**：

  * 地址：https://arxiv.org/abs/2501.12612

  * 简介：2025年提出的文生图安全性评测基准，从毒性、公平性和偏见三个维度评估模型安全性。包含12个任务、44个类别和68000张人工标注图像，是目前规模最大的文生图安全性评测数据集。

### 2.2 评测工具

* **ImageReward**：

  * 地址：https://github.com/zai-org/ImageReward ![](https://img.shields.io/github/stars/zai-org/ImageReward.svg)

  * 简介：由清华大学（THUDM）提出的首个通用人类偏好奖励模型（NeurIPS 2023），可作为文生图模型的评分函数。支持集成到微调流程中（ReFL），用于基于人类偏好的强化学习训练。可以对生成图像的质量、文图对齐度和美学表现进行自动化评分。

* **FlagEval**：

  * 地址：https://github.com/FlagOpen/FlagEval ![](https://img.shields.io/github/stars/FlagOpen/FlagEval.svg)

  * 简介：由BAAI（北京智源人工智能研究院）推出的开源AI大模型评测工具包。FlagEvalMM扩展版支持多模态和文生图评测，可全面评估视觉-语言理解和生成任务。提供标准化的评测流程和指标计算。

* **X-IQE**：

  * 地址：https://github.com/Schuture/Benchmarking-Awesome-Diffusion-Models ![](https://img.shields.io/github/stars/Schuture/Benchmarking-Awesome-Diffusion-Models.svg)

  * 简介：eXplainable Image Quality Evaluation，由港中深提出的基于MiniGPT-4的文生图扩散模型评测策略。使用COCO Caption和DrawBench作为提示词集，提供可解释的质量评估而非仅数值分数，帮助理解模型在不同维度上的表现差异。

* **TextPecker**：

  * 地址：https://github.com/CIawevy/TextPecker ![](https://img.shields.io/github/stars/CIawevy/TextPecker.svg) | [论文](https://arxiv.org/abs/2602.20903)

  * 简介：CVPR 2026 的视觉文字渲染专项 evaluator / reward 框架，可量化文字扭曲、模糊、错位等结构异常及语义一致性。已开放 InternVL3-8B、Qwen3-VL-8B 两个 evaluator、TextPecker-1.5M 数据集，以及训练、强化学习和评测代码（Apache-2.0）。

### 2.3 排行榜

* **LM Arena Text-to-Image**：

  * 地址：https://arena.ai/leaderboard/text-to-image

  * 简介：基于真实用户投票的Elo排名系统，是评估AI图像生成模型的重要参考。采用盲测对比方式，由用户选择更好的生成结果来计算排名。截至2026年7月5日，Text-to-Image Arena Overall包含72个模型、565万+投票；中文/国内相关模型中，qwen-image-2.0-pro-2026-06-22排名第12位（1193±8），Hunyuan Image 3.0排名第24位（1151±3），Seedream 4.5排名第28位（1146±3），Qwen-Image-2512排名第34位（1127±4），Z-Image-Turbo排名第49位（1081±6），GLM-Image排名第67位（1011±9）。

* **agicto.com 文生图排行榜**：

  * 地址：https://agicto.com/leaderboard/text-to-image

  * 简介：中文文生图模型排行榜，追踪和对比各文生图模型的排名表现。展示腾讯Hunyuan Image 3.0、字节Seedream等模型的排名情况，为中文用户提供直观的模型对比参考。


* **Artificial Analysis Text-to-Image**：
                                                                    
  * 地址：https://artificialanalysis.ai/text-to-image                                                                 
                                                                                                              
  * 简介：独立的AI模型分析平台，提供文生图模型的全面对比评测。从图像质量（Elo评分）、生成速度、价格等多维度对主流文生图模型进行基准测试和排名，覆盖FLUX、DALL-E、Stable Diffusion、Midjourney以及Hunyuan、Seedream等中国模型。提供交互式图表和详细的性能数据，帮助用户在质量、速度和成本之间做出选择。     

## 3. 数据集

### 3.1 开源训练数据集

* **WuKong（悟空数据集）**：

  * 地址：https://wukong-dataset.github.io/wukong-dataset/ | [HuggingFace](https://huggingface.co/datasets/wanng/wukong100m)

  * 简介：由华为诺亚方舟实验室创建的大规模中文多模态数据集，包含约1亿图文对。从百度搜索引擎利用100万个中文高频文本作为关键词进行图片搜索，获得接近20亿原始图文对，经过图片尺寸过滤（边长>200px，长宽比1/3~3）、无意义文本去除、短文本过滤、敏感词过滤等多步清洗后得到1亿高质量图文对。数据集分为256个文件，每个约8万对。是当时最大的中文开源多模态数据集，被Taiyi-SD、Taiyi-CLIP、WuKong-HuaHua（悟空画画）、PanGu-Draw等模型广泛使用。

* **Zero（零）**：

  * 地址：https://zero.so.com/

  * 简介：由奇虎360搜索引擎收集的中文图文对数据集，包含约2300万图文对，从50亿原始图文对中筛选而来。与悟空数据集一起被Taiyi-CLIP和Taiyi-SD用作预训练数据集，在A100x32上预训练24轮共6天。

* **LAION-5B中文子集**：

  * 地址：https://laion.ai/blog/laion-5b/ | [下载工具](https://github.com/opendatalab/laion5b-downloader) ![](https://img.shields.io/github/stars/opendatalab/laion5b-downloader.svg)

  * 简介：LAION-5B包含58.5亿CLIP过滤的图文对，其中LAION-2B-multi子集包含22.6亿非英文图文对，覆盖100+语言。LAION 官方统计中文（zh）图文对约 1.43 亿条，是其中较大的语言子集之一。数据从Common Crawl网页中收集，被MiaoBi、中文StableDiffusion-通用、AltDiffusion和VisCPM-Paint等模型使用。

* **TaiSu（太素）**：

  * 地址：https://github.com/ksOAn6g5/TaiSu ![](https://img.shields.io/github/stars/ksOAn6g5/TaiSu.svg)

  * 简介：大规模高质量中文多模态数据集，包含约1.66亿图文对。采用自动化数据获取和清洗框架构建，旨在填补大规模高质量中文图文数据的空白。可用于中文视觉-语言预训练研究。

* **WuDaoMM（悟道多模态语料）**：

  * 地址：https://arxiv.org/abs/2203.11480

  * 简介：由BAAI（北京智源人工智能研究院）作为悟道项目的一部分创建的大规模多模态语料库，包含超过6.5亿图文对，涵盖中文和多语言内容。AltCLIP和AltDiffusion使用了悟道数据集（结合LAION）进行训练，CogView系列也使用了悟道数据。

* **AnyWord-3M**：

  * 地址：https://modelscope.cn/datasets/iic/AnyWord-3M/summary

  * 简介：AnyText 系列使用的多语言视觉文字训练集。V1.1 共整理 3,034,486 张图像和 900 万余行文字，约含 160 万张中文、139 万张英文及 1 万张其他语言图像，并补充了 AnyText2 所需的长描述与文字颜色标注。数据来源混合多个公开数据集，使用时仍需核对原始图像的授权条件。

### 3.2 评测/标注数据集

* **COCO-CN**：

  * 地址：https://github.com/li-xirong/coco-cn ![](https://img.shields.io/github/stars/li-xirong/coco-cn.svg)

  * 简介：最大的中英跨语言图文数据集，基于MS-COCO构建，包含20342张图像，标注了27218条中文句子和70993个标签。支持图像标注、描述和检索任务，也可作为中文文生图模型计算FID和CLIP Score的参考数据集。Taiyi-XL在COCO-CN评测中超越了同类双语开源模型。

* **AI Challenger图像中文描述（AIC-ICC）**：

  * 地址：https://github.com/ruotianluo/Image_Captioning_AI_Challenger ![](https://img.shields.io/github/stars/ruotianluo/Image_Captioning_AI_Challenger.svg)

  * 简介：AI Challenger大规模数据集的一部分，包含约30万张图像及对应的中文描述标注（21万训练集、3万验证集和6万测试集）。是最早的大规模中文图像描述数据集之一，可用于中文图像描述和视觉-语言模型的评测。

* **MUGE（多模态理解与生成评测）**：

  * 地址：https://github.com/MUGE-2021/image-retrieval-baseline ![](https://img.shields.io/github/stars/MUGE-2021/image-retrieval-baseline.svg)

  * 简介：由阿里达摩院认知智能团队推出的大规模中文多模态评测基准，覆盖图文检索、图像描述等多模态任务。MUGE检索基线后来被Chinese-CLIP所超越。为中文多模态模型提供了标准化的评测平台。



## Star History

<a href="https://star-history.com/#leeguandong/Awesome-Chinese-Stable-Diffusion&Date">

  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=leeguandong/Awesome-Chinese-Stable-Diffusion&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=leeguandong/Awesome-Chinese-Stable-Diffusion&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=leeguandong/Awesome-Chinese-Stable-Diffusion&type=Date" />
  </picture>

</a>

## License

This project is licensed under the [MIT License](LICENSE).
