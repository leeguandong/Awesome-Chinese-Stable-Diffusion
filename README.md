<h1 align="center">
Awesome-Chinese-Stable-Diffusion
</h1>
<p align="center">
<font face="黑体" color=orange size=5"> An Awesome Collection for SD in Chinese </font>
</p>
<p align="center">
<font face="黑体" color=orange size=5"> 收集和梳理中文SD相关 </font>
</p>
<p align="center">
  <a href="https://github.com/leeguandong/Awesome-Chinese-Stable-Diffusion/stargazers"> <img src="https://img.shields.io/github/stars/leeguandong/Awesome-Chinese-Stable-Diffusion.svg?style=popout-square" alt="GitHub stars"></a>
  <a href="https://github.com/leeguandong/Awesome-Chinese-Stable-Diffusion/issues"> <img src="https://img.shields.io/github/issues/leeguandong/Awesome-Chinese-Stable-Diffusion.svg?style=popout-square" alt="GitHub issues"></a>
  <a href="https://github.com/leeguandong/Awesome-Chinese-Stable-Diffusion/forks"> <img src="https://img.shields.io/github/forks/leeguandong/Awesome-Chinese-Stable-Diffusion.svg?style=popout-square" alt="GitHub forks"></a>
</p>

本项目旨在收集和梳理中文Stable-Diffusion相关的开源模型、应用、数据集及教程等资料，主要是有中文的模型新数据和算法！

如果本项目能给您带来一点点帮助，麻烦点个⭐️吧～

同时也欢迎大家贡献本项目未收录的开源模型、应用、数据集等。提供新的仓库信息请发起PR，并按照本项目的格式提供仓库链接、star数，简介等相关信息，感谢~


## 目录
- [目录](#目录)
  - [1. 中文文生图模型](#1-中文文生图模型)
    - [1.1 开源模型](#11-开源模型)
    - [1.2 闭源模型](#12-闭源模型)
    
    
  
- [Star History](#star-history)

###  1. <a name='模型'></a>中文文生图模型

#### 1.1 开源模型

* SkyPaint：
  * 地址：https://github.com/SkyWorkAIGC/SkyPaint-AI-Diffusion
  ![](https://img.shields.io/github/stars/SkyWorkAIGC/SkyPaint-AI-Diffusion.svg)
  * 简介：SkyPaint文本生成图片模型主要由两大部分组成，即提示词文本编码器模型和扩散模型两大部分。因此我们的优化也分为两步： 首先，基于[OpenAI-CLIP](https://github.com/openai/CLIP)优化了提示词文本编码器模型使得SkyPaint具有中英文识别能力， 然后，优化了扩散模型，使得SkyPaint具有现代艺术能力可以产生高质量图片。
  
* Pai-Diffusion
  * 地址：https://github.com/alibaba/EasyNLP
    ![](https://img.shields.io/github/stars/alibaba/EasyNLP.svg)
  * 简介：由于现有Diffusion模型主要使用英文数据进行训练，如果直接使用机器翻译将英文数据翻译成中文进行模型训练，因为中英文在文化和表达上具有很大的差异性，产出的模型通常无法建模中文特有的现象。此外，通用的StableDiffusion模型由于数据源的限制，很难用于生成特定领域、特定场景下的高清图片。PAI-Diffusion系列模型由阿里云机器学习（PAI）团队发布并开源，除了可以用于通用文图生成场景，还具有一系列特定场景的定制化中文Diffusion模型，包括古诗配图、二次元动漫、魔幻现实等。在下文中，我们首先介绍PAI-Diffusion的模型Pipeline架构，包括中文CLIP模型、Diffusion模型、图像超分模型等。

* 中文StableDiffusion-通用领域：
  * 地址：https://modelscope.cn/models/damo/multi-modal_chinese_stable_diffusion_v1.0/summary
  * 简介：本模型采用的是[Stable Diffusion 2.1模型框架](https://github.com/Stability-AI/stablediffusion)，将原始英文领域的[OpenCLIP-ViT/H](https://github.com/mlfoundations/open_clip)文本编码器替换为中文CLIP文本编码器[chinese-clip-vit-huge-patch14](https://github.com/OFA-Sys/Chinese-CLIP)，并使用大规模中文图文pair数据进行训练。训练过程中，固定中文CLIP文本编码器，利用原始Stable Diffusion 2.1 权重对UNet网络参数进行初始化、利用64卡A100共训练35W steps。训练数据包括经中文翻译的公开数据集（LAION-400M、cc12m、Open Images）、以及互联网搜集数据，经过美学得分、图文相关性等预处理进行图像过滤，共计约4亿图文对。
  
* 文本到图像生成扩散模型-中英文-通用领域-tiny：
  * 地址：https://modelscope.cn/models/damo/cv_diffusion_text-to-image-synthesis_tiny/summary
  * 简介：文本到图像生成模型由文本特征提取与扩散去噪模型两个子网络组成。文本特征提取子网络为StructBert结构，扩散去噪模型为unet结构。通过StructBert提取描述文本的语义特征后，送入扩散去噪unet子网络，通过迭代去噪的过程，逐步生成复合文本描述的图像。训练数据包括LAION400M公开数据集，以及互联网图文数据。文本截断到长度64 (有效长度62)，图像缩放到64x64进行处理。模型分为文本特征提取与扩散去噪模型两个子网络，训练也是分别进行。文本特征提取子网络StructBert使用大规模中文文本数据上预训练得到。扩散去噪模型则使用预训练StructBert提取文本特征后，与图像一同训练文本到图像生成模型。
  
* 通义-文本生成图像大模型-中英文-通用领域：
  * 地址：https://www.modelscope.cn/models/damo/cv_diffusion_text-to-image-synthesis/summary
  * 简介：本模型基于多阶段文本到图像生成扩散模型, 输入描述文本，返回符合文本描述的2D图像。支持中英双语输入。文本到图像生成扩散模型由特征提取、级联生成扩散模型等模块组成。整体模型参数约50亿，支持中英双语输入。通过知识重组与可变维度扩散模型加速收敛并提升最终生成效果。训练数据包括LAION5B, ImageNet, FFHQ, AFHQ, WikiArt等公开数据集。经过美学得分、水印得分、去重等预处理进行图像过滤。模型分为文本特征提取、文本特征到图像特征生成、级联扩散生成模型等子网络组成，训练也是分别进行。文本特征提取使用大规模图文样本对数据上训练的CLIP的文本分支得到。文本到图像特征生成部分采用GPT结构，是一个width为2048、32个heads、24个blocks的Transformer网络，利用causal attention mask实现GPT预测。64x64、256x256、1024x1024扩散模型均为UNet结构，在64x64、256x256生成模型中使用了Cross Attention嵌入image embedding条件。为降低计算复杂度，在256扩散模型训练过程中，随机64x64 crop、128x128 crop、256x256 crop进行了multi-grid训练，来提升生成质量；在1024扩散模型中，对输入图随机256x256 crop。

* Taiyi：

  * 地址：https://github.com/IDEA-CCNL/Fengshenbang-LM
    ![](https://img.shields.io/github/stars/IDEA-CCNL/Fengshenbang-LM.svg)

  * 简介：Taiyi-clip：我们遵循CLIP的实验设置，以获得强大的视觉-语言表征。在训练中文版的CLIP时，我们使用[chinese-roberta-wwm](https://link.zhihu.com/?target=https%3A//huggingface.co/hfl/chinese-roberta-wwm-ext)作为语言的编码器，并将[open_clip](https://link.zhihu.com/?target=https%3A//github.com/mlfoundations/open_clip)中的**ViT-L-14**应用于视觉的编码器。为了快速且稳定地进行预训练，我们**冻结了视觉编码器并且只微调语言编码器**。此外，我们将[Noah-Wukong](https://link.zhihu.com/?target=https%3A//wukong-dataset.github.io/wukong-dataset/)数据集(100M)和[Zero](https://link.zhihu.com/?target=https%3A//zero.so.com/)数据集(23M)用作预训练的数据集。在悟空数据集和zero数据集上预训练24轮,在A100x32上训练了6天。

    Taiyi-SD：我们将[Noah-Wukong](https://link.zhihu.com/?target=https%3A//wukong-dataset.github.io/wukong-dataset/)数据集(100M)和[Zero](https://link.zhihu.com/?target=https%3A//zero.so.com/)数据集(23M)用作预训练的数据集，先用[IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese](https://link.zhihu.com/?target=https%3A//huggingface.co/IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese)对这两个数据集的图文对相似性进行打分，取CLIP Score大于0.2的图文对作为我们的训练集。 我们使用[IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese](https://link.zhihu.com/?target=https%3A//huggingface.co/IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese)作为初始化的text encoder，冻住[stable-diffusion-v1-4](https://link.zhihu.com/?target=https%3A//huggingface.co/CompVis/stable-diffusion-v1-4)([论文](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2112.10752))模型的其他部分，**只训练text encoder**，以便保留原始模型的生成能力且实现中文概念的对齐。该模型目前在0.2亿图文对上训练了一个epoch。 我们在 32 x A100 训练了大约100小时。
    补充: clip和sd的微调阶段都只调text encoder部分

* AltDiffusion:

  * 地址：https://github.com/FlagAI-Open/FlagAI
    ![](https://img.shields.io/github/stars/FlagAI-Open/FlagAI.svg)

  * 简介：AltClip：AltCLIP基于 [OpenAI CLIP](https://link.zhihu.com/?target=https%3A//github.com/openai/CLIP) 训练，训练数据来自 [WuDao数据集](https://link.zhihu.com/?target=https%3A//data.baai.ac.cn/details/WuDaoCorporaText) 和 [LAION](https://link.zhihu.com/?target=https%3A//huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6plus)，训练共有两个阶段。 在平行知识蒸馏阶段，我们只是使用平行语料文本来进行蒸馏（平行语料相对于图文对更容易获取且数量更大）。在双语对比学习阶段，我们使用少量的中-英图像-文本对（一共约2百万）来训练我们的文本编码器以更好地适应图像编码器。

    AltSD：基于 stable-diffusion v1-4 作为初始化，并使用 AltCLIP 或 AltCLIPM9 作为text encoder。在微调过程中，除了跨注意力块的键和值投影层之外，我们冻结了扩散模型中的所有参数。训练数据来自 [WuDao数据集](https://link.zhihu.com/?target=https%3A//data.baai.ac.cn/details/WuDaoCorporaText) 和 [LAION](https://link.zhihu.com/?target=https%3A//huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6plus)。

* VisCPM-Paint：

  * 地址：https://github.com/OpenBMB/VisCPM
    ![](https://img.shields.io/github/stars/OpenBMB/VisCPM.svg)

  * 简介：VisCPM-Paint支持中英双语的文到图生成。该模型使用CPM-Bee（10B）作为文本编码器，使用UNet作为图像解码器，并通过扩散模型训练目标融合语言和视觉模型。在训练过程中，语言模型参数始终保持固定。我们使用[Stable Diffusion 2.1](https://github.com/Stability-AI/stablediffusion)的UNet参数初始化视觉解码器，并通过逐步解冻其中关键的桥接参数将其与语言模型融合。该模型在[LAION 2B](https://laion.ai/)英文图文对数据上进行了训练。

    与VisCPM-Chat类似，我们发现得益于CPM-Bee的双语能力，VisCPM-Paint可以仅通过英文图文对训练，泛化实现良好的中文文到图生成能力，达到中文开源模型的最佳效果。通过进一步加入20M清洗后的原生中文图文对数据，以及120M翻译到中文的图文对数据，模型的中文文到图生成能力可以获得进一步提升。我们在标准图像生成测试集MSCOCO上采样了3万张图片，计算了常用评估图像生成指标FID (Fréchet Inception Distance)评估生成图片的质量。我们同样提供了两个模型版本，分别为VisCPM-Paint-balance和VisCPM-Paint-zhplus，前者在英文和中文两种语言上的能力较为平衡，后者在中文能力上更加突出。VisCPM-Paint-balance只使用了英文图文对进行训练，VisCPM-Paint-zhplus在VisCPM-Paint-balance基础上增加了20M原生中文图文对数据和120M翻译到中文的图文对数据进行训练。


#### 1.2 闭源模型

* 腾讯混元
  * 地址：https://mp.weixin.qq.com/s/hEqVR89qDyMckld-OikDPQ
  * 简介：大模型文生图的难点体现在对提示词的语义理解，生成内容的合理性以及生成图片的效果，针对这三个技术难点，腾讯进行了专项的技术研究，提出了一系列原创算法，来保证生成图片的可用性和画质。 

    1、在语义理解方面，腾讯混元采用了中英文双语细粒度的模型，模型同时建模中英文实现双语理解，而不是通过翻译，通过优化算法提升了模型对细节的感知能力与生成效果，有效避免多文化差异下的理解错误。

    2、在内容合理性方面，AI生成人体结构和手部经常容易变形。混元文生图通过增强算法模型的图像二维空间位置感知能力，并讲人体骨架和人手结构等先验信息引入到生成过程中，让生成的图像结构更合理，减少错误率。

    3、在画面质感方面，混元文生图基于多模型融合的方法，提升生成质感。经过模型算法的优化之后，混元文生图的人像模型，包含发丝、皱纹等细节的效果提升了30%，场景模型，包含草木、波纹等细节的效果提升了25%。

* 美图MiracleVision：
  * 地址：https://mp.weixin.qq.com/s/Hixjc6x-L-Zd5JLBjVXCZA
  * 简介：美图自研大模型名叫**MiracleVision**（奇想智能）。其最显著的特点是更懂美学。美图把长期积累的美学认知融入MiracleVision视觉大模型，并搭建了基于机器学习的美学评估系统，为模型生成结果打上“美学分数”，从而不断地提升模型对美学的理解。

* 网易丹青：
  * 地址：https://zhuanlan.zhihu.com/p/648712812

  * 简介：丹青模型基于原生中文语料数据及网易自有高质量图片数据训练，与其他文生图模型相比，丹青模型的差异化优势在于对中文的理解能力更强，对中华传统美食、成语、俗语、诗句的理解和生成更为准确。比如，丹青模型生成的图片中，鱼香肉丝没有鱼，红烧狮子头没有狮子。基于对中文场景的理解，丹青模型生成的图片更具东方美学，能生成“飞流直下三千尺”的水墨画，也能生成符合东方审美的古典美人。

    基于数据集和理解模型，网易伏羲对图文生成算法进行重构，依托于扩散模型的原理，在广泛的（8 亿）图文数据上训练以达到较好的生成结果。具体来说，丹青模型侧重文本与图片的交互，强化了在文图引导部分的参数作用，能够让文本更好地引导图片的生成，因此生成的结果也更加贴近用户意图。同时，丹青模型进行了图片多尺度的训练，充分考虑图片的不同尺寸和清晰度问题，将不同尺寸和分辨率的图片进行分桶。在充分保证训练图片训练的不失真的前提下，保留尽可能多的信息，适应不同分辨率的生成。

    在数据策略方面，丹青模型在初始阶段使用亿级别的广泛分布的数据，不仅在语义理解上具有广泛性，可以很好地理解一些成语、古文诗句，在生成的画风上也具有多样性，可以生成多种风格。在之后的阶段，丹青模型分别从图文关联度、图片清晰度、图片美观度等多个层面进行数据筛选，以优化生成能力，生成高质量图片。

    此外，丹青模型在训练和生成阶段还引入了人工反馈。在训练阶段，人工从多个维度的评估，筛选出来大批高质量图文匹配、高美观度数据，以补足自动流程缺失能力，帮助基础模型获得更好的效果；在生成阶段，人工对模型的语义生成能力和图片美观度进行评分，筛选出大批量优质生成的结果，引入模型当做正反馈，实现数据闭环。

* 腾讯太极：

  * 地址：https://zhuanlan.zhihu.com/p/590459240

  * 简介：1.太极-Imagen文生图模型：团队成员对Imagen模型进行了实现和改进，主要采用自研的中文文本编码器，优化模型训练过程，结合latent diffusion model优化超分辨率模型训练过程，在内部亿级别的中文场景数据上进行训练，获得了在中文场景下自研文生图模型。

    中文文本编码器：在训练Imagen模型的过程中，我们发现文本编码器对于生成模型的语义理解至关重要，在英文场景中Imagen采用了T5-XXL作为文本编码器并通过固定了文本编码器训练生成模型的方法使得模型具有强大的文本理解能力。在中文场景中，我们采用自研的混元sandwich模型作为文本编码器，该文本编码器在中文场景中强大的语义理解能力为中文文生图模型的训练奠定了良好的基础。同时，Imagen模型训练过程中，我们发现文本embedding和Imagen模型参数的匹配也对生成结果起了至关重要的作用。在模型训练的第一阶段，我们首先固定文本编码器，训练diffusion模型的参数，通过文本embedding来指导模型的生成结果。当第一阶段训练收敛后，我们发现模型对于中文场景的实体，物体关系等已经有了较好的理解，但是对于更难，更细粒度的语义提升困难。因此在第二阶段，我们通过放开文本编码器的参数，将其与diffusion model一起进行端到端的训练，能够进一步提升模型对于细粒度语义的理解。

    多阶段不同分辨率级联生成：Imagen通过级联的diffusion模型生成不同分辨率的图像，其中第一阶段的模型生成64x64分辨率的图像，第二阶段和第三阶段分别生成256x256分辨率和1024x1024分辨率的图像。通过多阶段级联的结构，可以使得第一阶段模型的文生图模型训练更加的高效。

    文生图大模型训练策略优化：最后，针对自研Imagen，我们训练了不同参数量和大小的模型。我们首先训练了u-net核心参数量为3亿的模型，已经能够获得中文场景下不错的效果，之后我们将模型规模扩大到核心参数量为13亿，基于团队在太极-DeepSpeed的大规模预训练加速优化技术，在亿级数据上，32*A100只需要2周时间即可收敛。经过实验对比，13亿参数的大模型比3亿参数模型在生成图像细节和语义捕获能力上都获得了更好的效果。

    2.太极-sd文生图模型：在中文场景的SD训练中，一方面对文本编码器进行了替换，将其从原生的CLIP替换为自研的中文太极-ImageCLIP图文匹配模型，并且在训练过程中，优先对文本encoder部分进行训练，以保留SD预训练模型的生成能力；另一方面，为了提升模型对于文本内语义，数量，实体等不同方面的捕捉能力，我们综合了太极-ImageCLIP和混元-Sandwich两类不同的中文encoder所生成的embedding，来指导图片的生成；最后，为了更好的捕捉长文本的信息，我们还将池化后的文本embedding也融合进u-net中，提升整体的生成效果。
    
###  2. <a name='评测'></a>测评



###  3. <a name='数据集'></a>数据集



## Star History

<a href="https://star-history.com/#leeguandong/Awesome-Chinese-Stable-Diffusion&Date">

  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=leeguandong/Awesome-Chinese-Stable-Diffusion&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=leeguandong/Awesome-Chinese-Stable-Diffusion&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=leeguandong/Awesome-Chinese-Stable-Diffusion&type=Date" />
  </picture>

</a>
