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

* Taiyi-xl-3.5B：

  * 地址：https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-XL-3.5B

  * 简介：文生图模型如谷歌的Imagen、OpenAI的DALL-E 3和Stability AI的Stable Diffusion引领了AIGC和数字艺术创作的新浪潮。然而，基于SD v1.5的中文文生图模型，如Taiyi-Diffusion-v0.1和Alt-Diffusion的效果仍然一般。中国的许多AI绘画平台仅支持英文，或依赖中译英的翻译工具。目前的开源文生图模型主要支持英文，双语支持有限。我们的工作，Taiyi-Diffusion-XL（Taiyi-XL），在这些发展的基础上，专注于保留英文理解能力的同时增强中文文生图生成能力，更好地支持双语文生图。

    Taiyi-Diffusion-XL文生图模型训练主要包括了3个阶段。首先，我们制作了一个高质量的图文对数据集，每张图片都配有详细的描述性文本。为了克服网络爬取数据的局限性，我们使用先进的视觉-语言大模型生成准确描述图片的caption。这种方法丰富了我们的数据集，确保了相关性和细节。然后，我们从预训练的英文CLIP模型开始，为了更好地支持中文和长文本我们扩展了模型的词表和位置编码，通过大规模双语数据集扩展其双语能力。训练涉及对比损失函数和内存高效的方法。最后，我们基于Stable-Diffusion-XL，替换了第二阶段获得的text encoder，在第一阶段获得的数据集上进行扩散模型的多分辨率、多宽高比训练。

    我们的机器评估包括了对不同模型的全面比较。评估指标包括CLIP相似度（CLIP Sim）、IS和FID，为每个模型在图像质量、多样性和与文本描述的对齐方面提供了全面的评估。在英文数据集（COCO）中，Taiyi-XL在所有指标上表现优异，获得了最好的CLIP Sim、IS和FID得分。这表明Taiyi-XL在生成与英文文本提示紧密对齐的图像方面非常有效，同时保持了高图像质量和多样性。同样，在中文数据集（COCO-CN）中，Taiyi-XL也超越了其他模型，展现了其强大的双语能力。

    尽管Taiyi-XL可能还未能与商业模型相媲美，但它比当前双语开源模型优越不少。我们认为我们模型与商业模型的差距主要归因于训练数据的数量、质量和多样性的差异。我们的模型仅使用学术数据集和符合版权要求的图文数据进行训练，未使用Midjourney和DALL-E 3等生成数据。XL版本模型，如SD-XL和Taiyi-XL，在1.5版本模型如SD-v1.5和Alt-Diffusion上显示出显著改进。DALL-E 3以其生动的色彩和prompt-following的能力而著称。Taiyi-XL模型偏向生成摄影风格的图片，与Midjourney较为类似，但是Taiyi-XL并在双语（中英文）文生图生成方面表现更出色。  
    
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

* WuKong-HuaHua：

  * 地址：https://github.com/JeffDing/WuKong-HuaHua
    ![](https://img.shields.io/github/stars/JeffDing/WuKong-HuaHua.svg)

  * 简介： Wukong-Huahua是基于扩散模型的中文以文生图大模型，由华为诺亚团队携手中软分布式并行实验室、昇腾计算产品部联合开发，使用昇思框架(MindSpore)+昇腾(Ascend)软硬件解决方案实现。该模型是基于目前最大的中文开源多模态数据集悟空数据集进行训练得来，具备优秀的中文文本-图像生成能力，能够实现多场景的绘画风格，可生成高质量的图像，给用户带来良好的使用体验。

    在1.0的基础上Wukong-Huahua模型基于华为MindSpore平台+昇腾硬件910进行大规模多机多卡训练，在新数据集上进行训练升级到2.0版本。相比于原版本，新版本大幅提升画质、艺术性和推理速度，更新内容包括以下3点：1.提升输出分辨率，2.0模型目前可以支持更高分辨率图形输出，从1.0版本的512x512提升到768x768，大图更清晰。2.采用自研Multistep-SDE采样加速推理技术，采样步数从原先的50步采样降到20-30步，加速2-3倍。3.采用自研RLAIF算法，提升生成图片的画质以及艺术性表达。

    悟空画画模型分别由中文文本编码器以及Stable Diffusion生成模型组成。具体的训练方法如下：

    1.预训练中文图文判别模型，得到一个具有中文图文对齐能力的文本编码器；

    2.我们结合Stable Diffusion图像生成模型和第一步训练得到的文本编码器，在悟空中文多模态数据集上进行训练，得到中文文图生成模型——悟空画画模型。

    悟空画画模型的训练依赖于悟空数据集，它是当时已开源的最大规模的中文多模态数据集。我们首先在百度搜索引擎上利用一百万个中文高频文本作为关键词进行图片搜索，获得接近20亿的原始图文对数据，此时这部分数据中包含了大量的噪声。第二步我们对这些原始数据进行多种方式的筛选清洗，主要操作包括：

    - 对图片的尺寸进行过滤，去除边长小于200px或者长宽比超出1/3~3范围的样本
    - 去除文本为无意义的词如 “Image”, “图片”，“照片”等的样本
    - 过滤文本长度过短，文本出现频次过高（如“如下图所示”等描述文本）的样本
    - 过滤文本中包含隐私/敏感词的样本
    
    最终我们经过过滤得到了一亿较高质量中文图文对。进一步地，在训练悟空画画模型时，我们对悟空数据集的数据根据图文匹配分数、水印分数以及艺术性分数https://github.com/christophschuhmann/improved-aesthetic-predictor 再次进行筛选，最终获得25M左右的数据进行训练。该部分数据具有较高的图像质量，并对常见文本内容进行了良好的覆盖，使得训练得到的悟空画画模型对文本拥有广泛的识别能力，并能根据不同的提示词生成多样的图片风格。

* PanGu-Draw：

  * 地址：https://github.com/mindspore-lab/mindone/blob/master/examples/pangu_draw_v3
    ![](https://img.shields.io/github/stars/mindspore-lab/mindone.svg)

  * 简介： 

    * 网络结构扩容，参数量从1B扩大到5B，是当前**业界最大的中文文生图模型**；
    *  支持**中英文双语**输入；
    *  提升输出分辨率，支持**原生1K输出**（v1->v2->v3: 512->768->1024）；
    *  多尺寸（16:9、4:3、2:1...）输出；
    *  **可量化的风格化调整**：动漫、艺术性、摄影控制；
    *  基于**昇腾硬件和昇思平台**进行大规模多机多卡训练、推理，全自研昇思MindSpore平台和昇腾Ascend硬件；
    *  采用**自研RLAIF**提升画质和艺术性表达。

* MiaoBi：

  * 地址：https://github.com/ShineChen1024/MiaoBi
    ![](https://img.shields.io/github/stars/ShineChen1024/MiaoBi.svg)

  * 简介： 妙笔的测试版本。妙笔，一个中文文生图模型，与经典的stable-diffusion 1.5版本拥有一致的结构，兼容现有的lora，controlnet，T2I-Adapter等主流插件及其权重。       

    妙笔的训练数据包含Laion-5B中的中文子集（经过清洗过滤），Midjourney相关的开源数据（将英文提示词翻译成中文），以及我们收集的一批数十万的caption数据。由于整个数据集大量缺少成语与古诗词数据，所以对成语与古诗词的理解可能存在偏差，对中国的名胜地标建筑数据的缺少以及大量的英译中数据，可能会导致出现一些对象的混乱。妙笔Beta0.9在8张4090显卡上完成训练，我们正在拓展我们的机器资源来训练SDXL来获得更优的结果。

* 腾讯混元DiT：

  * 地址：https://github.com/Tencent/HunyuanDiT
    ![](https://img.shields.io/github/stars/Tencent/HunyuanDiT.svg)

  * 混元DiT，一个基于Diffusion transformer的文本到图像生成模型，此模型具有中英文细粒度理解能力。为了构建混元DiT，我们精心设计了Transformer结构、文本编码器和位置编码。我们构建了完整的数据管道，用于更新和评估数据，为模型优化迭代提供帮助。为了实现细粒度的文本理解，我们训练了多模态大语言模型来优化图像的文本描述。最终，混元DiT能够与用户进行多轮对话，根据上下文生成并完善图像。
    
    Hunyuan-DiT是一个在潜在空间中的扩散模型。基于潜在扩散模型，使用预训练的变分自编码器（VAE）将图片压缩到低维度的潜在空间，并训练一个扩散模型来学习数据分布。我们的扩散模型是用transformer参数化的。为了编码文本提示，我们利用了预训练的双语（英语和中文）CLIP和多语言T5编码器的组合。混元DiT提供双语生成能力，中国元素理解具有优势。混元DiT能分析和理解长篇文本中的信息并生成相应艺术作品。混元DiT能捕捉文本中的细微之处，从而生成完美符合用户需要的图片。混元DiT可以在多轮对话中通过与用户持续协作，精炼并完善的创意构想。性能上超过SDXL，Playground 2.5等。

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

* 快手可画Kolors：

  * 地址：https://www.51cto.com/article/767164.html

  * 简介：它有着三大突出特点：强大的文本理解、丰富的细节刻画，以及多样的风格转化。

    首先，强大的文本理解能力。快手AI构建了数十亿的图文训练数据,数据来自开源社区、内部构建和自研AI技术合成。这些数据覆盖了常见的三千万中文实体概念,兼具世界知识。在此基础上训练研发了一个强大的中文CLIP模型，不仅懂我们的语言，也更懂中文世界的图像；其次,快手AI利用自研的中文LLM，融合CLIP的图文特征作为文生图的文本理解模块，不但实现了中文特色概念的理解,更解决了复杂概念、属性混淆等文生图领域常见问题。

    其次，丰富的细节刻画。快手AI研究团队更改了去噪算法的底层公式和加噪公式；同时精选了一批高细节、高美感的优质数据，在模型学习的后期进行有侧重学习。实现了单一基座模型在主体完整的前提下，可生成具有丰富细节和纹理的图片。同时，基座模型也实现了输入图片，输出细节丰富图片的图生图能力。

    第三,多样的风格转化。可图大模型具有基于Prompt的自动学习模型，基于知识的理解与扩充，为用户提供不同的风格模版。依据提示词自动扩充模块，可以丰富化用户描述,包括风格、构图、视觉要素等。配合强大的文生图基座模型，Kolors 可以帮助用户准确理解自己的需求，通过简单描述即可生成多样化风格的图片。

* 阿里通义万相：

  * 地址：https://www.jiqizhixin.com/articles/2023-07-07-6

  * 简介：通义万相基于阿里自研的组合式生成模型 Composer，它拥有 50 亿参数，并在数十亿个文本、图像对上进行训练。在业界都在考虑如何提升 AI 绘画模型的可控性这一点上，Composer 给出了它的创新性思路。

    通过一个基于扩散模型的「组合式生成」框架，Composer 能够对配色、布局、风格等图像设计元素进行拆解和组合，实现了高度可控性和极大自由度的图像生成效果。所谓拆解 - 组合，首先将图像分解为不同的设计元素，比如配色、草图、布局、风格、语义、材质等。然后使用 AI 模型将这些设计元素重新组合成新的图像。这里，拆解 - 组合过程中允许对用到的元素自由修改编辑，如此一来可控性大大增强。
    
    正是基于 Composer 框架，通义万相才能让我们体验到相似图生成和风格迁移这两种图生图功能。一边用图像理解模型将图像拆解为不同元素，一边用扩散模型将这些元素重新组合成新图像，双管齐下，图生图水到渠成。其中对于相似图生成，保持图像语义内容不变，仅仅改变图像中的局部细节，就能生成相似图片。过程中既可以较好地保持原图主体一致性，还提升了生成图的多样性和质量。对于风格迁移，一方面保留原图的基本形态、结构，另一方面将目标风格图片的风格、色彩、笔触等个性化信息，最终实现风格迁移。
    
    
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
