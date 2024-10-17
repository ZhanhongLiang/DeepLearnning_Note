# 背景

>因为ASR中的CPC/ Wav2Vec/ Vq-Wav2Vec /Wav2Vec2.0 /HuBERT /Wav2Vec-BERT里面都涉及了对比学习,今天需要单独来Contrastive Learning的知识
>
>`Contrastive Learning is sort of unsupervised learning`

# 综述部分

>可以参考B站up主 bryanyzhu的Contrastive Learning的review
>
>[Contrastive Learning Review](https://www.bilibili.com/video/BV19S4y1M7hm?spm_id_from=333.788.videopod.sections&vd_source=b5788d06ba0855a5ace28010dd8907be)
>
>[Review Paper](https://github.com/mli/paper-reading/?tab=readme-ov-file)

# 背景知识

## Example

![Illustration Example](https://tobiaslee.top/img/dollar.png)

>当你被要求画一张美元的时候,*左边是没有钞票在你面前，右边是面前摆着一张钞票画出来的结果*
>
>表达的一个核心思想就是：尽管我们已经见过很多次钞票长什么样子，但我们很少能一模一样的画出钞票；虽然我们画不出栩栩如生的钞票，但我们依旧可以轻易地辨别出钞票。基于此，也就意味着**表示学习算法并不一定要关注到样本的每一个细节，只要学到的特征能够使其和其他样本区别开来就行**，这就是对比学习和对抗生成网络（GAN）的一个主要不同所在。

## Framework

>既然是表示学习，那么我们的核心就是要学习一个映射函数 $$f$$，把样本 $$x$$ 编码成其表示$$f(x)$$ ，对比学习的核心就是使得这个 $$f$$满足下面这个式子：

$$
s\left(f(x), f\left(x^{+}\right)\right) \gg s\left(f(x), f\left(x^{-}\right)\right)
$$

>这里的 $$x^{+}$$ 就是和 $$x$$ 类似的样本，$$x^{-}$$ 就是和 $$x$$ 不相似的样本，$$s(⋅,⋅)$$ 这是一个度量样本之间相似程度的函数，`一个比较典型的 score 函数就是就是向量内积`，即优化下面这一期望：

$$
\underset{x, x^{+}, x^{-}}{\mathbb{E}}\left[-\log \left(\frac{e^{f(x)^{T} f\left(x^{+}\right)}}{e^{f(x)^{T} f\left(x^{+}\right)}+e^{f(x)^{T} f\left(x^{-}\right)}}\right)\right]
$$

>如果对于一个 $$x$$，我们有 1 个正例和 $$N-1$$ 个负例，那么这个 loss 就可以看做是一个 N 分类问题，实际上就是一个交叉熵，而这个函数在对比学习的文章中被称之为 `InfoNCE`。事实上，最小化这一 loss 能够最大化 $$f(x)$$ 和 $$f(x^{+})$$ 互信息的下界，让二者的表示更为接近。理解了这个式子其实就理解了整个对比学习的框架，后续研究的核心往往就聚焦于这个式子的两个方面：

>- 如何定义目标函数？最简单的一种就是上面提到的内积函数，另外一中 triplet 的形式就是 $$l=max(0,η+s(x,x^{+})−s(x,x^{-}))$$ ，直观上理解，就是希望正例 pair 和负例 pair 隔开至少 η 的距离，这一函数同样可以写成另外一种形式，让正例 pair 和负例 pair 采用不同的 s 函数，例如，$$s(x,x^{+})=|max(0,f(x)−f(x^{+})|,s(x,x^{-})=|max(η,f(x)−f(x^{+})|$$。
>- 如何构建正例和负例？针对不同类型数据，例如图像、文本和音频，如何合理的定义哪些样本应该被视作是 $$x^{+}$$，哪些该被视作是 $$x^{-}$$，；如何增加负例样本的数量，也就是上面式子里的 N？这个问题是目前很多 paper 关注的一个方向，因为虽然自监督的数据有很多，但是**设计出合理的正例和负例 pair，并且尽可能提升 pair 能够 cover 的 semantic relation，才能让得到的表示在 downstream task 表现的更好**

## Instance discrimination

>这个是定义pretask代理模式
>
>`重要的是构建正样本和负样本的规则`
>
>Video领域同一视频任意两帧都是正样本,其他都是负样本
>
>NLP中SimCSE中同个句子做两次foraward，但是使用不同dropout,得到两个特征就是正样本
>
>CMC中，一个物体的背面和正面都是正样本,RGB图像和深度图像是正样本

# MoCo

>Momentum Contrast for Unsupervised Visual Representation Learning(MoCo)
>
>何凯明团队的顶刊
>
>深度学习的本质就是做两件事情：Representation Learning（表示学习） 和 Inductive Bias Learning（归纳偏好学习）
>
>![](https://pic.imgdb.cn/item/67110315d29ded1a8c47915b.png)
>
>`对比学习学到的很好的特征：类似物体在这个特征空间 相邻，不类似的物体在特征空间 远离`
>
>**Q: 图 1 和 图 2 相似，和图 3 都不相似，难道不是有监督学习吗？Why 对比学习在 CV 领域被认为是无监督训练呢？**
>
>`CV 领域 设计巧妙的代理任务 pre-text task，人为设立一些规则 —— 定义哪些图片相似、哪些图片不相似，为自监督学习提供监督信号，从而自监督训练`

## Abstract

>`we build dynamic dictionary with a queue and a moving-averaged encoder`
>
>设计了一个动态字典，字典是用queue实现的
>
>the representations learned by MoCo transfer well to downstream tasks
>
>`MoCo在下游任务可以表现的很好`
>
>MoCo can outperform its supervised pre-training counterpart in 7 detection/segmentation tasks on PASCAL VOC, COCO, and other datasets
>
>`MoCo可以在检测和分割且在PASCAL VOC COCO和其他数据集中比其他有监督预训练模型取得更好的表现`
>
>This suggests that the gap between unsupervised and supervised representation learning has been largely closed in many vision tasks
>
>`MoCo将在有监督和无监督的视觉任务中表现学习的坑填上了！！！！`
>
>`MoCo （第一个）在 （分类、检测、分割）主流的 CV 任务证明 无监督学习 也不比 有监督学习 差`

>Q1：动态字典是什么?
>
>**dictionary look-up**, 字典查询任务, a dynamic dictionary with a queue and a moving-averaged encoder 动态字典
>
>- **一个队列**：队列中的样本**无需梯度回传**，可以放很多负样本，让字典变得很大
>- **一个移动平均的编码器**：让字典的特征尽可能的保持一致
>- 一个`大的、一致的字典`，有利于 无监督的对比学习 训练。

## Introduction

>Q2: 无监督学习为什么在CV领域不成功?
>
>**NLP 的离散单词更具语义性，CV的连续、高维信号不好构建字典**
>
>
>
>Q3:无监督在 CV 不成功的原因是什么？
>
>- 原始信号空间的不同
>- NLP 原始信号是离散的，词、词根、词缀，容易构建 tokenized dictionaries 做无监督学习
>  - tokenized: 把一个词对应成某一个特征
>  - Why tokenized dictionaries 有助于无监督学习？
>    - `把字典的 key 认为是一个类别，有类似标签的信息帮助学习`
>  - NLP 无监督学习很容易建模，建好的模型也好优化
>- `CV 原始信号是连续的、高维的，不像单词具有浓缩好的、简洁的语义信息，不适合构建一个字典`
>  - 如果没有字典，无监督学习很难建模
>
>
>
>Q4:MoCo前期的对比学习归纳了什么?
>
>![](https://pic.imgdb.cn/item/67110785d29ded1a8c4baeb8.png)
>
>$$x_{1}^{1}$$: anchor(锚点)
>
>$$x_{1}^{2}$$: positive(正样本)
>
>$$x^{2}$$, $$x^{1}$$......, $$x^{n}$$: negative  (负样本)
>
>`编码器 E_11 和 E_12 可以一样，可以不一样 `
>
>E_12：因为 positive 和 negative 都是相对 anchor f_11 来说的。
>
>`正负样本使用同样的编码器`正负样本肯定要是同一个编码器
>
>
>
>Q5:为什么对比学习归纳成在做一个动态字典呢?
>
>f11 当成 query 在 f12, f2, f3, ......, fn 组成的字典的 key 特征条目 k1, k2, ...... 里面查找，dictionary look-up 靠近 f12, 远离 f2, f3, ......
>
>- be similar to its matching key and dissimilar to others
>- learning is formulated as minimizing a contrastive loss 最小化对比学习的目标函数
>
>`f11是anchor(锚点),f12是正样本(positive),f2,f3,f4......,fn是负样本`
>
>
>
>Q6: 动态字典的角度看对比学习,什么样的字典才适合?
>
>- `large `
>  - 从连续高维空间做更多的采样。字典 key 越多，表示的视觉信息越丰富，匹配时更容易找到具有区分性的本质特征。
>  - 如果 字典小、key 少，模型可能学到 shortcut 捷径，不能泛化
>- `consistent `
>  - 字典里的 `key (k0, k1, k2, ......, kN) 应该由相同的 or 相似的编码器`生成
>  - 如果字典的 key 是由不同的编码器得到的，query q 做字典查询时，很有可能 找到和 query 使用同一个 or 相似编码器生成的 key，而不是语义相似的 key。另一种形式的 shortcut solution
>
>原文:
>
>`(i) large and (ii) consistent as they evolve during training.`
>
>`while the keys in the dictionary should be represented by the same or similar encoder so that their comparisons to the query are consistent`
>
>
>
>Q7: **为什么要提出 MoCo? **
>
>**给CV 无监督对比学习 构建一个 大 (by queue)+ 一致 (momentum encoder) 的字典**
>
>- queue 数据结构: 剥离 字典的大小 和 显卡内存的限制，让字典的大小 和 模型每次做前向传播的 batch size 的大小 分开
>  - `字典很大（成千上万），意味着要输入很多很多的图片，显卡内存吃不消`
>  - current mini-batch enqueued and the oldest mini-batch dequeued 当前 mini-batch 入队，最早进入队列的 mini-batch 出队
>  - 队列的大小 == 字典的大小，但是每次做 iteration 更新，并不需要更新字典中所有 key 元素的值。普通 GPU 训练
>
>- `momentum encoder`: 
>  - Q：使用 queue，只有当前 mini-batch 的特征是由当前的编码器得到的；之前的 key 是由不同时刻的编码器抽取的特征，如何保持 consistent 呢？
>  - `momentum encoder` 由 当前时刻的 encoder 初始化而来	
>    - $$\theta_{k}=m*\theta_{k-1} + (1-m)*\theta_{q}$$
>    - 动量参数 m 较大时，$$\theta_{k}$$ 的更新缓慢，不过多的依赖于 $$\theta_{q}$$ 当前时刻的编码器，`即不随着当前时刻的编码器快速改变，尽可能保证 字典里的 key 都是由相似的编码器生成的特征，保证特征的 consistent`
>
>`momentum公式是保持队列中key的编码连续性的关键`
>
>![](https://pic.imgdb.cn/item/6711164ed29ded1a8c57aa69.png)
>
>
>
>Q8: MoCo的代理任务是pretext task?
>
>**nstance discrimination**
>
>we follow a simple` instance discrimination `task [61, 63, 2]: a query matches a key if they are encoded views (e.g., different crops) of the same image.
>
>
>
>**instance discrimination**: query 和 key 匹配 如果它们来自于同一张图片的不同视角, i.e., 不同的裁剪
>
>MoCo 用 instance discrimination 无监督训练 在 ImageNet 上可以和之前最好的结果打个平手 or 更好的表现 competitive results
>
>
>
>Q9:MoCo的效果怎么样?
>
>**无监督学习的目的**：在一个很大的无标注的数据集上训练，模型学到的特征可以很好的迁移到下游任务。
>
>`MoCo 做到了。7个检测 or 分割的任务表现很不错。超越 ImageNet 有监督训练的结果，甚至有时大幅度超越 in some cases by nontrivial margins`.
>
>**无监督学习的期待：更多数据、更大的模型，性能会提升，不饱和。**
>
>`MoCo 在 10亿 Instagram 数据集（更糙 relatively curated 真实、一张图片有多个物体; ImageNet 数据集的图片大多只有一个图片、在图片中间） 上性能还有提升`
>
>
>
>Q10: MoCo在一系列的任务和数据集上效果很好 positive results ，体现在哪?
>
>- 1000 倍数据集数量的增加， MoCo 性能的提升不高
>  - 大规模数据集可能没有完全被利用
>  - 尝试开发其它的代理任务 pretext task
>- 除了 instance discrimination 代理任务，类似 NLP 的代理任务 masked auto-encoding
>  - MAE, 大佬 2 年前就有了想法，做了实验；做研究急不来
>  - 像 NLP 的 BERT 使用 masked language model 完形填空做自监督预训练

## Related Work

>两个可以做的点：pretext tasks and loss functions
>
>- **代理任务**：不是大家实际感兴趣的任务 (检测、分类、分割实际应用任务)，而是为了 学习一个好的数据特征表示
>- **损失函数**：和代理任务可以分开研究。 MoCo 的创新点在损失函数，又大又一致的字典 影响 info NCE 目标函数的计算



>Q11:那正常的对固定目标的loss怎么做?
>
>Common损失目标函数(Loss functions)：衡量 模型的预测输出 和 固定的目标之间的 difference。
>
>- L1 or L2 losses 
>  - i.e., Auto-encoder（生成式网络的做法）, 输入一张原图 or 一张被干扰的图，经过编码器、解码器 重构输入的图，衡量是原图 和 重构图 之间的差异。
>
>`such as reconstructing the input pixels(e.g., auto-encoders) by L1 or L2 losses, or classifying the input into pre-defined categories (e.g., eight positions [13],color bins [64]) by cross-entropy or margin-based losses.`
>
>
>
>Q12: 损失函数在判别式、生成式、对比学习、对抗学习中是什么样的?
>
>- 对比学习的损失：目标不固定，训练过程中不断改变。目标有编码器抽出来的特征（MoCo 的字典）而决定
>- 判别式：预测 8 个位置中的哪一个方位
>- 生成式：重建整张图
>- 对比学习的目标：测量 样本对 在特征空间的相似性。
>  - 相似样本离得近，不相似样本离得远
>- 最近无监督表现好的文章都用了 contrastive learning (Sec. 3.1 讨论)
>
>`Contrastive losses [29] measure the similarities of sample pairs in a representation space`
>
> ` contrastive loss formulations the target can vary on-the-fly during training and can be defined in terms of the data representation computed by a network`
>
>- 对抗学习的损失：衡量两个概率分布之间的差异，i.e., GAN
>  - unsupervised data generation 做无监督的数据生成
>  - 对抗性的方法做特征学习
>    - 如果可以生成很好、很真实的图片，模型应该学到数据的底层分布
>  - GAN 和 NCE 的关系 noise-contrastive estimation Ref. [24]
>
>
>
>Adversarial losses [24] measure the difference between `probability distributions`.
>
>Adversarial methods for representation learning are explored in [15, 16]. There are relations (see [24]) between generative adversarial networks and `noise-contrastive estimation (NCE) [28]`.
>
>
>
>Q13: 监督学习和无监督学习的区别是什么?
>
>监督学习: 存在ground truth也就是lable,Loss就是衡量输出Y和标签ground truth的差异
>
>无监督学习:从目标函数L和pretext task中生成ground truth(伪标签)
>
>`MoCo是无监督学习，目标就是解决Loss函数和生成伪标签`
>
>`Loss就采取InfoNCE ,`
>
>` 生成伪标签就是构建动态字典的过程,就是选择正样本，然后其他都为负样本,锚点和正样本、负样本进行LOSS过程`
>
>
>
>Q14: 对比学习和代理任务之间是什么关系?
>
>The instance discrimination method [61] is related to the exemplar-based task [17] and NCE [28].
>
>The pretext task in contrastive predictive coding (CPC) [46] is a form of context auto-encoding [48]
>
>contrastive multiview coding (CMC) [56] it is related to colorization [64].

## Method

>Q15: Contrastive learning的Loss是什么? InfoNCE
>$$
>\mathcal{L}_{q}=-\log \frac{\exp \left(q \cdot k_{+} / \tau\right)}{\sum_{i=0}^{K} \exp \left(q \cdot k_{i} / \tau\right)}
>$$
>whose value is low when q is similar to its positive key k+ and dissimilar to all other keys (considered negative keys for q)
>
>where` τ `is a `temperature hyper-parameter per` [61]. The sum is over one positive and K negative samples. Intuitively, this loss is the log loss of a (K+1)-way softmax-based classifier that tries to classify q as k+
>
>`K是代表K+1 个样本, InfoNCE就是将多分类问题变成二分类问题`,二分类----正样本和负样本
>
>- 是query(anchor)与正样本的乘积除于$$\tau$$-------代表q和正样本的联系，$$ \tau$$代表温度系数,用来调控
>  - 如果$$ \tau $$越小,那么Loss对于负样本都是一视同仁的,导致模型没有轻重
>  - 如果$$ \tau $$是越大,那么loss只关注特别困难的负样本,导致学好特征无法泛化
>
>![](https://pic.imgdb.cn/item/671127f7d29ded1a8c670ce2.png)
>
>
>
>`注意联系softmax的关系,观察N是类别`
>$$
>y_{i}=\operatorname{softmax}\left(x_{i}\right)=\frac{e^{x_{i}}}{\sum_{k=1}^{N} e^{x_{k}}}
>$$
>`Cross entropy loss：`
>
>![](https://pic.imgdb.cn/item/6711249bd29ded1a8c640d14.png)
>
>
>
>Q16：在Momentum contrast之前，对比学习其他方法有什么局限性?
>
>![](https://pic.imgdb.cn/item/67112b48d29ded1a8c69b95d.png)
>
>- end-to-end结构: 因为key参数是连续在同一个encoder中进行编码,mini-batch size过大，因为mini-batch size的大小和字典的大小成正比,那么就会存在显存无法存放的问题,例如SimCLR就是end-to-end学习,存在`字典大小太大的局限large`
>- memory bank结构: memory bank可以存下Imagenet的特征值，Imagenet的特征值有128w的key,仅需要600M的内存就能存进,但是存在的问题是每次k更新不是连续的编码器，也就是在快速迭代中,编码器是会发生变化,每次k的值是会变化的，存在`编码器不连续的问题`unconsistent
>
>![](https://pic.imgdb.cn/item/671131a3d29ded1a8c6f1441.png)
>
>- 基于上面两个问题，提出`动量更新和`queue`的方案
>
>![](https://pic.imgdb.cn/item/67113260d29ded1a8c6fab5c.png)
>
>m是动量参数,设置在0.99就可以保持$$\theta_{k}$$的`更新连续性`，可以高度和之前的$$\theta_{k}$$保持高度一致,但也可以收到$$\theta_{q}$$的影响,
>
>`queue`可以更新当前的mini-batch ，出队旧的mini-batch,所以将queue和mini-batch隔离开,达到可以调大queue的大小，不至于mini-batch放不进的现存的问题
>
>
>
>Q16:那么为什么$$\theta_{k}$$不直接复制之前的特征值呢?
>
>`因为我们通过编码器处理得到的k代表正样本和负样本的特征值,不通过反向传播的时候,是需要进行更新的,因为我们需要loss不断的收敛,是需要k参数进行更新的,让锚点样本、正样本更加靠近,和负样本更加远离,如果k不更新的话,会导致loss不会收敛`
>
>