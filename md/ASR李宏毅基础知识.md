# Acoustic Feature处理大致流程

## 采样流程

![](https://pic.superbed.cc/item/670a0b5cfa9f77b4dca0cf43.png)

**理论知识参考台大李琳山老师的数位语音处理**

![](https://pic.superbed.cc/item/670a0e6afa9f77b4dca0eac9.png)

# LAS(Listen,Attend,and Spell)

>参考论文:
>
>[Chorowski. et al., NIPS’15] Jan Chorowski, Dzmitry Bahdanau, Dmitriy 
>Serdyuk, Kyunghyun Cho, Yoshua Bengio, Attention-Based Models for Speech 
>Recognition, NIPS, 15

## 1. Listen过程

>Input:  acoustic features(声音向量)
>
>output: 一串向量
>
>作用: 提取预言内容信息,将音频信号分析为高级的声学特征表示

### Listen 过程种类:

- RNN(Recurrent Neural Network)

![](https://pic.superbed.cc/item/670a19b7fa9f77b4dca1a94d.png)

- 1-D CNN

![](https://pic.superbed.cc/item/670a1a03fa9f77b4dca1b342.png)

- Self-attention Layers机制

>[Zeyer, et al., ASRU’19] Albert Zeyer, Parnia Bahar, Kazuki Irie, Ralf Schlüter, 
>Hermann Ney, A Comparison of Transformer and LSTM Encoder Decoder Models 
>for ASR, ASRU, 2019

>Shigeki Karita, Nanxin Chen, Tomoki Hayashi, Takaaki Hori, Hirofumi Inaguma, Ziyan Jiang, Masao Someki, Nelson Enrique Yalta  Soplin, Ryuichi Yamamoto, Xiaofei Wang, Shinji Watanabe, Takenori Yoshimura, Wangyou Zhang, A Comparative Study on Transformer vs RNN in  Speech Applications, ASRU, 2019



![](https://pic.superbed.cc/item/670a1b7bfa9f77b4dca1d4a2.png)

- Listen-Down Sampling过程

>Dzmitry Bahdanau, Jan Chorowski, Dmitriy  Serdyuk, Philemon Brakel, Yoshua Bengio, End-to-End Attention-based Large  Vocabulary Speech Recognition, ICASSP, 2016

![](https://pic.superbed.cc/item/670a1dc4fa9f77b4dca208d9.png)

>Vijayaditya Peddinti, Daniel Povey, Sanjeev  Khudanpur, A time delay neural network architecture for efficient modeling of  long temporal contexts, INTERSPEECH, 2015 

![](https://pic.superbed.cc/item/670a21b5fa9f77b4dca24ed0.png)

## 2.Attention过程

>主要作用是通过注意力机制建立声学特征和文本之间的关联，帮助模型更好地将声学特征转化为文本输出。其可以动态地分配注意力权重，以捕获输入序列中不同位置的重要信息的关键组件。

`上一层通过listen过程，将acoustic feature向量转化为特征信号了，接下来的工作就是要将特征信号h通过Attention层与文本建立联系`

![](https://pic.superbed.cc/item/66d811defcada11d3757c7c8.png)

`参考这个流程,声学特征与文本也进行QKV操作`

![](https://pic.superbed.cc/item/670a6d71fa9f77b4dca5b60a.png)

>![](https://pic.superbed.cc/item/670a6e03fa9f77b4dca5bb3d.png)
>
>综合以上三个图分析，输入的是`h向量`
>
>首先需要跟Z0关键字做`QK匹配`,也就是Query就是Z0，Key就是h向量
>
>`为什么需要Z0呢?这个问题值得探讨`
>
>因为传统的encoder-decoder框架中,seq2seq模型中,是需要进行target input words的输入的

- 步骤

>1. 需要通过match计算QK之间的相似度分数,确定value对当前时间步query最重要,然后通过softmax操作，归一化成概率
>
>- Key（键）：为Input之一，指的是图中的【h1~h4】。Key是编码器（通常是RNN或CNN）生成的中间表示的一部分。在LAS中，编码器生成的声学特征表示被认为是Key。每个时间步都有对应的Key，因此它是一个时间步的特征向量。
>
>- Query（查询）：为Input之一，指的是图中的【z0】。Query是解码器中的隐藏状态，通常用于生成文本输出。在LAS模型中，Query是解码器中的隐藏状态向量，它表示当前时间步的解码器状态。
>
>- Value（值）：在这里与Key是一个东西，也是指的是图中的【h1~h4】。Value是与Key对应的权重信息，通常也是编码器生成的声学特征表示。每个时间步都有一个对应的Value。Value包含了有关输入序列的有用信息。
>
>`Z0就是隐藏层的`

>2. 相似度分数经过softmax操作
>3. 注意力权重用于Value进行加权求和，生成加权的声学特征表示,这些特征将用于生成文本输出,这个加权求和过程允许模型自适应关注输入序列中与当前解码步骤相关的信息，提高语音识别或文本生成的性能

![](https://pic.superbed.cc/item/670a7000fa9f77b4dca5cc15.png)



>这个C0就是Contex Vector
>
>Context Vector（上下文向量）：就是Attend环节的 Output，指的是图中的【c0】。其随后与解码器的隐藏状态相结合，通常是通过简单的拼接或加法操作，以生成一个包含上下文信息的声学特征表示。这个合并后的特征表示将被解码器用于生成下一个文本标记，从而实现语音识别或文本生成任务（在后续的 Decoder 中有介绍。
>

- Match Function

1. Dot-product Attention

![](https://pic.superbed.cc/item/670a708afa9f77b4dca5d0c8.png)

2. Additive Attention

![](https://pic.superbed.cc/item/670a70a1fa9f77b4dca5d1a5.png)

>z，h进行生成矩阵后，然后相加，组后经过tanh, 再通过一个linear transform

## 3.Spell 过程

>1.上下文融合:
>
>​	1.Input:
>
>- 前一个时间步的隐藏状态(zi,初始位z0)
>- 前一个时间步生成的标记(token,初始位起始符号<s>)
>- 当前时间步"Attend"环节的输出(上下文向量Contex Vector)
>
>2.Method 通常使用`前馈神经网络`,如RNN
>
>3.Output
>
>- 1.新的隐藏状态z1
>- vocabulary各个token的概率值



>2.从distribution取出概率值最大的token，作文本次token输出，比如cat
>
>![](https://pic.superbed.cc/item/670a7587fa9f77b4dca5fb48.png)

>3.然后将刚刚得到的新的hidden state z1，去做attention，得到c1，然后再进行刚刚的操作，即将得到的 z1 投入新一轮的 attention 中，计算得到c1，再去计算下一轮。
>
>![](https://pic.superbed.cc/item/670a76f4fa9f77b4dca60765.png)

>**事实*上，这里也分了2种情况，一种是这样的，一种是这里的c1还会影响自己那一步的输出：*
>
>![](https://pic.superbed.cc/item/670a83e3fa9f77b4dca674f1.png)
>
>*那么我们的第一篇利用seq2seq模型做语音辨识的论文是采用了什么方式呢？我全都要！*
>
>![](https://pic.superbed.cc/item/670a8447fa9f77b4dca67897.png)



# HMM过程

## 1.介绍

>在人工智能（AI）和机器学习领域，**HMM** 代表 **Hidden Markov Model（隐马尔可夫模型）**，这是一种统计模型，常用于处理**时间序列**数据，尤其是那些具有潜在状态的序列，比如语音、文本、手势等。HMM 在语音识别、自然语言处理和生物信息学等领域有广泛的应用。
>
>HMM 是基于马尔可夫过程的一个扩展，其中存在**隐藏的（不可见的）状态序列**，但我们只能观察到与这些状态相关的**输出（观测）序列**。关键概念包括：
>
>1. **隐藏状态（Hidden States）**：
>   - 序列中的每一步都有一个隐藏状态。虽然这些状态是无法直接观测到的，但它们决定了输出的结果。比如在语音识别中，隐藏状态可能代表语音信号对应的音素。
>2. **观测（Observations）**：
>   - 每个隐藏状态产生一个可观测的输出。这些观测值是我们可以获取的数据，但我们并不知道它们对应的具体隐藏状态。
>3. **状态转移概率（Transition Probabilities）**：
>   - 这是从一个隐藏状态转移到下一个隐藏状态的概率。HMM 假设状态的转移具有马尔可夫性质，即当前状态仅依赖于前一个状态，而与更早的状态无关（“无记忆”性质）。
>4. **观测概率（Emission Probabilities）**：
>   - 每个隐藏状态生成某个观测值的概率。这表示给定一个隐藏状态，输出某个观测值的可能性。
>5. **初始状态概率（Initial State Probabilities）**：
>   - 系统最初处于某个状态的概率分布。它表示序列开始时各个隐藏状态的可能性。
>
>### **HMM 的工作流程**：
>
>在 HMM 中，观察到的序列（如音频信号、文本单词等）被认为是通过隐藏的马尔可夫链生成的，具体的过程如下：
>
>1. 模型从初始状态开始，根据初始状态概率选择一个隐藏状态。
>2. 根据状态转移概率从一个隐藏状态转移到下一个隐藏状态。
>3. 在每个隐藏状态中，依据观测概率生成一个观测值。
>4. 重复上述步骤，生成整个观测序列。

![](https://pic.superbed.cc/item/670bca43fa9f77b4dcb21fc3.png)

- 在过去，我们可以使用统计模型来做语音识别。给定输入语音序列 **X**，我们只需要找到最大概率的输出文字 **Y** 就可以了，也就是穷举所有可能的 **Y**，找到一个 **Y\*** 使得 P(**Y**|**X**) 最大化。我们也把这个过程叫作解码（decode），公式如下：

$$
Y^{*}=\arg \max _{Y} P(Y \mid X)
$$

- 穷举需要非常好的演算法，这个问题太复杂。好在我们可以使用贝叶斯定理对其进行变换，变换后的公式如下。由于 P(**X**) 与我们的解码任务是无关的，因为不会随着 **Y** 变化而变化。所以我们只需要保留分子部分即可。

$$
\begin{aligned}
Y^{*} & =\arg \max _{Y} P(Y \mid X) \\
& =\arg \max _{Y} \frac{P(X \mid Y) P(Y)}{P(X)} \\
& =\arg \max _{Y} P(X \mid Y) P(Y)
\end{aligned}
$$

>概率论中贝叶斯定理
>
>![](https://pic.superbed.cc/item/670bcb0dfa9f77b4dcb228b6.png)

- 变换后，我们将式子的前半部分 P(X|Y) 称为 Acoustic Model，后面这项 P(Y) 称为 Language Model。而前者所经常使用的就是 HMM。我们看到，如果需要使用 HMM，就必须搭配 LM 来进行使用。而常规的 E2E 模型是直接解未变行的式子的，表面上看上去好像不需要 LM，实际上 E2E 模型加上 LM 后表现往往会好很多，这个可以参考之后对 LM 的讲解。

## 2.States(隐藏状态)

>在前面我们说过，语音识别模型中，目标 Y 是 Token 序列，然而，`我们在 HMM 中会将目标 Token 序列转为 States 序列，用 S 来表示。State 是什么？它是由人定义的，比音素 Phoneme 还要小的单位`。

![](https://pic.superbed.cc/item/670bd113fa9f77b4dcb27f03.png)

>我们使用 what do you think 句子来举例，
>
>使用 phoneme 作为 token 单位的话，分解结果如下。不过，由于每个因素都会受到前后因素的影响，
>
>所以相同的因素 uw 说不定实际上发音会不同。所以我们会更加细分，
>
>`采用 Tri-phone 来作为 token 单位，即当前音素加上前面的和后面的音素`。

>而` State 就是比 Tri-phone 更小的单位，我们可以规定每个 Tri-phone 由 3 或者 5 个 state 构成`。多少就取决于你所拥有的计算资源。而拆解出来的 State 也保留了发音顺序信息。

![](https://pic.superbed.cc/item/670bd4abfa9f77b4dcb2ae24.png)

>既然我们需要计算给定 States 时，
>
>声学特征序列 X 的几率，那我们就需要弄清楚 State 是怎么产生出声学特征的。
>
>其实很简单，
>
>`假设我们一共有3个 State，而 X 有 6 个 Vector，那么我们首先进入第一个 State，产生一些向量，足够了以后进入下一个 State，以此类推，依次走完就结束了`。

![](https://pic.superbed.cc/item/670bd624fa9f77b4dcb2bf6c.png)

## 3.转移概率和发射概率

- 转移概率

>transition Probability：本次的 vector 是由状态 a 产生的，下一个 vector 是由状态 b 产生的概率。

![](https://pic.superbed.cc/item/670bd785fa9f77b4dcb2d3bb.png)

- 发射概率

>Emission Probability：给定一个 State，产生某种 acoustic feature 的概率。我们认为，每一个状态所能发射的声学特征都有固定的概率分布，我们会用 GMM（Gaussian Mixture Model，高斯混合模型）来表示这个概率。`也就是P(X|S)`

![](https://pic.superbed.cc/item/670bd7bdfa9f77b4dcb2d6f1.png)

![](https://pic.superbed.cc/item/670bd97bfa9f77b4dcb2ef21.png)

>`这样的技术发展到现在已经出现了最终形态：Subspace GMM。这其中，所有的State都共用同一个高斯混合模型。它实际上是一个高斯混合分布池（pool），里面有很多高斯混合分布。每一个State，就好比是一个网子，它去这个池子中捞几个高斯分布出来，当作自己要发射的高斯混合分布。所以每个State既有不同的高斯分布，又有相同的高斯分布。`

- 高斯混合模型(GMM)

>**高斯混合模型（Gaussian Mixture Model, GMM）** 是一种常用于聚类和概率密度估计的统计模型。它假设数据由多个高斯分布（正态分布）混合而成，每个高斯分布代表数据中的一个潜在子类。GMM 是一种**无监督学习方法**，经常用于从复杂的、多峰的数据中提取模式和结构。
>
>### **基本概念**：
>
>GMM 假设数据来自多个高斯分布的组合，即数据点 xxx 的生成可以表示为：
>$$
>p(x)=\sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(x \mid \mu_{k}, \Sigma_{k}\right)
>$$
>其中：
>
>- K是高斯分布的数量（即模型中的混合成分数）。
>- πk 是第 kkk 个高斯分布的混合系数（权重），满足
>
>$$
>\sum_{k=1}^{K} \pi_{k}=1
>$$
>
>
>$$
>\mathcal{N}\left(x \mid \mu_{k}, \Sigma_{k}\right)
>$$
>
>
>-  是第 kkk 个高斯分布的概率密度函数，参数为均值 μk 和协方差矩阵 Σk。
>
>### **高斯混合模型的组成部分**：
>
>1. **混合系数（Mixing Coefficients） πk\pi_kπk**：
>   - 每个高斯分布对应一个混合系数 πk\pi_kπk，表示该分布在整体模型中所占的比例。所有混合系数之和为 1，表示不同分布的加权组合。
>2. **高斯分布（Gaussian Components）**：
>   - 每个组件是一个高斯分布，其由均值向量 μk\mu_kμk 和协方差矩阵 Σk\Sigma_kΣk 描述。均值 μk\mu_kμk 描述该分布的中心，协方差矩阵 Σk\Sigma_kΣk 描述该分布的形状和方向。
>3. **隐变量（Latent Variables）**：
>   - GMM 假设每个观测数据点是由多个隐藏的、高斯分布的生成过程产生的。每个数据点都被假设属于某一个高斯分布，具体是哪个分布则是隐变量决定的。
>
>### **GMM 的特点**：
>
>- **软聚类（Soft Clustering）**：
>  - 与传统的 K-Means 聚类不同，GMM 允许数据点以一定的概率属于多个不同的类别。每个点可以同时隶属于多个高斯分布，概率之和为 1。这种柔性分类称为软聚类。
>- **捕捉复杂分布**：
>  - GMM 能够表示复杂的概率密度分布，特别适合处理具有多峰的、非对称的和不规则形状的数据分布。
>- **协方差结构**：
>  - 每个高斯成分可以有不同的协方差矩阵（对角矩阵或全矩阵），这允许 GMM 捕捉不同类别之间的变化程度和形状。

## 4.Alignment

>- 假设我们已经知道了 Transition Probability 和 Emission Probability，然而我们还是计算不出来我们的目标概率 P(X|S)，因为我们还缺少 Alignment。这是什么意思？
>- `就是我们还是不知道这些 vector 是对应到哪一个状态的。也就是说我们需要知道哪一个声学特征，是由哪一个状态产生的，才有办法用发射概率和转移概率去计算 P(X|S)。`

![](https://pic.superbed.cc/item/670bde2ffa9f77b4dcb327e2.png)

>假设我们有3个状态 abc，
>
>6个向量 x1~6，我们需要得到状态于向量的对齐方式 h（即状态序列）
>
>比如 aabbcc，也就是 x1 x2 由状态 a 产生，以此类推。知道了对齐方式，我们就可以用两个概率计算目标概率了。现实中，也正因为我们不知道 Alignment，这个信息是隐藏的，所以 HMM 中的 Hidden 命名就此诞生。不同的状态序列，计算出的概率也就会不一样。

![](https://pic.superbed.cc/item/670bde4ffa9f77b4dcb3296b.png)

>图中h就是hidden
>
>p(x|a)就是当a状态发生时，x发生的概率
>
>p(b|a)就是当a状态发生时，b发生的概率

![](https://pic.superbed.cc/item/670be02ffa9f77b4dcb33f4d.png)

>那么我们是如何解决隐藏的 Alignment 信息问题的呢？我们选择`穷举所有可能，把所有的状态序列的概率全都计算出来并加起来`，`最终的结果就是我们的目标概率 P(X|S)`。这便是 HMM 在解码过程中在做的事情。
>
>当然，诸如 abccbc、abbbbb 这样的序列都是不算在内的，其原因是回跳和少状态。

# Deep Learning下的HMM

## 1. Tandem

![](https://pic.superbed.cc/item/670be1a1fa9f77b4dcb34db1.png)

>- `而 Tandem 则是训练一个基于深度神经网络的 State Classifier，它可以输入一个MFCC vector，来预测它属于哪一个状态的概率，输出就是其概率分布。我们将这个概率分布代替之前的声学特征，来作为 HMM 的新的输入`。

## 2. DNN-HMM Hybrid

>Discriminative training 和 Generative Training 是机器学习中两种不同的训练方法，通常用于分类和生成模型。
>
>Discriminative Training：
>
>定义：这种训练方法旨在学习数据的条件分布或决策边界，以便区分不同类别之间的差异。它主要关注于对输入数据进行标签分类的任务。这种方法专注于学习直接给出类别标签的条件概率分布，例如，在监督学习中学习从输入到标签的映射。
>
>示例：常见的例子包括支持向量机（SVM）、逻辑回归和神经网络等。
>
>Generative Training：
>
>定义：这种训练方法专注于建模数据的生成分布，试图理解数据的产生方式。它不仅仅关注于分类任务，还试图模拟数据生成的过程。通过学习数据的分布模型，可以生成与原始数据相似的新数据。
>
>示例：典型的例子是生成对抗网络（GANs）、变分自编码器（VAEs）和隐马尔可夫模型（HMM）等。
>
>这两种方法在目标和应用上有所不同。Discriminative training 更多关注于数据分类问题，寻找边界或条件概率，使得能够对输入数据进行准确分类。而 Generative Training 则关注于学习数据的生成过程，以便能够生成与原始数据相似的新样本，同时也可以应用于分类任务。
>

![](https://pic.superbed.cc/item/670cb0b6fa9f77b4dcbbbbc6.png)

- 原来的 HMM 中有个高斯混合模型，我们就想使用 DNN 来取代它。然而，高斯混合模型是给定一个 State，输出各声学特征的概率分布，也就是 P(x|a)；`刚刚讲的 State Classifier 却是给定一个声学特征向量，输出其属于各个状态的概率分布，也就是 P(a|x)。这二者似乎是相反的存在`。


>对于这个问题就是这两个似乎是对立的存在, 我们根据贝叶斯定理可以进行变换

![](https://pic.superbed.cc/item/670cb15afa9f77b4dcbbc408.png)

>那么，为什么用 DNN 去计算 P(x|a) 要比高斯混合模型计算来的好呢？有的人认为，DNN 的训练过程是 Discriminative training，而原来的 HMM 则是 Generative Training，前者要更好。然而，事实上，虽然 HMM 是生成模型，但是它也可以使用 Discriminative training，并且也有很多人在 DNN 前做过相关研究了。也有人觉得他厉害之处在于 DNN 拥用有更多参数。但这小看了参数量也大起来时，GMM的表征能力，最终实际上 DNN 用的参数和 GMM-based HMM 使用的参数是差不多的。
>
>实际上，这篇论文的贡献在，它让所有的给定观测计算当前可能状态概率，都共用了一个模型。而不是像GMM那样，每一个 State 都需要有它们自己的 GMM，有着不同的 mean 和 variance。所以它是一个非常厉害的以数据作为驱动的状态标注的方法。
>
>那么，DNN 的效果如何呢？事实证明它非常强大。要知道，DNN 可以不是全连接层组成的那种网络，而是可以是任何类型的神经网络，比如 CNN，LSTM 等等。
>
>

## 3. DNN中State Classifier的训练方式

>- 那么我们如何去训练 State Classifier 呢？它的输入是一个声学特征，输出是它是某个状态的概率。我们训练这个之前，需要知道每个声学特征和状态之间的对应关系。但实际中的标注数据都是没对齐的，只有声学特征和对应的文本。

>过去的做法是先训练一个 HMM-GMM，有了以后你就可以算出概率最大的 Alignment，有了对齐方式就可以去训练 State Classifier 了。

![](https://pic.superbed.cc/item/670cb25cfa9f77b4dcbbd0c0.png)

>不过这样也会有人担心，HMM-GMM 不是说表现不好吗，用它的结果来训练 DNN 是不是不太好？
>
>那么我们也可以用刚刚训练好的第一代 DNN 再替换 HMM-GMM，给出新的对齐序列，再用它来对 DNN 进行迭代，这样可以一直循环训练下去，一直到你满意为止。

![](https://pic.superbed.cc/item/670cb2bdfa9f77b4dcbbd545.png)

![](https://pic.superbed.cc/item/670cb2fcfa9f77b4dcbbd877.png)

# Alignment of HMM，CTC and RNN-T，对齐方式详解

## 1. E2E(End to End)模型和CTC、RNN-T区别

>实际上，对于端对端模型来说，比如 LAS，`它在解码的时候都是去寻找一个 token 序列，使得 P of Token Sequence **Y** given Acoustic features vectors **X** 最大`。

$$
\begin{align}
Decoding:  \mathrm{Y}^{*} & = \arg \max _{\mathrm{Y}} \log P(\mathrm{Y} \mid X) \\

Training:  \quad \theta^{*} & = \arg \max _{\theta} \log \mathrm{P}_{\theta}(\hat{Y} \mid X) 
\end{align}
$$

>为什么这么说？我们来简单看一下 LAS 的结构，每一次我们都是输出一个概率分布，我们就可以将这个概率分布作为输出 token 的概率，因此将最后所有 token 的概率相乘，结果就是 P(Y|X)。
>
>当然，在解上面那个式子的时候，我们也并不是直接找出每一个概率分布中最大的 token，而是采用束搜索等策略去找最优解。而在训练过程中，我们也可以将训练目标带入上面的式子。假设 Y^hat 就是最终正确的结果，那么训练目标就是找一个最优的模型参数，来让P(Y^hat|X)越大越好

![](https://pic.superbed.cc/item/670cb51cfa9f77b4dcbbf648.png)

## 待更新，可参考别的博客

# language modeling，语言模型详解

## LM的必要性

>Language Model 是什么？是估计一段 token 序列的概率的方法
>
>- Estimate the probability of token sequence
>
>- 这就是因为，LAS 是对条件概率建模，需要成对的数据，收集比较困难；而 LM 用到的只有文本的数据，非常容易收集，因而我们很容易得到 P(Y) 的分布。
>- `事实上，只要你的模型最后输出是文本， 那么加上 LM 总会有用的。`
>- **BERT** 大型 LM 模型，只有文本，大约有30亿以上的词

![](https://pic.superbed.cc/item/670cd1c8fa9f77b4dcbe0cfa.png)



## 2. 如何估计token sequence几率

- N-gram language model 计算 token sequence 几率方法（以 2-gram 为样例）：

- 例如

![](https://pic.superbed.cc/item/670cd3e7fa9f77b4dcbe2d19.png)

- N-gram出现的问题在于:当较大的n时，由于数据的稀疏性,会导致条件项从未在语料库中出现,导致最终计算的概率为0
  - 解决方法:采用language model smoothing，让原本概率为0的情况不为0

![](https://pic.superbed.cc/item/670cd45efa9f77b4dcbe312e.png)

- matrix factorization

**Matrix Factorization（矩阵分解）** 是一种数学技术，用于将一个矩阵分解为多个较小的矩阵的乘积。它在机器学习和数据分析中有着广泛的应用，尤其是在**推荐系统**、**降维**和**协同过滤**领域。

- 基本概念

矩阵分解的基本思想是：对于一个给定的矩阵 M，希望找到两个或多个较小的矩阵 U 和 V，使得它们的乘积可以近似表示 M。常见的目标是通过这种分解找到数据的**潜在模式**或**特征**，从而简化问题或进行预测。

`目标是找到 U 和 V 使得 M≈U×V，即尽可能减少误差`。

常用的矩阵分解

>1. SVD(Singular Value Decomposition，奇异值分解)
>2. PCA(Principal Component Analysis，主成分分析)
>3. NMF（Non-negative Matrix Factorization，非负矩阵分解）
>4. ALS（Alternating Least Squares，交替最小二乘法）

- 应用场景

>**推荐系统**：
>
>- 在推荐系统中，矩阵分解特别适合用于**协同过滤**。例如，在用户-物品评分矩阵中，许多用户没有对所有物品进行评分，这导致了一个**稀疏矩阵**。通过矩阵分解，可以预测这些缺失的评分，进而推荐用户可能喜欢的物品。
>- **Netflix Prize** 比赛中，矩阵分解方法被证明是解决推荐问题的有效方法之一。
>
>**降维**：
>
>- 在处理高维数据时，矩阵分解能够提取数据中的最重要特征，减少数据维度。例如，PCA 和 SVD 都是降维常用的技术。通过将数据投影到一个低维的子空间，可以减少计算复杂度并去除噪声。
>
>**自然语言处理**：
>
>- 在 NLP 中，矩阵分解可以用于**词嵌入**和语义分析。例如，通过对词-文档共现矩阵进行分解，可以将词汇映射到低维的语义空间，进而捕捉词之间的语义相似性。
>
>**图像处理**：
>
>- 矩阵分解也被广泛应用于图像处理任务中，比如图像压缩、降噪和特征提取。通过对图像矩阵进行分解，可以提取图像中的重要特征并减少冗余信息。

- 优点与挑战

>**优点**：
>
>1. **数据压缩**：矩阵分解可以有效减少数据的维度，提取出数据的核心信息。
>2. **可解释性强**：尤其是 NMF，它能够生成更具有现实意义的解释，例如文本中的主题或推荐系统中的用户偏好。
>3. **广泛应用**：从推荐系统到降维，矩阵分解适合各种需要特征提取、模式识别的任务。
>
>**挑战**：
>
>1. **处理稀疏数据**：在一些应用场景中，如推荐系统中，评分矩阵通常是高度稀疏的，如何有效地处理稀疏数据是一个挑战。
>2. **局部最优问题**：一些矩阵分解方法，如 NMF，容易陷入局部最优，需要合理的初始化和优化方法。

## 3.计算过程

关键是其采用的 language model smoothing 技术。其来自于推荐算法中的 Matrix Factorization（矩阵分解）。示意图如下：

![](https://pic.superbed.cc/item/670cd7bdfa9f77b4dcbe616b.png)

- 表格中，dog，cat 是前置词汇，表格中的值表示前置词汇后接 ran，jumped等词汇的次数。（二者相交处的格子填共现的频次）
- 我们的目的是尝试使用hi、vi向量来表示各个词汇，也就是找出一个function，输入单词，输出其向量。
- 因此我们采用梯度下降法来找到这些词汇所对应的向量，采用L2损失，Loss函数如图，我们想保证hi·vi的数值接近目前已观察到的次数。

![](https://pic.superbed.cc/item/670cd820fa9f77b4dcbe6a0e.png)

- `然后我们就用向量的乘积来替代原来观察的次数`。通过这种方式，它就会自动把0补成学到的参数。由于 dog 和 cat 与其它很多词有类似的共现关系。即便没有见过 dog jumped，但知道 cat jumped，它也会自动学出 dog 和 jumped 的关系。

![](https://pic.superbed.cc/item/670cd8ebfa9f77b4dcbe770f.png)

- 而进一步的，我们也可以将这个过程想象成只有一个隐藏层的神经网络，输入的就是 one-hot vector（独热向量），dog对应维度为1，隐藏层就是词汇的向量表达，输出就是后接词汇的类distribution。由此我们就可以引出 **NN-based LM**

## 4.NN-based LM

其做的事情和上面我们引申的 Continuous LM 相同，就是训练一个 NN。

![](https://pic.superbed.cc/item/670cda2ffa9f77b4dcbe8a40.png)

训练完成后，我们就可以使用这个LM，对之前拆分的概率公式中的各项进行计算。

![](https://pic.superbed.cc/item/670cda64fa9f77b4dcbe8d3d.png)

事实上，NN-based LM 其实出现得比 Continuous LM 还要早。在Continuous LM变得流行之后，才有人把 NN-based LM 找出来。之后随着深度学习崛起，才变成了现在的主流。在2003年，超过15年前发表的文章，Bengio 就有提到过 Word embedding 的概念。它有把中间的参数层可视化出来，就像我们现在看到的词嵌入类比实验可视化一样。
![](https://pic.superbed.cc/item/670cdad5fa9f77b4dcbe924f.png)

## 5. RNN-based LM

>有的时候我们需要选取前面词汇更多的情况，也就是5-gram，6-gram甚至更多，这会导致NN的输入序列过长。
>
>于是便有了 **RNN-based LM** 。这样有多长的gram我们就可以做多少。

![](https://pic.superbed.cc/item/670cdc82fa9f77b4dcbeaa9e.png)

>RNN 有各式各样的变形。曾经人们的想法是，把 RNN 尽可能地做复杂，看能不能做出更强的语言模型。甚至还有人用 Nerual Turning Machine 改一改来做语言模型。近几年也有研究表明，LSTM 加上合适的优化器和正则项就可以表现得很好。也不见得需要用非常神妙的奇技淫巧。
>

![](https://pic.superbed.cc/item/670cdcc2fa9f77b4dcbeaee1.png)

## 6.如何使用LM改善LAS

- Shallow Fusion

>这种结合方式非常符合直觉，就是将训练好的LAS和LM算出来的概率取log并进行相加（可以带权重），然后进行束搜索等方式找到最大值并采用。

![](https://pic.superbed.cc/item/670cdd6ffa9f77b4dcbebb3e.png)

- Deep Fusion

>这个是深度融合，也就是将隐藏层的内容拿出来，通过一个Network进行混合，来预测最终的概率分布。示意图如下。因此在采用这种方式的时候，我们还需要对这个用来混合的Network进行训练。但是这样问题也随之而来。就如图中所说，我们无法随时切换我们的 LM。因为不同的LM的隐藏层的每个维度所表示的含义可能都不同。因此我们要是切换LM就需要重新训练用来混合的网络。
>
>什么时候需要我们切换LM？就是当我们在面对不同领域（domain）的发音的时候。比如城市和程式发音相同。如果是工业领域，则城市的概率更高；如果是数学领域，可能就是程式的概率会更高。后续应用中，我们甚至可以针对一个人来做LM，根据这个人的常用词来做个性化的LM。

![](https://pic.superbed.cc/item/670cddb5fa9f77b4dcbebefe.png)

- Cold Fusion

>冷融合则与上面两个融合方式不同。这个融合是在LAS尚未完成训练时加入的。也就是在当前环境下，LM已经训练好，但是LAS还未完成训练。我们像Deep Fusion那样将二者相结合，然后再一起训练。
>
>这样做的好处是可以大大加快LAS的训练。因为此时LAS就可以不用去在意LM能够做到的事情，而专注于语音到文本的映射。
>
>也有一定的坏处，那就是这回是真的不能随便换LM了，一换就需要重新训练LAS。在Cold Fusion中，LM自LAS出生时就已经存在了，做一个引导成长的作用。因而切换LM，那么LAS也需要重新成长。
>

![](https://pic.superbed.cc/item/670cde42fa9f77b4dcbec76a.png)

