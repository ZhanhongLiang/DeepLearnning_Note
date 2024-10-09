# BERT模型原理

>参考月来客栈的This Post Is All You Need-----层层剥开Transfromer

## 面临问题

>在论文的介绍部分作者谈到，预训练语言模型（Language model pre-training）对于下游很多自然语言处理任务都有着显著的改善。但是作者继续说到，`现有预训练模型的网络结构限制了模型本身的表达能力，其中最主要的限制就是没有采用双向编码的方法来对输入进行编码`。
>
>`没有双向编码`
>
>例如在 OpenAI GPT 模型中，它使用了从左到右（left-to-right）的网络架构，这就使得模型在编码过程中只能够看到当前时刻之前的信息，而不能够同时捕捉到当前时刻之后的信息。

## 解决思路

>作者提出了BERT模型(Bidirectional Encoder Representations from Transformers)
>
>这一网络结构来实现模型的`双向编码学习能力`
>
>BERT训练过程中使用了基于`掩盖的语言模型(Masked Language Model, MLM)`,随机对输入序列中的某些位置进行遮蔽,然后通过模型来对其进行预测
>
>MLM预测任务能使得模型编码得到的结果同时包含上下文的语境信息,`训练BERT中加入下句预测任务(Next Sentence Prediction, NSP)`,同时输入到两句话到模型中，然后预测第2句话是不是第一句话的下一句话

# 技术实现

## BERT网络结构

![](https://pic.imgdb.cn/item/66f10299f21886ccc0765784.png)

>会发现BERT的结构跟Transformers结构类似，只不过Input Embedding位置是多出了Segment Embedding来进行判断两句话的顺序

## Input Embedding

![](https://pic.imgdb.cn/item/66f1044cf21886ccc077ff2b.png)

>包括Token Embedding   Positional Embeddign Segment Embedding 
>
>`'Positional Embedding'对于每个位置的编码并不是采用公式计算得到,而是类似普通的词嵌入一样为每一个位置初始化了一个向量，然后随着网络一起训练得到`。
>
>`BERT 开源的预训练模型最大只支持 512 个字符的长度，这是因为其在训练过程中（位置）词表的最大长度只有 512`。

### Segment Embedding

>BERT 的主要目的是构建一个通用的预训练模型，因此难免需要兼顾到各种NLP 任务场景下的输入。
>
>`Segment Embedding 的作用便是用来区分输入序列中的不同部分，其本质就是通过一个普通的词嵌入来区分每一个序列所处的位置`

![](https://pic.imgdb.cn/item/66f1058ff21886ccc078ffc9.png)

>CLS 代表一个句子的开头,特殊的分类标志
>
>SEP 代表将两句话分开的标志
>
>`Segment Embedding 层则同样是用来区分两句话所在的不同位置，对于每句话来说其内部各自的位置向量都是一样的`

## BertEncoder

![](https://pic.imgdb.cn/item/66f106bdf21886ccc07a18c1.png)

>`BertEncoder由多个BertLayer构成`

>BertLayer是类似Transformer中的结构,Attention位置也是self-Attention机制

## MLM和NSP任务

### MLM 任务

>对于 MLM 任务来说，其做法是随机掩盖掉输入序列中15%的 Token（即用“[MASK]”替换掉原有的 Token），然后在 BERT 的输出结果中取对应掩盖位置上的向量进行真实值预测
>
>接着作者提到，虽然 MLM 的这种做法能够得到一个很好的预训练模型，但是仍旧存在不足之处。由于在 fine-tuning 时，由于输入序列中并不存在“[MASK]”这样的 Token，因此这将导致 pre-training 和 fine-tuning 之间存在不匹配不一致的问题（GAP）。
>
>`为了解决这一问题，作者在原始 MLM 的基础了做了部分改动，即先选定15% 的 Token，然后将其中的80%替换为“[MASK]”、10% 随机替换为其它Token、剩下的10% 不变。最后取这15% 的 Token 对应的输出做分类来预测其真实值`。

![](https://pic.imgdb.cn/item/66f10933f21886ccc07c5dd5.png)

### NSP任务

>由于很多下游任务需要依赖于分析两句话之间的关系来进行建模，例如问题回答等。为了使得模型能够具备有这样的能力，作者在论文中又提出了二分类的下句预测任务。

![](https://pic.superbed.cc/item/66f10a822e3b94edab8ae540.png)

