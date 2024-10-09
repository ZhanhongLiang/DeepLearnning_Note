# 项目工程结构

## 工程结构

- `bert_base_chinese`目录中是BERT base中文预训练模型以及配置文件

  模型下载地址：<https://huggingface.co/bert-base-chinese/tree/main>

- `bert_base_uncased_english`目录中是BERT base英文预训练模型以及配置文件

  模型下载地址：<https://huggingface.co/bert-base-uncased/tree/main>

  注意：`config.json`中需要添加`"pooler_type": "first_token_transform"`这个参数

- `data`目录中是各个下游任务所使用到的数据集

  - `SingleSentenceClassification`是今日头条的15分类中文数据集；
  - `PairSentenceClassification`是MNLI（The Multi-Genre Natural Language Inference Corpus, 多类型自然语言推理数据库）数据集；
  - `MultipeChoice`是SWAG问题选择数据集
  - `SQuAD`是斯坦福大学开源的问答数据集1.1版本
  - `WikiText`是维基百科英文语料用于模型预训练
  - `SongCi`是宋词语料用于中文模型预训练
  - `ChineseNER`是用于训练中文命名体识别的数据集

- `model`目录中是各个模块的实现

  - ```
    BasicBert中是基础的BERT模型实现模块
    ```

    - `MyTransformer.py`是自注意力机制实现部分；
    - `BertEmbedding.py`是Input Embedding实现部分；
    - `BertConfig.py`用于导入开源的`config.json`配置文件；
    - `Bert.py`是BERT模型的实现部分；

  - ```
    DownstreamTasks目录是下游任务各个模块的实现
    ```

    - `BertForSentenceClassification.py`是单标签句子分类的实现部分；
    - `BertForMultipleChoice.py`是问题选择模型的实现部分；
    - `BertForQuestionAnswering.py`是问题回答（text span）模型的实现部分；
    - `BertForNSPAndMLM.py`是BERT模型预训练的两个任务实现部分；
    - `BertForTokenClassification.py`是字符分类（如：命名体识别）模型的实现部分；

- `Task`目录中是各个具体下游任务的训练和推理实现

  - `TaskForSingleSentenceClassification.py`是单标签单文本分类任务的训练和推理实现，可用于普通的文本分类任务；
  - `TaskForPairSentence.py`是文本对分类任务的训练和推理实现，可用于蕴含任务（例如MNLI数据集）；
  - `TaskForMultipleChoice.py`是问答选择任务的训练和推理实现，可用于问答选择任务（例如SWAG数据集）；
  - `TaskForSQuADQuestionAnswering.py`是问题回答任务的训练和推理实现，可用于问题问答任务（例如SQuAD数据集）；
  - `TaskForPretraining.py`是BERT模型中MLM和NSP两个预训练任务的实现部分，可用于BERT模型预训练；
  - `TaskForChineseNER.py`是基于BERT模型的命名体任务训练和推理部分的实现；

- `test`目录中是各个模块的测试案例

- `utils`是各个工具类的实现

  - `data_helpers.py`是各个下游任务的数据预处理及数据集构建模块；
  - `log_helper.py`是日志打印模块；
  - `creat_pretraining_data.py`是用于构造BERT预训练任务的数据集；



# 下游任务一:  文本分类任务

>`BERT`是一个强大的预训练模型，可以基于谷歌发布的预训练参数在各个下游任务中进行微调
>
>`基于 BERT的文本分类（准确的是单文本，也就是输入只包含一个句子）模型就是在原始的 BERT 模型后再加上一个分类层即可`
>
>分类层就是类似下面图片中一样，在原来模型中添加分类层

![](https://pic.superbed.cc/item/66f13b2c991d0115df1252bc.png)

## 任务构造原理

>​     总的来说，基于 BERT的文本分类（准确的是单文本，也就是输入只包含一个句子）`模型就是在原始的 BERT 模型后再加上一个分类层即可，类似的结构掌柜在文章[6]（基于 Transformer 的分类模型）中也介绍过，大家可以去看一下。同时，对于分类层的输入（也就是原始 BERT 的输出），默认情况下取 BERT输出结果中[CLS]位置对于的向量即可，当然也可以修改为其它方式，例如所有位置向量的均值等（见 2.4.3 节内容，将配置文件 config.json 中的 pooler_type 字段设置为"all_token_average"即可`。
>
>​    因此，对于基于 BERT 的文本分类模型来说其输入就是 BERT 的输入，输出则是每个类别对应的 logits 值。接下来，掌柜首先就来介绍如何构造文本分类的数据集。

## 数据预处理

### 输入数据

>`对于文本分类问题输入只有一个序列,构建数据集的时候不需要构造SegmentEmbedding，直接默认全为0`
>
>`对于文本分类这个场景来说,只需要构造原始文本对应的Token序列，首尾分别再加上一个[CLS]和[SEP]符作为输入就行`

### 数据集分析

![](https://pic.superbed.cc/item/66f140c3991d0115df13cab1.png)

>`①先进行 Tokenize处理 ->  ②使用谷歌开源的vocab.txt文件构造字典，不需要自己构造字典 -> ③根据字典将tokenize后的文本序列转换为Token序列,同时再Token序列的首尾加上[CLS]和[SEP]符号,并进行Padding->④就是根据第3步处理后的结果生成对应的Padding Mask向量`

## 数据集构建

### ①定义tokenize

>和transformers模型类似，需要先对文本进行tokenizer操作

```python
# Tokenizer的操作,底层源码不用管
import sys
sys.path.append('../')
from Tasks.TaskForSingleSentenceClassification import ModelConfig
from transformers import BertTokenizer

if __name__ == '__main__':
    model_config = ModelConfig()
    tokenizer = BertTokenizer.from_pretrained(model_config.pretrained_model_dir).tokenize
    print(tokenizer("青山不改，绿水长流，我们月来客栈见！"))
    print(tokenizer("10 年前的今天，纪念 5.12 汶川大地震 10 周年"))
```

```
['青', '山', '不', '改', '，', '绿', '水', '长', '流', '，', '我', '们', '月', '来', '客', '栈', '见', '！']
['10', '年', '前', '的', '今', '天', '，', '纪', '念', '5', '.', '12', '汶', '川', '大', '地', '震', '10', '周', '年']
```

### ② 建立词表

>因为谷歌是开源了 vocab.txt 词表,  不需要根据自己的语料来建立一个词表
>
>`不能根据自己的语料来构建词表`,因为当用自己的语料构建词表的时候，会导致后期提取的token.id是错误的

>该文件在"/media/wislab/Dataset_SSD2T/lzh/BertWithPretrained/utils/data_helpers.py"

```python
class Vocab:
    """
    根据本地的vocab文件，构造一个词表
    vocab = Vocab()
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['我'])  # 通过单词返回得到词表中对应的索引
    print(len(vocab))  # 返回词表长度
    """
    UNK = '[UNK]'

    def __init__(self, vocab_path):
        # 字典key:values,返回词表中每个词的索引
        self.stoi = {}
        # 数组代表列表,返回词表中每一个词
        self.itos = []
        # 打开vocab_path路径,用\n进行分割,然后
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f):
                w = word.strip('\n')
                self.stoi[w] = i
                self.itos.append(w)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)


def build_vocab(vocab_path):
    """
    vocab = Vocab()
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['我'])  # 通过单词返回得到词表中对应的索引
    """
    return Vocab(vocab_path)
```

>在经过上述代码处理后，我们便能够通过 vocab.itos 得到一个列表，返回词表中的每一个词；通过 vocab.itos[2]返回得到词表中对应索引位置上的词；通过vocab.stoi 得到一个字典，返回词表中每个词的索引；通过 vocab.stoi['月']返回得到词表中对应词的索引；通过 len(vocab)来返回词表的长度。如下便是建立后的词表

```
': 21027, '##齒': 21028, '##齡': 21029, '##齢': 21030, '##齣': 21031, '##齦': 21032, '##齿': 21033, '##龄': 21034, '##龅': 21035, '##龈': 21036, '##龊': 21037, '##龋': 21038, '##龌': 21039, '##龍': 21040, '##龐': 21041, '##龔': 21042, '##龕': 21043, '##龙': 21044, '##龚': 21045, '##龛': 21046, '##龜': 21047, '##龟': 21048, '##︰': 21049, '##︱': 21050, '##︶': 21051, '##︿': 21052, '##﹁': 21053, '##﹂': 21054, '##﹍': 21055, '##﹏': 21056, '##﹐': 21057, '##﹑': 21058, '##﹒': 21059, '##﹔': 21060, '##﹕': 21061, '##﹖': 21062, '##﹗': 21063, '##﹙': 21064, '##﹚': 21065, '##﹝': 21066, '##﹞': 21067, '##﹡': 21068, '##﹣': 21069, '##！': 21070, '##＂': 21071, '##＃': 21072, '##＄': 21073, '##％': 21074, '##＆': 21075, '##＇': 21076, '##（': 21077, '##）': 21078, '##＊': 21079, '##，': 21080, '##－': 21081, '##．': 21082, '##／': 21083, '##：': 21084, '##；': 21085, '##＜': 21086, '##？': 21087, '##＠': 21088, '##［': 21089, '##＼': 21090, '##］': 21091, '##＾': 21092, '##＿': 21093, '##｀': 21094, '##ｆ': 21095, '##ｈ': 21096, '##ｊ': 21097, '##ｕ': 21098, '##ｗ': 21099, '##ｚ': 21100, '##｛': 21101, '##｝': 21102, '##｡': 21103, '##｢': 21104, '##｣': 21105, '##､': 21106, '##･': 21107, '##ｯ': 21108, '##ｰ': 21109, '##ｲ': 21110, '##ｸ': 21111, '##ｼ': 21112, '##ｽ': 21113, '##ﾄ': 21114, '##ﾉ': 21115, '##ﾌ': 21116, '##ﾗ': 21117, '##ﾙ': 21118, '##ﾝ': 21119, '##ﾞ': 21120, '##ﾟ': 21121, '##￣': 21122, '##￥': 21123, '##👍': 21124, '##🔥': 21125, '##😂': 21126, '##😎': 21127}
2769
```



- 集成类中进行加载

```python
class LoadSingleSentenceClassificationDataset:
    def __init__(self,
                 vocab_path='./vocab.txt',  #
                 tokenizer=None,
                 batch_size=32,
                 max_sen_len=None,
                 split_sep='\n',
                 max_position_embeddings=512,
                 pad_index=0,
                 is_sample_shuffle=True
                 ):

        """

        :param vocab_path: 本地词表vocab.txt的路径
        :param tokenizer:
        :param batch_size:
        :param max_sen_len: 在对每个batch进行处理时的配置；
                            当max_sen_len = None时，即以每个batch中最长样本长度为标准，对其它进行padding
                            当max_sen_len = 'same'时，以整个数据集中最长样本为标准，对其它进行padding
                            当max_sen_len = 50， 表示以某个固定长度符样本进行padding，多余的截掉；
        :param split_sep: 文本和标签之前的分隔符，默认为'\t'
        :param max_position_embeddings: 指定最大样本长度，超过这个长度的部分将本截取掉
        :param is_sample_shuffle: 是否打乱训练集样本（只针对训练集）
                在后续构造DataLoader时，验证集和测试集均指定为了固定顺序（即不进行打乱），修改程序时请勿进行打乱
                因为当shuffle为True时，每次通过for循环遍历data_iter时样本的顺序都不一样，这会导致在模型预测时
                返回的标签顺序与原始的顺序不一样，不方便处理。

        """
        self.tokenizer = tokenizer
        # 构建词表，返回Vocab对象
        self.vocab = build_vocab(vocab_path)
        self.PAD_IDX = pad_index
        # SEP_IDX是两个句子分开的标志
        self.SEP_IDX = self.vocab['[SEP]']
        # CLS是句子之间开头位置
        self.CLS_IDX = self.vocab['[CLS]']
        # self.UNK_IDX = '[UNK]'

        self.batch_size = batch_size
        self.split_sep = split_sep
        self.max_position_embeddings = max_position_embeddings
        if isinstance(max_sen_len, int) and max_sen_len > max_position_embeddings:
            max_sen_len = max_position_embeddings
        self.max_sen_len = max_sen_len
        self.is_sample_shuffle = is_sample_shuffle

    @process_cache(unique_key=["max_sen_len"])
    def data_process(self, file_path=None):
        """
        将每一句话中的每一个词根据字典转换成索引的形式，同时返回所有样本中最长样本的长度
        :param file_path: 数据集路径
        :return:
        """
        raw_iter = open(file_path, encoding="utf8").readlines()
        data = []
        max_len = 0
        for raw in tqdm(raw_iter, ncols=80):
            line = raw.rstrip("\n").split(self.split_sep)
            s, l = line[0], line[1]
            tmp = [self.CLS_IDX] + [self.vocab[token] for token in self.tokenizer(s)]
            if len(tmp) > self.max_position_embeddings - 1:
                tmp = tmp[:self.max_position_embeddings - 1]  # BERT预训练模型只取前512个字符
            tmp += [self.SEP_IDX]
            tensor_ = torch.tensor(tmp, dtype=torch.long)
            l = torch.tensor(int(l), dtype=torch.long)
            max_len = max(max_len, tensor_.size(0))
            data.append((tensor_, l))
        return data, max_len

    def load_train_val_test_data(self, train_file_path=None,
                                 val_file_path=None,
                                 test_file_path=None,
                                 only_test=False):
        test_data, _ = self.data_process(file_path=test_file_path)
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch)
        if only_test:
            return test_iter
        train_data, max_sen_len = self.data_process(file_path=train_file_path)  # 得到处理好的所有样本
        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len
        val_data, _ = self.data_process(file_path=val_file_path)
        train_iter = DataLoader(train_data, batch_size=self.batch_size,  # 构造DataLoader
                                shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch)
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=False, collate_fn=self.generate_batch)
        return train_iter, test_iter, val_iter

    def generate_batch(self, data_batch):
        batch_sentence, batch_label = [], []
        for (sen, label) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
            batch_sentence.append(sen)
            batch_label.append(label)
        batch_sentence = pad_sequence(batch_sentence,  # [batch_size,max_len]
                                      padding_value=self.PAD_IDX,
                                      batch_first=False,
                                      max_len=self.max_sen_len)
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return batch_sentence, batch_label
```

>并在类的初始化过程中根据训练语料完成字典的构建等工作
>
>`该文件也在:"/media/wislab/Dataset_SSD2T/lzh/BertWithPretrained/utils/data_helpers.py"`
>
>当 max_sen_len = None 时，即以每个 batch 中最长样本长度为标准，对其它进行 padding；当 max_sen_len = 'same'时，以整个数据集中最长样本为标准，对其它进行 padding；当 max_sen_len = 50， 表示以某个固定长度符样本进行 padding，多余的截掉
>
>split_sep表示样本与标签之间的分隔符。is_sample_shuffle 表示是否打乱数据集

### ③ 转换为Token序列

- 利用tqdm进行显示训练的进度

```python
    """
    tqdm库，用于显示python库训练的进度
    只是一个三方库
    """
    @process_cache(unique_key=["max_sen_len"])
    def data_process(self, file_path=None):
        """
        将每一句话中的每一个词根据字典转换成索引的形式，同时返回所有样本中最长样本的长度
        :param file_path: 数据集路径
        :return:
        """
        # open函数
        # 且调用readlines()函数
        # 调用readline()可以每次读取一行内容，调用readlines()一次读取所有内容并按行返回list。因此，要根据需要决定怎么调用。
        raw_iter = open(file_path, encoding="utf8").readlines()
        data = []
        max_len = 0
        # tqdm库，就是按照
        for raw in tqdm(raw_iter, ncols=80):
            line = raw.rstrip("\n").split(self.split_sep)
            s, l = line[0], line[1]
            tmp = [self.CLS_IDX] + [self.vocab[token] for token in self.tokenizer(s)]
            if len(tmp) > self.max_position_embeddings - 1:
                tmp = tmp[:self.max_position_embeddings - 1]  # BERT预训练模型只取前512个字符
            tmp += [self.SEP_IDX]
            tensor_ = torch.tensor(tmp, dtype=torch.long)
            l = torch.tensor(int(l), dtype=torch.long)
            max_len = max(max_len, tensor_.size(0))
            data.append((tensor_, l))
        return data, max_len
```

- 这个是pytohn装饰器

process_cache是

```python
# 这个是python装饰器
# 在 data_process上面定义了@process_cache(unique_key=["max_sen_len"])
def process_cache(unique_key=None):
    """
    数据预处理结果缓存修饰器
    :param : unique_key
    :return:
    """
    if unique_key is None:
        raise ValueError(
            "unique_key 不能为空, 请指定相关数据集构造类的成员变量，如['top_k', 'cut_words', 'max_sen_len']")

    def decorating_function(func):
        def wrapper(*args, **kwargs):
            logging.info(f" ## 索引预处理缓存文件的参数为：{unique_key}")
            obj = args[0]  # 获取类对象，因为data_process(self, file_path=None)中的第1个参数为self
            file_path = kwargs['file_path']
            file_dir = f"{os.sep}".join(file_path.split(os.sep)[:-1])
            file_name = "".join(file_path.split(os.sep)[-1].split('.')[:-1])
            paras = f"cache_{file_name}_"
            for k in unique_key:
                paras += f"{k}{obj.__dict__[k]}_"  # 遍历对象中的所有参数
            cache_path = os.path.join(file_dir, paras[:-1] + '.pt')
            start_time = time.time()
            if not os.path.exists(cache_path):
                logging.info(f"缓存文件 {cache_path} 不存在，重新处理并缓存！")
                data = func(*args, **kwargs)
                with open(cache_path, 'wb') as f:
                    torch.save(data, f)
            else:
                logging.info(f"缓存文件 {cache_path} 存在，直接载入缓存文件！")
                with open(cache_path, 'rb') as f:
                    data = torch.load(f)
            end_time = time.time()
            logging.info(f"数据预处理一共耗时{(end_time - start_time):.3f}s")
            return data

        return wrapper

    return decorating_function
```

### ④ padding处理与mask

```python
def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    """
    对一个List中的元素进行padding
    Pad a list of variable length Tensors with ``padding_value``
    a = torch.ones(25)
    b = torch.ones(22)
    c = torch.ones(15)
    pad_sequence([a, b, c],max_len=None).size()
    torch.Size([25, 3])
        sequences:
        batch_first: 是否把batch_size放到第一个维度
        padding_value:
        max_len :
                当max_len = 50时，表示以某个固定长度对样本进行padding，多余的截掉；
                当max_len=None是，表示以当前batch中最长样本的长度对其它进行padding；
    Returns:
    """
    # 如果max_len=None的时候，那么就令max_len等于文本中最长长度
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    # 遍历每个Token序列，根据max_len进行padding
    for tensor in sequences:
        # 如果当前序列长度是小于max_len
        if tensor.size(0) < max_len:
            tensor = torch.cat([tensor, torch.tensor([padding_value] * (max_len - tensor.size(0)))], dim=0)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    # 将batch_size维度放在前面
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors
```

>  在上述代码中，第 1 行 sequences为待 padding 的序列所构成的列表，其中的每一个元素为一个样本的 Token 序列；batch_first 表示是否将 batch_size 这个维度放在第 1 个；max_len 表示指定最大序列长度，当 max_len = 50 时，表示以某个固定长度对样本进行 padding 多余的截掉，当 max_len=None 时表示以当前batch中最长样本的长度对其它进行padding。第2-3行用来获取padding的长度；第 5-11 行则是遍历每一个 Token 序列，根据 max_len 来进行 padding。第 12-13行是将 batch_size 这个维度放到最前面。



```python
    def generate_batch(self, data_batch):
        batch_sentence, batch_label = [], []
        for (sen, label) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
            batch_sentence.append(sen)
            batch_label.append(label)
        batch_sentence = pad_sequence(batch_sentence,  # [batch_size,max_len]
                                      padding_value=self.PAD_IDX,
                                      batch_first=False,
                                      max_len=self.max_sen_len)
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return batch_sentence, batch_label
```

>作用就是对每个 batch 的 Token 序列进行 padding 处理。最后，对于每一序列的 attention_mask 向量，我们只需要判断其是否等于padding_value 便可以得到这一结果，可见第 5 步中的使用示例

### ⑤构造DataLoader和使用案例

```python
    def load_train_val_test_data(self, train_file_path=None,
                                 val_file_path=None,
                                 test_file_path=None,
                                 only_test=False):
        # 提取文本出来
        test_data, _ = self.data_process(file_path=test_file_path)
        # DataLoader数据加载
        # generate_bath就是对
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch)
        if only_test:
            return test_iter
        train_data, max_sen_len = self.data_process(file_path=train_file_path)  # 得到处理好的所有样本
        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len
        val_data, _ = self.data_process(file_path=val_file_path)
        train_iter = DataLoader(train_data, batch_size=self.batch_size,  # 构造DataLoader
                                shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch)
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=False, collate_fn=self.generate_batch)
        return train_iter, test_iter, val_iter
```

返回的是DataLoader构建的数据迭代器

## 加载预训练模型

>在介绍模型微调之前，我们先来看看当我们拿到一个开源的模型参数后怎么读取以及分析。下面掌柜就以 huggingface 开源的 PyTorch 训练的 bert-base￾chinese 模型参数[10]为例进行介绍。

### ①查看模型参数

>PyTorch来读取和加载模型参数

```python
import sys
sys.path.append('../')
import torch
import os
from Tasks.TaskForSingleSentenceClassification import ModelConfig
from model.BasicBert.BertConfig import BertConfig
from model.BasicBert.Bert import BertModel

if __name__ == '__main__':
    model_config = ModelConfig()
    # /media/wislab/Dataset_SSD2T/lzh/BertWithPretrained/bert_base_chinese/pytorch_model.bin
    bin_path = os.path.join(model_config.pretrained_model_dir,'pytorch_model.bin')

    loaded_paras = torch.load(bin_path)
    print(type(loaded_paras))
    print(len(list(loaded_paras)))
    print(list(loaded_paras.keys()))
    for name in loaded_paras.keys():
        print(f"### 参数:{name},形状{loaded_paras[name].size()}")
```

- 这个是查看预训练模型的参数

```
[2024-09-25 20:06:21] - INFO: 成功导入BERT配置文件 /media/wislab/Dataset_SSD2T/lzh/BertWithPretrained/bert_base_chinese/config.json
[2024-09-25 20:06:21] - INFO:  ### 将当前配置打印到日志文件中 
[2024-09-25 20:06:21] - INFO: ###  project_dir = /media/wislab/Dataset_SSD2T/lzh/BertWithPretrained
[2024-09-25 20:06:21] - INFO: ###  dataset_dir = /media/wislab/Dataset_SSD2T/lzh/BertWithPretrained/data/SingleSentenceClassification
[2024-09-25 20:06:21] - INFO: ###  pretrained_model_dir = /media/wislab/Dataset_SSD2T/lzh/BertWithPretrained/bert_base_chinese
[2024-09-25 20:06:21] - INFO: ###  vocab_path = /media/wislab/Dataset_SSD2T/lzh/BertWithPretrained/bert_base_chinese/vocab.txt
[2024-09-25 20:06:21] - INFO: ###  device = cpu
[2024-09-25 20:06:21] - INFO: ###  train_file_path = /media/wislab/Dataset_SSD2T/lzh/BertWithPretrained/data/SingleSentenceClassification/toutiao_train.txt
[2024-09-25 20:06:21] - INFO: ###  val_file_path = /media/wislab/Dataset_SSD2T/lzh/BertWithPretrained/data/SingleSentenceClassification/toutiao_val.txt
[2024-09-25 20:06:21] - INFO: ###  test_file_path = /media/wislab/Dataset_SSD2T/lzh/BertWithPretrained/data/SingleSentenceClassification/toutiao_test.txt
[2024-09-25 20:06:21] - INFO: ###  model_save_dir = /media/wislab/Dataset_SSD2T/lzh/BertWithPretrained/cache
[2024-09-25 20:06:21] - INFO: ###  logs_save_dir = /media/wislab/Dataset_SSD2T/lzh/BertWithPretrained/logs
[2024-09-25 20:06:21] - INFO: ###  split_sep = _!_
[2024-09-25 20:06:21] - INFO: ###  is_sample_shuffle = True
[2024-09-25 20:06:21] - INFO: ###  batch_size = 64
[2024-09-25 20:06:21] - INFO: ###  max_sen_len = None
[2024-09-25 20:06:21] - INFO: ###  num_labels = 15
[2024-09-25 20:06:21] - INFO: ###  epochs = 10
[2024-09-25 20:06:21] - INFO: ###  model_val_per_epoch = 2
[2024-09-25 20:06:21] - INFO: ###  vocab_size = 21128
[2024-09-25 20:06:21] - INFO: ###  hidden_size = 768
[2024-09-25 20:06:21] - INFO: ###  num_hidden_layers = 12
[2024-09-25 20:06:21] - INFO: ###  num_attention_heads = 12
[2024-09-25 20:06:21] - INFO: ###  hidden_act = gelu
[2024-09-25 20:06:21] - INFO: ###  intermediate_size = 3072
[2024-09-25 20:06:21] - INFO: ###  pad_token_id = 0
[2024-09-25 20:06:21] - INFO: ###  hidden_dropout_prob = 0.1
[2024-09-25 20:06:21] - INFO: ###  attention_probs_dropout_prob = 0.1
[2024-09-25 20:06:21] - INFO: ###  max_position_embeddings = 512
[2024-09-25 20:06:21] - INFO: ###  type_vocab_size = 2
[2024-09-25 20:06:21] - INFO: ###  initializer_range = 0.02
[2024-09-25 20:06:21] - INFO: ###  directionality = bidi
[2024-09-25 20:06:21] - INFO: ###  pooler_fc_size = 768
[2024-09-25 20:06:21] - INFO: ###  pooler_num_attention_heads = 12
[2024-09-25 20:06:21] - INFO: ###  pooler_num_fc_layers = 3
[2024-09-25 20:06:21] - INFO: ###  pooler_size_per_head = 128
[2024-09-25 20:06:21] - INFO: ###  pooler_type = first_token_transform
207
### 参数:bert.embeddings.word_embeddings.weight,形状torch.Size([21128, 768])
### 参数:bert.embeddings.position_embeddings.weight,形状torch.Size([512, 768])
### 参数:bert.embeddings.token_type_embeddings.weight,形状torch.Size([2, 768])
### 参数:bert.embeddings.LayerNorm.gamma,形状torch.Size([768])
### 参数:bert.embeddings.LayerNorm.beta,形状torch.Size([768])
### 参数:bert.encoder.layer.0.attention.self.query.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.0.attention.self.query.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.0.attention.self.key.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.0.attention.self.key.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.0.attention.self.value.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.0.attention.self.value.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.0.attention.output.dense.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.0.attention.output.dense.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.0.attention.output.LayerNorm.gamma,形状torch.Size([768])
### 参数:bert.encoder.layer.0.attention.output.LayerNorm.beta,形状torch.Size([768])
### 参数:bert.encoder.layer.0.intermediate.dense.weight,形状torch.Size([3072, 768])
### 参数:bert.encoder.layer.0.intermediate.dense.bias,形状torch.Size([3072])
### 参数:bert.encoder.layer.0.output.dense.weight,形状torch.Size([768, 3072])
### 参数:bert.encoder.layer.0.output.dense.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.0.output.LayerNorm.gamma,形状torch.Size([768])
### 参数:bert.encoder.layer.0.output.LayerNorm.beta,形状torch.Size([768])
### 参数:bert.encoder.layer.1.attention.self.query.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.1.attention.self.query.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.1.attention.self.key.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.1.attention.self.key.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.1.attention.self.value.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.1.attention.self.value.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.1.attention.output.dense.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.1.attention.output.dense.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.1.attention.output.LayerNorm.gamma,形状torch.Size([768])
### 参数:bert.encoder.layer.1.attention.output.LayerNorm.beta,形状torch.Size([768])
### 参数:bert.encoder.layer.1.intermediate.dense.weight,形状torch.Size([3072, 768])
### 参数:bert.encoder.layer.1.intermediate.dense.bias,形状torch.Size([3072])
### 参数:bert.encoder.layer.1.output.dense.weight,形状torch.Size([768, 3072])
### 参数:bert.encoder.layer.1.output.dense.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.1.output.LayerNorm.gamma,形状torch.Size([768])
### 参数:bert.encoder.layer.1.output.LayerNorm.beta,形状torch.Size([768])
### 参数:bert.encoder.layer.2.attention.self.query.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.2.attention.self.query.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.2.attention.self.key.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.2.attention.self.key.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.2.attention.self.value.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.2.attention.self.value.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.2.attention.output.dense.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.2.attention.output.dense.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.2.attention.output.LayerNorm.gamma,形状torch.Size([768])
### 参数:bert.encoder.layer.2.attention.output.LayerNorm.beta,形状torch.Size([768])
### 参数:bert.encoder.layer.2.intermediate.dense.weight,形状torch.Size([3072, 768])
### 参数:bert.encoder.layer.2.intermediate.dense.bias,形状torch.Size([3072])
### 参数:bert.encoder.layer.2.output.dense.weight,形状torch.Size([768, 3072])
### 参数:bert.encoder.layer.2.output.dense.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.2.output.LayerNorm.gamma,形状torch.Size([768])
### 参数:bert.encoder.layer.2.output.LayerNorm.beta,形状torch.Size([768])
### 参数:bert.encoder.layer.3.attention.self.query.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.3.attention.self.query.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.3.attention.self.key.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.3.attention.self.key.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.3.attention.self.value.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.3.attention.self.value.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.3.attention.output.dense.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.3.attention.output.dense.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.3.attention.output.LayerNorm.gamma,形状torch.Size([768])
### 参数:bert.encoder.layer.3.attention.output.LayerNorm.beta,形状torch.Size([768])
### 参数:bert.encoder.layer.3.intermediate.dense.weight,形状torch.Size([3072, 768])
### 参数:bert.encoder.layer.3.intermediate.dense.bias,形状torch.Size([3072])
### 参数:bert.encoder.layer.3.output.dense.weight,形状torch.Size([768, 3072])
### 参数:bert.encoder.layer.3.output.dense.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.3.output.LayerNorm.gamma,形状torch.Size([768])
### 参数:bert.encoder.layer.3.output.LayerNorm.beta,形状torch.Size([768])
### 参数:bert.encoder.layer.4.attention.self.query.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.4.attention.self.query.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.4.attention.self.key.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.4.attention.self.key.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.4.attention.self.value.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.4.attention.self.value.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.4.attention.output.dense.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.4.attention.output.dense.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.4.attention.output.LayerNorm.gamma,形状torch.Size([768])
### 参数:bert.encoder.layer.4.attention.output.LayerNorm.beta,形状torch.Size([768])
### 参数:bert.encoder.layer.4.intermediate.dense.weight,形状torch.Size([3072, 768])
### 参数:bert.encoder.layer.4.intermediate.dense.bias,形状torch.Size([3072])
### 参数:bert.encoder.layer.4.output.dense.weight,形状torch.Size([768, 3072])
### 参数:bert.encoder.layer.4.output.dense.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.4.output.LayerNorm.gamma,形状torch.Size([768])
### 参数:bert.encoder.layer.4.output.LayerNorm.beta,形状torch.Size([768])
### 参数:bert.encoder.layer.5.attention.self.query.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.5.attention.self.query.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.5.attention.self.key.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.5.attention.self.key.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.5.attention.self.value.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.5.attention.self.value.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.5.attention.output.dense.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.5.attention.output.dense.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.5.attention.output.LayerNorm.gamma,形状torch.Size([768])
### 参数:bert.encoder.layer.5.attention.output.LayerNorm.beta,形状torch.Size([768])
### 参数:bert.encoder.layer.5.intermediate.dense.weight,形状torch.Size([3072, 768])
### 参数:bert.encoder.layer.5.intermediate.dense.bias,形状torch.Size([3072])
### 参数:bert.encoder.layer.5.output.dense.weight,形状torch.Size([768, 3072])
### 参数:bert.encoder.layer.5.output.dense.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.5.output.LayerNorm.gamma,形状torch.Size([768])
### 参数:bert.encoder.layer.5.output.LayerNorm.beta,形状torch.Size([768])
### 参数:bert.encoder.layer.6.attention.self.query.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.6.attention.self.query.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.6.attention.self.key.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.6.attention.self.key.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.6.attention.self.value.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.6.attention.self.value.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.6.attention.output.dense.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.6.attention.output.dense.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.6.attention.output.LayerNorm.gamma,形状torch.Size([768])
### 参数:bert.encoder.layer.6.attention.output.LayerNorm.beta,形状torch.Size([768])
### 参数:bert.encoder.layer.6.intermediate.dense.weight,形状torch.Size([3072, 768])
### 参数:bert.encoder.layer.6.intermediate.dense.bias,形状torch.Size([3072])
### 参数:bert.encoder.layer.6.output.dense.weight,形状torch.Size([768, 3072])
### 参数:bert.encoder.layer.6.output.dense.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.6.output.LayerNorm.gamma,形状torch.Size([768])
### 参数:bert.encoder.layer.6.output.LayerNorm.beta,形状torch.Size([768])
### 参数:bert.encoder.layer.7.attention.self.query.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.7.attention.self.query.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.7.attention.self.key.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.7.attention.self.key.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.7.attention.self.value.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.7.attention.self.value.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.7.attention.output.dense.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.7.attention.output.dense.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.7.attention.output.LayerNorm.gamma,形状torch.Size([768])
### 参数:bert.encoder.layer.7.attention.output.LayerNorm.beta,形状torch.Size([768])
### 参数:bert.encoder.layer.7.intermediate.dense.weight,形状torch.Size([3072, 768])
### 参数:bert.encoder.layer.7.intermediate.dense.bias,形状torch.Size([3072])
### 参数:bert.encoder.layer.7.output.dense.weight,形状torch.Size([768, 3072])
### 参数:bert.encoder.layer.7.output.dense.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.7.output.LayerNorm.gamma,形状torch.Size([768])
### 参数:bert.encoder.layer.7.output.LayerNorm.beta,形状torch.Size([768])
### 参数:bert.encoder.layer.8.attention.self.query.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.8.attention.self.query.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.8.attention.self.key.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.8.attention.self.key.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.8.attention.self.value.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.8.attention.self.value.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.8.attention.output.dense.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.8.attention.output.dense.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.8.attention.output.LayerNorm.gamma,形状torch.Size([768])
### 参数:bert.encoder.layer.8.attention.output.LayerNorm.beta,形状torch.Size([768])
### 参数:bert.encoder.layer.8.intermediate.dense.weight,形状torch.Size([3072, 768])
### 参数:bert.encoder.layer.8.intermediate.dense.bias,形状torch.Size([3072])
### 参数:bert.encoder.layer.8.output.dense.weight,形状torch.Size([768, 3072])
### 参数:bert.encoder.layer.8.output.dense.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.8.output.LayerNorm.gamma,形状torch.Size([768])
### 参数:bert.encoder.layer.8.output.LayerNorm.beta,形状torch.Size([768])
### 参数:bert.encoder.layer.9.attention.self.query.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.9.attention.self.query.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.9.attention.self.key.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.9.attention.self.key.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.9.attention.self.value.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.9.attention.self.value.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.9.attention.output.dense.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.9.attention.output.dense.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.9.attention.output.LayerNorm.gamma,形状torch.Size([768])
### 参数:bert.encoder.layer.9.attention.output.LayerNorm.beta,形状torch.Size([768])
### 参数:bert.encoder.layer.9.intermediate.dense.weight,形状torch.Size([3072, 768])
### 参数:bert.encoder.layer.9.intermediate.dense.bias,形状torch.Size([3072])
### 参数:bert.encoder.layer.9.output.dense.weight,形状torch.Size([768, 3072])
### 参数:bert.encoder.layer.9.output.dense.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.9.output.LayerNorm.gamma,形状torch.Size([768])
### 参数:bert.encoder.layer.9.output.LayerNorm.beta,形状torch.Size([768])
### 参数:bert.encoder.layer.10.attention.self.query.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.10.attention.self.query.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.10.attention.self.key.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.10.attention.self.key.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.10.attention.self.value.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.10.attention.self.value.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.10.attention.output.dense.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.10.attention.output.dense.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.10.attention.output.LayerNorm.gamma,形状torch.Size([768])
### 参数:bert.encoder.layer.10.attention.output.LayerNorm.beta,形状torch.Size([768])
### 参数:bert.encoder.layer.10.intermediate.dense.weight,形状torch.Size([3072, 768])
### 参数:bert.encoder.layer.10.intermediate.dense.bias,形状torch.Size([3072])
### 参数:bert.encoder.layer.10.output.dense.weight,形状torch.Size([768, 3072])
### 参数:bert.encoder.layer.10.output.dense.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.10.output.LayerNorm.gamma,形状torch.Size([768])
### 参数:bert.encoder.layer.10.output.LayerNorm.beta,形状torch.Size([768])
### 参数:bert.encoder.layer.11.attention.self.query.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.11.attention.self.query.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.11.attention.self.key.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.11.attention.self.key.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.11.attention.self.value.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.11.attention.self.value.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.11.attention.output.dense.weight,形状torch.Size([768, 768])
### 参数:bert.encoder.layer.11.attention.output.dense.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.11.attention.output.LayerNorm.gamma,形状torch.Size([768])
### 参数:bert.encoder.layer.11.attention.output.LayerNorm.beta,形状torch.Size([768])
### 参数:bert.encoder.layer.11.intermediate.dense.weight,形状torch.Size([3072, 768])
### 参数:bert.encoder.layer.11.intermediate.dense.bias,形状torch.Size([3072])
### 参数:bert.encoder.layer.11.output.dense.weight,形状torch.Size([768, 3072])
### 参数:bert.encoder.layer.11.output.dense.bias,形状torch.Size([768])
### 参数:bert.encoder.layer.11.output.LayerNorm.gamma,形状torch.Size([768])
### 参数:bert.encoder.layer.11.output.LayerNorm.beta,形状torch.Size([768])
### 参数:bert.pooler.dense.weight,形状torch.Size([768, 768])
### 参数:bert.pooler.dense.bias,形状torch.Size([768])
### 参数:cls.predictions.bias,形状torch.Size([21128])
### 参数:cls.predictions.transform.dense.weight,形状torch.Size([768, 768])
### 参数:cls.predictions.transform.dense.bias,形状torch.Size([768])
### 参数:cls.predictions.transform.LayerNorm.gamma,形状torch.Size([768])
### 参数:cls.predictions.transform.LayerNorm.beta,形状torch.Size([768])
### 参数:cls.predictions.decoder.weight,形状torch.Size([21128, 768])
### 参数:cls.seq_relationship.weight,形状torch.Size([2, 768])
### 参数:cls.seq_relationship.bias,形状torch.Size([2])
```

>但是如果要将我们的网络结果迁移到官方的bert模型中，但是参数对应不上应该怎么办??

### ② 载入数据并初始化

>但是对于如何载入已有参数来初始化网络中的参数还并未介绍。在将本地参数迁移到一个新的模型之前，除了像上面那样分析本地参数之外，我们还需要将网络的参数信息也打印出来看一下，以便将两者一一对应上

#### `查看预训练模型和自己模型参数的区别`

```python
import sys
sys.path.append('../')
import torch
import os
from Tasks.TaskForSingleSentenceClassification import ModelConfig
from model.BasicBert.BertConfig import BertConfig
from model.BasicBert.Bert import BertModel

if __name__ == '__main__':
    model_config = ModelConfig()
    # /media/wislab/Dataset_SSD2T/lzh/BertWithPretrained/bert_base_chinese/pytorch_model.bin
    bin_path = os.path.join(model_config.pretrained_model_dir,'pytorch_model.bin')

    loaded_paras = torch.load(bin_path)
    # print(type(loaded_paras))
    # print(len(list(loaded_paras)))
    # print(list(loaded_paras.keys()))
    print(len(list(loaded_paras)))
    for name in loaded_paras.keys():
        print(f"### 参数:{name},形状{loaded_paras[name].size()}")
    
    josn_file = os.path.join(model_config.pretrained_model_dir,'config.json')
    # 利用BertConfig
    config = BertConfig.from_json_file(josn_file)
    bert_model = BertModel(config=config)
    print(len(bert_model.state_dict()))
    for param_tensor in bert_model.state_dict():
        print(param_tensor,"\t",bert_model.state_dict()[param_tensor].size())
    
```

```
200
bert_embeddings.position_ids     torch.Size([1, 512])
bert_embeddings.word_embeddings.embedding.weight         torch.Size([21128, 768])
bert_embeddings.position_embeddings.embedding.weight     torch.Size([512, 768])
bert_embeddings.token_type_embeddings.embedding.weight   torch.Size([2, 768])
bert_embeddings.LayerNorm.weight         torch.Size([768])
bert_embeddings.LayerNorm.bias   torch.Size([768])
bert_encoder.bert_layers.0.bert_attention.self.multi_head_attention.q_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.0.bert_attention.self.multi_head_attention.q_proj.bias          torch.Size([768])
bert_encoder.bert_layers.0.bert_attention.self.multi_head_attention.k_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.0.bert_attention.self.multi_head_attention.k_proj.bias          torch.Size([768])
bert_encoder.bert_layers.0.bert_attention.self.multi_head_attention.v_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.0.bert_attention.self.multi_head_attention.v_proj.bias          torch.Size([768])
bert_encoder.bert_layers.0.bert_attention.self.multi_head_attention.out_proj.weight      torch.Size([768, 768])
bert_encoder.bert_layers.0.bert_attention.self.multi_head_attention.out_proj.bias        torch.Size([768])
bert_encoder.bert_layers.0.bert_attention.output.LayerNorm.weight        torch.Size([768])
bert_encoder.bert_layers.0.bert_attention.output.LayerNorm.bias          torch.Size([768])
bert_encoder.bert_layers.0.bert_intermediate.dense.weight        torch.Size([3072, 768])
bert_encoder.bert_layers.0.bert_intermediate.dense.bias          torch.Size([3072])
bert_encoder.bert_layers.0.bert_output.dense.weight      torch.Size([768, 3072])
bert_encoder.bert_layers.0.bert_output.dense.bias        torch.Size([768])
bert_encoder.bert_layers.0.bert_output.LayerNorm.weight          torch.Size([768])
bert_encoder.bert_layers.0.bert_output.LayerNorm.bias    torch.Size([768])
bert_encoder.bert_layers.1.bert_attention.self.multi_head_attention.q_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.1.bert_attention.self.multi_head_attention.q_proj.bias          torch.Size([768])
bert_encoder.bert_layers.1.bert_attention.self.multi_head_attention.k_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.1.bert_attention.self.multi_head_attention.k_proj.bias          torch.Size([768])
bert_encoder.bert_layers.1.bert_attention.self.multi_head_attention.v_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.1.bert_attention.self.multi_head_attention.v_proj.bias          torch.Size([768])
bert_encoder.bert_layers.1.bert_attention.self.multi_head_attention.out_proj.weight      torch.Size([768, 768])
bert_encoder.bert_layers.1.bert_attention.self.multi_head_attention.out_proj.bias        torch.Size([768])
bert_encoder.bert_layers.1.bert_attention.output.LayerNorm.weight        torch.Size([768])
bert_encoder.bert_layers.1.bert_attention.output.LayerNorm.bias          torch.Size([768])
bert_encoder.bert_layers.1.bert_intermediate.dense.weight        torch.Size([3072, 768])
bert_encoder.bert_layers.1.bert_intermediate.dense.bias          torch.Size([3072])
bert_encoder.bert_layers.1.bert_output.dense.weight      torch.Size([768, 3072])
bert_encoder.bert_layers.1.bert_output.dense.bias        torch.Size([768])
bert_encoder.bert_layers.1.bert_output.LayerNorm.weight          torch.Size([768])
bert_encoder.bert_layers.1.bert_output.LayerNorm.bias    torch.Size([768])
bert_encoder.bert_layers.2.bert_attention.self.multi_head_attention.q_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.2.bert_attention.self.multi_head_attention.q_proj.bias          torch.Size([768])
bert_encoder.bert_layers.2.bert_attention.self.multi_head_attention.k_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.2.bert_attention.self.multi_head_attention.k_proj.bias          torch.Size([768])
bert_encoder.bert_layers.2.bert_attention.self.multi_head_attention.v_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.2.bert_attention.self.multi_head_attention.v_proj.bias          torch.Size([768])
bert_encoder.bert_layers.2.bert_attention.self.multi_head_attention.out_proj.weight      torch.Size([768, 768])
bert_encoder.bert_layers.2.bert_attention.self.multi_head_attention.out_proj.bias        torch.Size([768])
bert_encoder.bert_layers.2.bert_attention.output.LayerNorm.weight        torch.Size([768])
bert_encoder.bert_layers.2.bert_attention.output.LayerNorm.bias          torch.Size([768])
bert_encoder.bert_layers.2.bert_intermediate.dense.weight        torch.Size([3072, 768])
bert_encoder.bert_layers.2.bert_intermediate.dense.bias          torch.Size([3072])
bert_encoder.bert_layers.2.bert_output.dense.weight      torch.Size([768, 3072])
bert_encoder.bert_layers.2.bert_output.dense.bias        torch.Size([768])
bert_encoder.bert_layers.2.bert_output.LayerNorm.weight          torch.Size([768])
bert_encoder.bert_layers.2.bert_output.LayerNorm.bias    torch.Size([768])
bert_encoder.bert_layers.3.bert_attention.self.multi_head_attention.q_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.3.bert_attention.self.multi_head_attention.q_proj.bias          torch.Size([768])
bert_encoder.bert_layers.3.bert_attention.self.multi_head_attention.k_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.3.bert_attention.self.multi_head_attention.k_proj.bias          torch.Size([768])
bert_encoder.bert_layers.3.bert_attention.self.multi_head_attention.v_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.3.bert_attention.self.multi_head_attention.v_proj.bias          torch.Size([768])
bert_encoder.bert_layers.3.bert_attention.self.multi_head_attention.out_proj.weight      torch.Size([768, 768])
bert_encoder.bert_layers.3.bert_attention.self.multi_head_attention.out_proj.bias        torch.Size([768])
bert_encoder.bert_layers.3.bert_attention.output.LayerNorm.weight        torch.Size([768])
bert_encoder.bert_layers.3.bert_attention.output.LayerNorm.bias          torch.Size([768])
bert_encoder.bert_layers.3.bert_intermediate.dense.weight        torch.Size([3072, 768])
bert_encoder.bert_layers.3.bert_intermediate.dense.bias          torch.Size([3072])
bert_encoder.bert_layers.3.bert_output.dense.weight      torch.Size([768, 3072])
bert_encoder.bert_layers.3.bert_output.dense.bias        torch.Size([768])
bert_encoder.bert_layers.3.bert_output.LayerNorm.weight          torch.Size([768])
bert_encoder.bert_layers.3.bert_output.LayerNorm.bias    torch.Size([768])
bert_encoder.bert_layers.4.bert_attention.self.multi_head_attention.q_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.4.bert_attention.self.multi_head_attention.q_proj.bias          torch.Size([768])
bert_encoder.bert_layers.4.bert_attention.self.multi_head_attention.k_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.4.bert_attention.self.multi_head_attention.k_proj.bias          torch.Size([768])
bert_encoder.bert_layers.4.bert_attention.self.multi_head_attention.v_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.4.bert_attention.self.multi_head_attention.v_proj.bias          torch.Size([768])
bert_encoder.bert_layers.4.bert_attention.self.multi_head_attention.out_proj.weight      torch.Size([768, 768])
bert_encoder.bert_layers.4.bert_attention.self.multi_head_attention.out_proj.bias        torch.Size([768])
bert_encoder.bert_layers.4.bert_attention.output.LayerNorm.weight        torch.Size([768])
bert_encoder.bert_layers.4.bert_attention.output.LayerNorm.bias          torch.Size([768])
bert_encoder.bert_layers.4.bert_intermediate.dense.weight        torch.Size([3072, 768])
bert_encoder.bert_layers.4.bert_intermediate.dense.bias          torch.Size([3072])
bert_encoder.bert_layers.4.bert_output.dense.weight      torch.Size([768, 3072])
bert_encoder.bert_layers.4.bert_output.dense.bias        torch.Size([768])
bert_encoder.bert_layers.4.bert_output.LayerNorm.weight          torch.Size([768])
bert_encoder.bert_layers.4.bert_output.LayerNorm.bias    torch.Size([768])
bert_encoder.bert_layers.5.bert_attention.self.multi_head_attention.q_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.5.bert_attention.self.multi_head_attention.q_proj.bias          torch.Size([768])
bert_encoder.bert_layers.5.bert_attention.self.multi_head_attention.k_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.5.bert_attention.self.multi_head_attention.k_proj.bias          torch.Size([768])
bert_encoder.bert_layers.5.bert_attention.self.multi_head_attention.v_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.5.bert_attention.self.multi_head_attention.v_proj.bias          torch.Size([768])
bert_encoder.bert_layers.5.bert_attention.self.multi_head_attention.out_proj.weight      torch.Size([768, 768])
bert_encoder.bert_layers.5.bert_attention.self.multi_head_attention.out_proj.bias        torch.Size([768])
bert_encoder.bert_layers.5.bert_attention.output.LayerNorm.weight        torch.Size([768])
bert_encoder.bert_layers.5.bert_attention.output.LayerNorm.bias          torch.Size([768])
bert_encoder.bert_layers.5.bert_intermediate.dense.weight        torch.Size([3072, 768])
bert_encoder.bert_layers.5.bert_intermediate.dense.bias          torch.Size([3072])
bert_encoder.bert_layers.5.bert_output.dense.weight      torch.Size([768, 3072])
bert_encoder.bert_layers.5.bert_output.dense.bias        torch.Size([768])
bert_encoder.bert_layers.5.bert_output.LayerNorm.weight          torch.Size([768])
bert_encoder.bert_layers.5.bert_output.LayerNorm.bias    torch.Size([768])
bert_encoder.bert_layers.6.bert_attention.self.multi_head_attention.q_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.6.bert_attention.self.multi_head_attention.q_proj.bias          torch.Size([768])
bert_encoder.bert_layers.6.bert_attention.self.multi_head_attention.k_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.6.bert_attention.self.multi_head_attention.k_proj.bias          torch.Size([768])
bert_encoder.bert_layers.6.bert_attention.self.multi_head_attention.v_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.6.bert_attention.self.multi_head_attention.v_proj.bias          torch.Size([768])
bert_encoder.bert_layers.6.bert_attention.self.multi_head_attention.out_proj.weight      torch.Size([768, 768])
bert_encoder.bert_layers.6.bert_attention.self.multi_head_attention.out_proj.bias        torch.Size([768])
bert_encoder.bert_layers.6.bert_attention.output.LayerNorm.weight        torch.Size([768])
bert_encoder.bert_layers.6.bert_attention.output.LayerNorm.bias          torch.Size([768])
bert_encoder.bert_layers.6.bert_intermediate.dense.weight        torch.Size([3072, 768])
bert_encoder.bert_layers.6.bert_intermediate.dense.bias          torch.Size([3072])
bert_encoder.bert_layers.6.bert_output.dense.weight      torch.Size([768, 3072])
bert_encoder.bert_layers.6.bert_output.dense.bias        torch.Size([768])
bert_encoder.bert_layers.6.bert_output.LayerNorm.weight          torch.Size([768])
bert_encoder.bert_layers.6.bert_output.LayerNorm.bias    torch.Size([768])
bert_encoder.bert_layers.7.bert_attention.self.multi_head_attention.q_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.7.bert_attention.self.multi_head_attention.q_proj.bias          torch.Size([768])
bert_encoder.bert_layers.7.bert_attention.self.multi_head_attention.k_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.7.bert_attention.self.multi_head_attention.k_proj.bias          torch.Size([768])
bert_encoder.bert_layers.7.bert_attention.self.multi_head_attention.v_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.7.bert_attention.self.multi_head_attention.v_proj.bias          torch.Size([768])
bert_encoder.bert_layers.7.bert_attention.self.multi_head_attention.out_proj.weight      torch.Size([768, 768])
bert_encoder.bert_layers.7.bert_attention.self.multi_head_attention.out_proj.bias        torch.Size([768])
bert_encoder.bert_layers.7.bert_attention.output.LayerNorm.weight        torch.Size([768])
bert_encoder.bert_layers.7.bert_attention.output.LayerNorm.bias          torch.Size([768])
bert_encoder.bert_layers.7.bert_intermediate.dense.weight        torch.Size([3072, 768])
bert_encoder.bert_layers.7.bert_intermediate.dense.bias          torch.Size([3072])
bert_encoder.bert_layers.7.bert_output.dense.weight      torch.Size([768, 3072])
bert_encoder.bert_layers.7.bert_output.dense.bias        torch.Size([768])
bert_encoder.bert_layers.7.bert_output.LayerNorm.weight          torch.Size([768])
bert_encoder.bert_layers.7.bert_output.LayerNorm.bias    torch.Size([768])
bert_encoder.bert_layers.8.bert_attention.self.multi_head_attention.q_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.8.bert_attention.self.multi_head_attention.q_proj.bias          torch.Size([768])
bert_encoder.bert_layers.8.bert_attention.self.multi_head_attention.k_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.8.bert_attention.self.multi_head_attention.k_proj.bias          torch.Size([768])
bert_encoder.bert_layers.8.bert_attention.self.multi_head_attention.v_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.8.bert_attention.self.multi_head_attention.v_proj.bias          torch.Size([768])
bert_encoder.bert_layers.8.bert_attention.self.multi_head_attention.out_proj.weight      torch.Size([768, 768])
bert_encoder.bert_layers.8.bert_attention.self.multi_head_attention.out_proj.bias        torch.Size([768])
bert_encoder.bert_layers.8.bert_attention.output.LayerNorm.weight        torch.Size([768])
bert_encoder.bert_layers.8.bert_attention.output.LayerNorm.bias          torch.Size([768])
bert_encoder.bert_layers.8.bert_intermediate.dense.weight        torch.Size([3072, 768])
bert_encoder.bert_layers.8.bert_intermediate.dense.bias          torch.Size([3072])
bert_encoder.bert_layers.8.bert_output.dense.weight      torch.Size([768, 3072])
bert_encoder.bert_layers.8.bert_output.dense.bias        torch.Size([768])
bert_encoder.bert_layers.8.bert_output.LayerNorm.weight          torch.Size([768])
bert_encoder.bert_layers.8.bert_output.LayerNorm.bias    torch.Size([768])
bert_encoder.bert_layers.9.bert_attention.self.multi_head_attention.q_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.9.bert_attention.self.multi_head_attention.q_proj.bias          torch.Size([768])
bert_encoder.bert_layers.9.bert_attention.self.multi_head_attention.k_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.9.bert_attention.self.multi_head_attention.k_proj.bias          torch.Size([768])
bert_encoder.bert_layers.9.bert_attention.self.multi_head_attention.v_proj.weight        torch.Size([768, 768])
bert_encoder.bert_layers.9.bert_attention.self.multi_head_attention.v_proj.bias          torch.Size([768])
bert_encoder.bert_layers.9.bert_attention.self.multi_head_attention.out_proj.weight      torch.Size([768, 768])
bert_encoder.bert_layers.9.bert_attention.self.multi_head_attention.out_proj.bias        torch.Size([768])
bert_encoder.bert_layers.9.bert_attention.output.LayerNorm.weight        torch.Size([768])
bert_encoder.bert_layers.9.bert_attention.output.LayerNorm.bias          torch.Size([768])
bert_encoder.bert_layers.9.bert_intermediate.dense.weight        torch.Size([3072, 768])
bert_encoder.bert_layers.9.bert_intermediate.dense.bias          torch.Size([3072])
bert_encoder.bert_layers.9.bert_output.dense.weight      torch.Size([768, 3072])
bert_encoder.bert_layers.9.bert_output.dense.bias        torch.Size([768])
bert_encoder.bert_layers.9.bert_output.LayerNorm.weight          torch.Size([768])
bert_encoder.bert_layers.9.bert_output.LayerNorm.bias    torch.Size([768])
bert_encoder.bert_layers.10.bert_attention.self.multi_head_attention.q_proj.weight       torch.Size([768, 768])
bert_encoder.bert_layers.10.bert_attention.self.multi_head_attention.q_proj.bias         torch.Size([768])
bert_encoder.bert_layers.10.bert_attention.self.multi_head_attention.k_proj.weight       torch.Size([768, 768])
bert_encoder.bert_layers.10.bert_attention.self.multi_head_attention.k_proj.bias         torch.Size([768])
bert_encoder.bert_layers.10.bert_attention.self.multi_head_attention.v_proj.weight       torch.Size([768, 768])
bert_encoder.bert_layers.10.bert_attention.self.multi_head_attention.v_proj.bias         torch.Size([768])
bert_encoder.bert_layers.10.bert_attention.self.multi_head_attention.out_proj.weight     torch.Size([768, 768])
bert_encoder.bert_layers.10.bert_attention.self.multi_head_attention.out_proj.bias       torch.Size([768])
bert_encoder.bert_layers.10.bert_attention.output.LayerNorm.weight       torch.Size([768])
bert_encoder.bert_layers.10.bert_attention.output.LayerNorm.bias         torch.Size([768])
bert_encoder.bert_layers.10.bert_intermediate.dense.weight       torch.Size([3072, 768])
bert_encoder.bert_layers.10.bert_intermediate.dense.bias         torch.Size([3072])
bert_encoder.bert_layers.10.bert_output.dense.weight     torch.Size([768, 3072])
bert_encoder.bert_layers.10.bert_output.dense.bias       torch.Size([768])
bert_encoder.bert_layers.10.bert_output.LayerNorm.weight         torch.Size([768])
bert_encoder.bert_layers.10.bert_output.LayerNorm.bias   torch.Size([768])
bert_encoder.bert_layers.11.bert_attention.self.multi_head_attention.q_proj.weight       torch.Size([768, 768])
bert_encoder.bert_layers.11.bert_attention.self.multi_head_attention.q_proj.bias         torch.Size([768])
bert_encoder.bert_layers.11.bert_attention.self.multi_head_attention.k_proj.weight       torch.Size([768, 768])
bert_encoder.bert_layers.11.bert_attention.self.multi_head_attention.k_proj.bias         torch.Size([768])
bert_encoder.bert_layers.11.bert_attention.self.multi_head_attention.v_proj.weight       torch.Size([768, 768])
bert_encoder.bert_layers.11.bert_attention.self.multi_head_attention.v_proj.bias         torch.Size([768])
bert_encoder.bert_layers.11.bert_attention.self.multi_head_attention.out_proj.weight     torch.Size([768, 768])
bert_encoder.bert_layers.11.bert_attention.self.multi_head_attention.out_proj.bias       torch.Size([768])
bert_encoder.bert_layers.11.bert_attention.output.LayerNorm.weight       torch.Size([768])
bert_encoder.bert_layers.11.bert_attention.output.LayerNorm.bias         torch.Size([768])
bert_encoder.bert_layers.11.bert_intermediate.dense.weight       torch.Size([3072, 768])
bert_encoder.bert_layers.11.bert_intermediate.dense.bias         torch.Size([3072])
bert_encoder.bert_layers.11.bert_output.dense.weight     torch.Size([768, 3072])
bert_encoder.bert_layers.11.bert_output.dense.bias       torch.Size([768])
bert_encoder.bert_layers.11.bert_output.LayerNorm.weight         torch.Size([768])
bert_encoder.bert_layers.11.bert_output.LayerNorm.bias   torch.Size([768])
bert_pooler.dense.weight         torch.Size([768, 768])
bert_pooler.dense.bias   torch.Size([768])
```

#### 将预训练模型参数赋值给自己的模型参数

>观察上面的结果，发现参数只有这部分的是和bert-base-chinese中有区别
>
>### 参数:cls.predictions.bias,形状torch.Size([21128])
>### 参数:cls.predictions.transform.dense.weight,形状torch.Size([768, 768])
>### 参数:cls.predictions.transform.dense.bias,形状torch.Size([768])
>### 参数:cls.predictions.transform.LayerNorm.gamma,形状torch.Size([768])
>### 参数:cls.predictions.transform.LayerNorm.beta,形状torch.Size([768])
>### 参数:cls.predictions.decoder.weight,形状torch.Size([21128, 768])
>### 参数:cls.seq_relationship.weight,形状torch.Size([2, 768])
>### 参数:cls.seq_relationship.bias,形状torch.Size([2])
>
>那么就需要将bert-base-chinese赋给自己的Bert模型

```python
    def from_pretrained(cls, config, pretrained_model_dir=None):
        model = cls(config)  # 初始化模型，cls为未实例化的对象，即一个未实例化的BertModel对象
        # 这个是预训练模型，
        pretrained_model_path = os.path.join(pretrained_model_dir, "pytorch_model.bin")
        if not os.path.exists(pretrained_model_path):
            raise ValueError(f"<路径：{pretrained_model_path} 中的模型不存在，请仔细检查！>\n"
                             f"中文模型下载地址：https://huggingface.co/bert-base-chinese/tree/main\n"
                             f"英文模型下载地址：https://huggingface.co/bert-base-uncased/tree/main\n")
        # 保存模型的方法,这个需要重点关注，是后期调试模型的重点
        loaded_paras = torch.load(pretrained_model_path)
        # 拷贝一份BertModel中的网络参数,无法修改里面的值
        state_dict = deepcopy(model.state_dict())
        # 因为bert-base-chinese中的参数是207
        # 且发现BertModel中的参数是200
        # 那么需要将bert-base-chinse中的参数赋值到state_dict
        # loaded_paras_names是bert-base-chinese中的参数
        loaded_paras_names = list(loaded_paras.keys())[:-8]
        # model_paras_names是BertModel中参数
        model_paras_names = list(state_dict.keys())[1:]
        if 'use_torch_multi_head' in config.__dict__ and config.use_torch_multi_head:
            torch_paras = format_paras_for_torch(loaded_paras_names, loaded_paras)
            for i in range(len(model_paras_names)):
                logging.debug(f"## 成功赋值参数:{model_paras_names[i]},形状为: {torch_paras[i].size()}")
                if "position_embeddings" in model_paras_names[i]:
                    # 这部分代码用来消除预训练模型只能输入小于512个字符的限制
                    if config.max_position_embeddings > 512:
                        new_embedding = replace_512_position(state_dict[model_paras_names[i]],
                                                             loaded_paras[loaded_paras_names[i]])
                        state_dict[model_paras_names[i]] = new_embedding
                        continue
                # 那么需要将bert-base-chinse中的参数赋值到state_dict
                state_dict[model_paras_names[i]] = torch_paras[i]
            logging.info(f"## 注意，正在使用torch框架中的MultiHeadAttention实现")
        else:
            for i in range(len(loaded_paras_names)):
                logging.debug(f"## 成功将参数:{loaded_paras_names[i]}赋值给{model_paras_names[i]},"
                              f"参数形状为:{state_dict[model_paras_names[i]].size()}")
                if "position_embeddings" in model_paras_names[i]:
                    # 这部分代码用来消除预训练模型只能输入小于512个字符的限制
                    if config.max_position_embeddings > 512:
                        new_embedding = replace_512_position(state_dict[model_paras_names[i]],
                                                             loaded_paras[loaded_paras_names[i]])
                        state_dict[model_paras_names[i]] = new_embedding
                        continue
                # 这个代码很重要，是将bert-base-chinese中的参数赋给自己的BERT模型中
                state_dict[model_paras_names[i]] = loaded_paras[loaded_paras_names[i]]
            logging.info(f"## 注意，正在使用本地MyTransformer中的MyMultiHeadAttention实现，"
                         f"如需使用torch框架中的MultiHeadAttention模块可通过config.__dict__['use_torch_multi_head'] = True实现")
        model.load_state_dict(state_dict)
        return model
```

>这个代码代表将state_dict[model_paras_names[i]]=loaded_paras[loaded_paras_names[i]]，也就是将

- 假如引入预训练模型进行训练

```python
import sys
sys.path.append('../')
import torch
import os
from Tasks.TaskForSingleSentenceClassification import ModelConfig
from model.BasicBert.BertConfig import BertConfig
from model.BasicBert.Bert import BertModel

if __name__ == '__main__':
    model_config = ModelConfig()
    # # /media/wislab/Dataset_SSD2T/lzh/BertWithPretrained/bert_base_chinese/pytorch_model.bin
    # bin_path = os.path.join(model_config.pretrained_model_dir,'pytorch_model.bin')

    # loaded_paras = torch.load(bin_path)
    # # print(type(loaded_paras))
    # # print(len(list(loaded_paras)))
    # # print(list(loaded_paras.keys()))
    # print(len(list(loaded_paras)))
    # for name in loaded_paras.keys():
    #     print(f"### 参数:{name},形状{loaded_paras[name].size()}")
    # # ==========测试本模型Config中与bert-base-chinese中的区别==========
    # # 看bert_model 与BertModel的区别
    josn_file = os.path.join(model_config.pretrained_model_dir,'config.json')
    config = BertConfig.from_json_file(josn_file)
    # bert_model = BertModel(config=config)
    # print(len(bert_model.state_dict()))
    # for param_tensor in bert_model.state_dict():
    #     print(param_tensor,"\t",bert_model.state_dict()[param_tensor].size())
    # 这个是初值的赋予
    # from_pretrained是类似反射机制，就是不需要通过实例化类直接可以初始化
    bert = BertModel.from_pretrained(config=config,pretrained_model_dir=model_config.pretrained_model_dir)
    # 假如我冻结某些层的参数不参与模型训练
    # for para in bert.parameters():
    
```

## 文本分类

### 前向传播

>介绍如何分析和载入本地BERT预训练模型后, 接下来我们首先要做的就是实现文本分类的前向传播过程
>
>定义一个类来完成整个前向传播过程

```python
from ..BasicBert.Bert import BertModel
import torch.nn as nn

"""
定义一个类进行实现文本的前向传播
bert_pretrained_model_dir = /home/wislab/lzh/BertWithPretrained/bert_base_chinese
"""
class BertForSentenceClassification(nn.Module):
    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertForSentenceClassification, self).__init__()
        self.num_labels = config.num_labels
        # 通过BertModel进行加载，返回一个model
        # self.bert是一个model
        # 如果bert_pretrained_model_dir是存在,那么需要导入预训练好的pytorch_model.bin文件
        # 如果不存在,则重新利用自己的config训练文件
        # 这句话的意思就是是否存在预训练模型,如果存在,那么我们可以将其引入
        # 用BertModel.from_pretrained进行引入
        if bert_pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(config, bert_pretrained_model_dir)
        else:
            self.bert = BertModel(config)
        # "hidden_dropout_prob": 0.1,
        # 预训练好的模型的hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
```

>`如果存在bert_pretrained_model_dir`也就是`bert_pretrained_model中存在bin模型，那么就需要导入该模型进行修改本模型的参数`



```python
    def forward(self, input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                labels=None):
        """

        :param input_ids:  [src_len, batch_size]
        :param attention_mask: [batch_size, src_len]
        :param token_type_ids: 句子分类时为None
        :param position_ids: [1,src_len]
        :param labels: [batch_size,]
        :return:
        """
        # return pooled_output, all_encoder_outputs
        # pooled_output 是pooled_output
        # _是all_encoder_outputs
        pooled_output, _ = self.bert(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids)  # [batch_size,hidden_size]
        pooled_output = self.dropout(pooled_output)
        # 因为是分类问题,我们是直接在BERT之后添加一个分类器
        # classifier是一个分类器,基于Transformer图6.1结构图可知
        # 文本分类是加一个分类器进去的
        logits = self.classifier(pooled_output)  # [batch_size, num_label]
        if labels is not None:
            # 这句话的意思就是如果没定义label
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits

```

>```python
>        pooled_output, _ = self.bert(input_ids=input_ids,
>                                     attention_mask=attention_mask,
>                                     token_type_ids=token_type_ids,
>                                     position_ids=position_ids)  # [batch_size,hidden_size]
>```
>
>这部分是原始的BERT网络的输出，其中pooled_output为BERT第一个位置的向量经过一个全连接层后的结果；
>
>第2个参数是BERT中所有位置的向量，
>
>

>```python
>        pooled_output = self.dropout(pooled_output)
>        # 因为是分类问题,我们是直接在BERT之后添加一个分类器
>        # classifier是一个分类器,基于Transformer图6.1结构图可知
>        # 文本分类是加一个分类器进去的
>        logits = self.classifier(pooled_output)  # [batch_size, num_label]
>```
>
>pooled_out是用来进行文本分类的分类层

>```python
>        if labels is not None:
>            # 这句话的意思就是如果没定义label
>            loss_fct = nn.CrossEntropyLoss()
>            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
>            return loss, logits
>        else:
>            return logits
>```
>
>假如labels是空的，那么返回loss,或者返回logits

### 模型训练

>TaskForSingleSentenceClassification.py模块来完成模型的微调训练任务

```python
class ModelConfig:
    """
    ModelConfig是进行下游训练初始化参数的类
    project_dir是本项目存在的根目录.例如/home/wislab/lzh/BertWithPretrained
    dataset_dir是/home/wislab/lzh/BertWithPretrained/data/SingleSentenceClassification
    pretrained_model_dir = /home/wislab/lzh/BertWithPretrained/bert_base_chinese
    vocab_path = /home/wislab/lzh/BertWithPretrained/bert_base_chinese/vocab.txt
    train_file_path = /home/wislab/lzh/BertWithPretrained/data/SingleSentenceClassification/toutiao_train.txt
    val_file_path = /home/wislab/lzh/BertWithPretrained/data/SingleSentenceClassification/toutiao_val.txt
    test_file_path = /home/wislab/lzh/BertWithPretrained/data/SingleSentenceClassification/toutiao_test.txt
    model_save_dir = /home/wislab/lzh/BertWithPretrained/cache
    logs_save_dir = /home/wislab/lzh/BertWithPretrained/logs
    bert_config_path = /home/wislab/lzh/BertWithPretrained/bert_base_chinese/config.json
    """
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data', 'SingleSentenceClassification')
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_chinese")
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        # torch.device('cuda:0' if torch.cuda.is_available())
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.train_file_path = os.path.join(self.dataset_dir, 'toutiao_train.txt')
        self.val_file_path = os.path.join(self.dataset_dir, 'toutiao_val.txt')
        self.test_file_path = os.path.join(self.dataset_dir, 'toutiao_test.txt')
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.split_sep = '_!_'
        self.is_sample_shuffle = True
        self.batch_size = 64
        self.max_sen_len = None
        self.num_labels = 15
        self.epochs = 10
        self.model_val_per_epoch = 2
        # logger_init是日志函数用来打印日志的
        # 保存为/home/wislab/lzh/BertWithPretrained/logs/single.txt
        logger_init(log_file_name='single', log_level=logging.INFO,
                    log_dir=self.logs_save_dir)
        # 如果os.path里面不存在model_save_dir文件夹
        # 那么创造model_save_dir文件夹
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        # 把原始bert中的配置参数也导入进来
        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        # 从bert_config.__dict__里面取出items()
        # items()是key:value形式也就是键值形式
        # 那么我们可以将其放进自己的__dict__中
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        # 将当前配置打印到日志文件中
        logging.info(" ### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"###  {key} = {value}")

```

>初始化代码:这个类代码是用来初始化各类参数的
>
>设置打印任务用的



```python
"""
train过程,导入config对象
"""
def train(config):
    # 引用model/DownstreamTasks中模块
    # Bert是ForSentenceClassification生成的一个model
    # 初始化之后会有model
    model = BertForSentenceClassification(config,
                                          config.pretrained_model_dir)
    # 打印看看model到底是什么
    print(model)
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行追加训练......")
    # 将该模型加入到设备里面,可以使得模型转移到指定的设备里面运行
    model = model.to(config.device)
    # .Variable的基本概念, .Variable的自动求导
    # variable类型
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    # 进行训练
    model.train()
    # Transformer里面的BertTokenizer类
    # 得仔细研读一下
    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    # 加载数据集合,LoadSingleSentenceClassificationDataset已经写好再data_helper.py里面了
    data_loader = LoadSingleSentenceClassificationDataset(vocab_path=config.vocab_path,
                                                          tokenizer=bert_tokenize,
                                                          batch_size=config.batch_size,
                                                          max_sen_len=config.max_sen_len,
                                                          split_sep=config.split_sep,
                                                          max_position_embeddings=config.max_position_embeddings,
                                                          pad_index=config.pad_token_id,
                                                          is_sample_shuffle=config.is_sample_shuffle)
    # data_loader的工作原理,得仔细研读一下
    train_iter, test_iter, val_iter = data_loader.load_train_val_test_data(config.train_file_path,
                                                                           config.val_file_path,
                                                                           config.test_file_path)
    max_acc = 0
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (sample, label) in enumerate(train_iter):
            sample = sample.to(config.device)  # [src_len, batch_size]
            label = label.to(config.device)
            padding_mask = (sample == data_loader.PAD_IDX).transpose(0, 1)
            loss, logits = model(
                input_ids=sample,
                attention_mask=padding_mask,
                token_type_ids=None,
                position_ids=None,
                labels=label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            acc = (logits.argmax(1) == label).float().mean()
            if idx % 10 == 0:
                logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                             f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")
        end_time = time.time()
        train_loss = losses / len(train_iter)
        logging.info(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        if (epoch + 1) % config.model_val_per_epoch == 0:
            acc = evaluate(val_iter, model, config.device, data_loader.PAD_IDX)
            logging.info(f"Accuracy on val {acc:.3f}")
            if acc > max_acc:
                max_acc = acc
                torch.save(model.state_dict(), model_save_path)

```

>- 初始化一个基于BERT的分类模型
>
>  ```python
>      # 引用model/DownstreamTasks中模块
>      # Bert是ForSentenceClassification生成的一个model
>      # 初始化之后会有model
>      model = BertForSentenceClassification(config,
>                                            config.pretrained_model_dir)
>      # 打印看看model到底是什么
>      print(model)
>      model_save_path = os.path.join(config.model_save_dir, 'model.pt')
>      if os.path.exists(model_save_path):
>          loaded_paras = torch.load(model_save_path)
>          model.load_state_dict(loaded_paras)
>          logging.info("## 成功载入已有模型，进行追加训练......")
>      # 将该模型加入到设备里面,可以使得模型转移到指定的设备里面运行
>      model = model.to(config.device)
>      # .Variable的基本概念, .Variable的自动求导
>      # variable类型
>      optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
>      # 进行训练
>      model.train()
>  ```
>
>- 载入相应的数据集
>
>```python
>    data_loader = LoadSingleSentenceClassificationDataset(vocab_path=config.vocab_path,
>                                                          tokenizer=bert_tokenize,
>                                                          batch_size=config.batch_size,
>                                                          max_sen_len=config.max_sen_len,
>                                                          split_sep=config.split_sep,
>                                                          max_position_embeddings=config.max_position_embeddings,
>                                                          pad_index=config.pad_token_id,
>                                                          is_sample_shuffle=config.is_sample_shuffle)
>    # data_loader的工作原理,得仔细研读一下
>    train_iter, test_iter, val_iter = data_loader.load_train_val_test_data(config.train_file_path,
>                                                                       config.val_file_path,
>                                                                           config.test_file_path)
>```
>
>- 进行训练
>
>```python
>    max_acc = 0
>    for epoch in range(config.epochs):
>        losses = 0
>        start_time = time.time()
>        for idx, (sample, label) in enumerate(train_iter):
>            sample = sample.to(config.device)  # [src_len, batch_size]
>            label = label.to(config.device)
>            padding_mask = (sample == data_loader.PAD_IDX).transpose(0, 1)
>            loss, logits = model(
>                input_ids=sample,
>                attention_mask=padding_mask,
>                token_type_ids=None,
>                position_ids=None,
>                labels=label)
>            optimizer.zero_grad()
>            loss.backward()
>            optimizer.step()
>            losses += loss.item()
>            acc = (logits.argmax(1) == label).float().mean()
>            if idx % 10 == 0:
>                logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
>                             f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")
>        end_time = time.time()
>        train_loss = losses / len(train_iter)
>        logging.info(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
>        if (epoch + 1) % config.model_val_per_epoch == 0:
>            acc = evaluate(val_iter, model, config.device, data_loader.PAD_IDX)
>            logging.info(f"Accuracy on val {acc:.3f}")
>            if acc > max_acc:
>                max_acc = acc
>                torch.save(model.state_dict(), model_save_path)
>```
>
>

# 下游任务二: 文本蕴含任务

## 任务构造原理

>`同时给模型输入两句话，然后让模型来判断两句话之间的关系，所以本质上也就变成了一个文本分类任务`
>
>最终都是对一个文本序列进行分类。只是按照 BERT 模型的思想，`文本对分类任务在数据集的构建过程中需要通过 Segment Embedding来区分前后两个不同的序列，并且两个句子之间需要通过一个[SEP]符号来进行分
>割，因此本节内容的核心就在于如何构建数据集`
>
>`文本对的分类任务除了在模型输入上发生了变换，其它地方均与单文本分类任务一样，同样也是取最后一层的[CLS]向量进行分类。接下来掌柜首先就来介绍如何构造文本分类的数据集`

## 数据预处理

### 输入数据

>不仅仅需要进行Token Embedding操作，同时还要对`两个序列进行Segment Embedding 操作`。对于 Position Embedding 来说在任何场景下都不需要对其指定输入，因为我们在代码实现时已经做了相应默认时的处理
>
>- 需要构造原始文本对应的Token序列,然后在最前面加上一个[CLS]符,两个序列之间以及整个序列的末尾分别再加上一个[SEP]符号
>- 根据两个序列各自的长度再构建一个类似[0,0,0,...,1,1,1,...]的token_type_ids向量
>- 最后将两者作为模型的输入即可

### 数据集分析

![](https://pic.superbed.cc/item/66f5425e991d0115dfc8b3aa.png)

>①原始数据样本进行分词(tokenize)处理 -> ②根据tokenize后结果构建字典 -> ③将tokenize后的文本序列转换为Token id序列,同时在Token id序列起始位置加上[CLS],两个序列之间以及整个序列末尾加上[SEP]符号 ，并进行padding->④ 根据第三步分别生成对应的token types ids 和attention mask向量

## 数据集构建

### ① 定义tokenize

>和transformers模型类似，需要先对文本进行tokenizer操作

```python
# Tokenizer的操作,底层源码不用管
import sys
sys.path.append('../')
from Tasks.TaskForPairSentenceClassification import ModelConfig
from transformers import BertTokenizer

if __name__ == '__main__':
    model_config = ModelConfig()
    tokenizer = BertTokenizer.from_pretrained(model_config.pretrained_model_dir).tokenize
    # print(tokenizer("青山不改，绿水长流，我们月来客栈见！"))
    # print(tokenizer("10 年前的今天，纪念 5.12 汶川大地震 10 周年"))
    print(tokenizer("From Home Work to Modern Manufacture. Modern manufacturing has changed over time."))
```

### ②建立词表

> 因为谷歌是开源了 vocab.txt 词表,  不需要根据自己的语料来建立一个词表
>
> `不能根据自己的语料来构建词表`,因为当用自己的语料构建词表的时候，会导致后期提取的token.id是错误的

> 该文件在"/media/wislab/Dataset_SSD2T/lzh/BertWithPretrained/utils/data_helpers.py"

```python
# 测试载入数据
import sys

sys.path.append('../')
from Tasks.TaskForPairSentenceClassification import ModelConfig
from utils.data_helpers import LoadPairSentenceClassificationDataset
from transformers import BertTokenizer

vocab_path = ""

class Vocab:
    """
    根据本地的vocab文件，构造一个词表
    vocab = Vocab()
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['我'])  # 通过单词返回得到词表中对应的索引
    print(len(vocab))  # 返回词表长度
    """
    UNK = '[UNK]'

    def __init__(self, vocab_path):
        self.stoi = {}
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，
            # 同时列出数据和数据下标，一般用在 for 循环当中。
            for i, word in enumerate(f):
                # 去除头尾\n
                w = word.strip('\n')
                self.stoi[w] = i
                self.itos.append(w)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)


def build_vocab(vocab_path):
    """
    vocab = Vocab()
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['我'])  # 通过单词返回得到词表中对应的索引
    """
    return Vocab(vocab_path)

if __name__ == '__main__':
    model_config = ModelConfig()
    vocab = build_vocab(model_config.vocab_path)
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    # print(vocab.stoi['我'])  # 通过单词返回得到词表中对应的索引

```

>输出结果

```python
, 'atkins': 21087, 'turrets': 21088, 'inadvertently': 21089, 'disagree': 21090, 'libre': 21091, 'vodka': 21092, 'reassuring': 21093, 'weighs': 21094, '##yal': 21095, 'glide': 21096, 'jumper': 21097, 'ceilings': 21098, 'repertory': 21099, 'outs': 21100, 'stain': 21101, '##bial': 21102, 'envy': 21103, '##ucible': 21104, 'smashing': 21105, 'heightened': 21106, 'policing': 21107, 'hyun': 21108, 'mixes': 21109, 'lai': 21110, 'prima': 21111, '##ples': 21112, 'celeste': 21113, '##bina': 21114, 'lucrative': 21115, 'intervened': 21116, 'kc': 21117, 'manually': 21118, '##rned': 21119, 'stature': 21120, 'staffed': 21121, 'bun': 21122, 'bastards': 21123, 'nairobi': 21124, 'priced': 21125, '##auer': 21126, 'thatcher': 21127, '##kia': 21128, 'tripped': 21129, 'comune': 21130, '##ogan': 21131, '##pled': 21132, 'brasil': 21133, 'incentives': 21134, 'emanuel': 21135, 'hereford': 21136, 'musica': 21137, '##kim': 21138, 'benedictine': 21139, 'biennale': 21140, '##lani': 21141, 'eureka': 21142, 'gardiner': 21143, 'rb': 21144, 'knocks': 21145, 'sha': 21146, '##ael': 21147, '##elled': 21148, '##onate': 21149, 'efficacy': 21150, 'ventura': 21151, 'masonic': 21152, 'sanford': 21153, 'maize': 21154, 'leverage': 21155, '##feit': 21156, 'capacities': 21157, 'santana': 21158, '##aur': 21159, 'novelty': 21160, 'vanilla': 21161, '##cter': 21162, '##tour': 21163, 'benin': 21164, '##oir': 21165, '##rain': 21166,....
```

### ③转换为Token序列

```python
class LoadPairSentenceClassificationDataset(LoadSingleSentenceClassificationDataset):
    def __init__(self, **kwargs):
        super(LoadPairSentenceClassificationDataset, self).__init__(**kwargs)
        pass
```

>按照前一篇的方法,这里继承了LoadSingleSentenceClassificationDataset的方法
>
>重写data_process()和generate_batch()方法



```python
     # data_process()相当于调用了
    # data_process = process_cache(unique_key=["max_sen_len"])(data_process)
    # 相当于首先执行了process_cache(unique_key=["max_sen_len"]),返回的是decorating_function函数
    # 再调用返回的函数,参数是data_process(self, file_path=None),返回值是wrapper函数
    @process_cache(unique_key=["max_sen_len"])
    def data_process(self, file_path=None):
        """
        将每一句话中的每一个词根据字典转换成索引的形式，同时返回所有样本中最长样本的长度
        :param filepath: 数据集路径
        :return:
        """
        raw_iter = open(file_path).readlines()
        data = []
        max_len = 0
        # tqdm是python的进度条库，可以在python长循环中添加一个进度提示信息
        for raw in tqdm(raw_iter, ncols=80):
            # 取得文本和标签;
            line = raw.rstrip("\n").split(self.split_sep)
            s1, s2, l = line[0], line[1], line[2]
            # 分别对两个序列s1和s2转换为词表中对应的Token
            token1 = [self.vocab[token] for token in self.tokenizer(s1)]
            token2 = [self.vocab[token] for token in self.tokenizer(s2)]
            # 将两个序列拼接起来，并在序列的开始加上[CLS]符号
            # 在两个序列之间及末尾加上[SEP]符号
            tmp = [self.CLS_IDX] + token1 + [self.SEP_IDX] + token2
            if len(tmp) > self.max_position_embeddings - 1:
                tmp = tmp[:self.max_position_embeddings - 1]  # BERT预训练模型只取前512个字符
            tmp += [self.SEP_IDX]
            # 构造Segment Embedding的输入向量
            seg1 = [0] * (len(token1) + 2)  # 2 表示[CLS]和中间的[SEP]这两个字符
            seg2 = [1] * (len(tmp) - len(seg1))
            segs = torch.tensor(seg1 + seg2, dtype=torch.long)
            # 整合得到对应的样本数据
            tensor_ = torch.tensor(tmp, dtype=torch.long)
            l = torch.tensor(int(l), dtype=torch.long)
            max_len = max(max_len, tensor_.size(0))
            data.append((tensor_, segs, l))
        return data, max_len
```



```
[2024-09-26 20:20:54] - INFO:  ## 索引预处理缓存文件的参数为：['max_sen_len']
[2024-09-26 20:20:54] - INFO: 缓存文件 /media/wislab/Dataset_SSD2T/lzh/BertWithPretrained/data/PairSentenceClassification/cache_test_max_sen_lenNone.pt 不存在，重新处理并缓存！
100%|███████████████████████████████████| 39271/39271 [00:18<00:00, 2106.22it/s]
[2024-09-26 20:21:15] - INFO: 数据预处理一共耗时21.049s
[2024-09-26 20:21:15] - INFO:  ## 索引预处理缓存文件的参数为：['max_sen_len']
[2024-09-26 20:21:15] - INFO: 缓存文件 /media/wislab/Dataset_SSD2T/lzh/BertWithPretrained/data/PairSentenceClassification/cache_train_max_sen_lenNone.pt 不存在，重新处理并缓存！
100%|█████████████████████████████████| 274891/274891 [02:11<00:00, 2097.04it/s]
[2024-09-26 20:23:43] - INFO: 数据预处理一共耗时147.958s
[2024-09-26 20:23:43] - INFO:  ## 索引预处理缓存文件的参数为：['max_sen_len']
[2024-09-26 20:23:43] - INFO: 缓存文件 /media/wislab/Dataset_SSD2T/lzh/BertWithPretrained/data/PairSentenceClassification/cache_val_max_sen_lenNone.pt 不存在，重新处理并缓存！
100%|███████████████████████████████████| 78540/78540 [00:37<00:00, 2121.24it/s]
[2024-09-26 20:24:25] - INFO: 数据预处理一共耗时42.038s
torch.Size([70, 16])
tensor([[ 101, 1998, 2061,  ...,    0,    0,    0],
        [ 101, 1998, 2116,  ...,    0,    0,    0],
        [ 101, 1998, 1996,  ...,    0,    0,    0],
        ...,
        [ 101, 1999, 2621,  ...,    0,    0,    0],
        [ 101, 2035, 1997,  ...,    0,    0,    0],
        [ 101, 2273, 2293,  ...,    0,    0,    0]])
torch.Size([16, 70])
torch.Size([16])
torch.Size([70, 16])
tensor([1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 0, 2, 1, 0, 2])
tensor([[0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        ...,
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0]])
```

>`注意:`这个是应该要事先判断cache_train_max_sen_lenNone.pt是否存在
>
>`这部分内容可以看Pyton语法基础的md笔记，其中在python装饰器那一章`
>
>看输出结果101就是[CLS]在词表中的索引位置,102 则是[SEP]在词表中的索引；其它非 0 值就是 tokenize 后的文本序列转换成的 Token序列
>
>

### ④ padding处理与mask

