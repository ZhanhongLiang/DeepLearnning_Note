# Bert 下游任务环境配置

>python 3.7
>
>torch==1.5.0
>torchtext==0.6.0
>torchvision==0.6.0
>transformers==4.5.1
>numpy==1.19.5
>pandas==1.1.5
>scikit-learn==0.24.0
>tqdm==4.61.0

## 创建bert虚拟虚拟环境

```
conda create -n bert python=3.8
```

>但是当我建立虚拟环境中建立python3.6时候会发现冲突

## 查看当前的虚拟环境

```
conda env list
```

## 进入虚拟环境

```
source activate bert
```

## 查看当前的安装的包

```
conda list
```

## 安装pytorch

>根据官网给定对应
>
>```
># CUDA 9.2
>conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=9.2 -c pytorch
>
># CUDA 10.1
>conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch
>
># CUDA 10.2
>conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
>
># CPU Only
>conda install pytorch==1.5.1 torchvision==0.6.1 cpuonly -c pytorch
>```

```
# CUDA 10.2
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorc
```

## 安装torchtext

```
conda install torchtext==0.6.0 -c pytorch
```

## 安装torchvision

```
pip install torchvision==0.6.0
```

## 安装pandas

```
conda install pandas==1.1.5
```

## 安装scikit-learn

```
pip install scikit-learn==0.24.0
```

## 安装transformers

```
pip install transformers==4.5.1
```

## 报错汇总

>如果报错，那么可以用pip进行下载,因为conda 存在下载不了的情况

## 查看显卡

```
nvidia-smi -L
```

## 查看显卡

```
nvidia-smi.exe -l 5
```



