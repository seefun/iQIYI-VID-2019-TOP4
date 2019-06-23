# 2019 iQIYI Celebrity Video Identification Challenge

## Doing
- [x] 算法说明文档
- [x] 整理比赛训练代码测试代码
- [x] 答辩PPT和准备答辩

## 竞赛简介
比赛网址： http://challenge.ai.iqiyi.com/detail?raceId=5c767dc41a6fa0ccf53922e7

数据集参考[论文](https://arxiv.org/pdf/1811.07548.pdf)

比赛提供了约6万的训练视频和大约等量的验证集视频数据，并对视频中一些帧提好了512维的人脸、人头、身体特征，整个视频提了一个512维的声音特征。iQIYI-VID-2019挑战赛的目标如下:对于训练集中的每个身份，在测试集中找到相同身份的所有视频片段。检索结果列表应根据每个测试片段和查询之间的相似性排序。共10034个不同人物，即要找到包含对应人物的视频序列，同时测试集中含有不属于这10034类的视频。

## 算法说明文档
本算法只使用了官方提供的人脸特征这单一模态数据，在排行榜取得了TOP4，具体文档见[PDF](./data/TOP4-solution-seefun.pdf)

## 训练测试代码
见 ./src

## 答辩PPT
答辩PPT见[PDF](./data/TOP4-seefun-PPT.pdf)


-------------------------------------------------


以下是过去的整理

## similar solution / papers
### 2018 iQIYI_VID比赛：
#### 第一名(Infinivision)解决方案：
1. 人脸检测对齐
2. ArcFace/InsightFace
3. 多模型early fusion的人脸特征+场景特征再fc融合

####  第二名(百度VAR)解决方案：
改进的loss 非监督数据清洗 结合人脸质量 不详

#### 第三名解决方案
不详

#### 第四名解决方案
[https://github.com/Jasonbaby/IQIYI_VID_FACE](https://github.com/Jasonbaby/IQIYI_VID_FACE)


#### 第五名解决方案
[https://github.com/luckycallor/IQIYI_VID_5th](https://github.com/luckycallor/IQIYI_VID_5th)

#### ？th解决方案
[https://github.com/deepinx/iqiyi-vid-challenge](https://github.com/deepinx/iqiyi-vid-challenge)

### related papers
[iQIYI-VID-datasset](https://arxiv.org/pdf/1811.07548.pdf)


### related codes
[insight face](https://github.com/deepinsight/insightface)

[insight face pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)


## EDA & stable CV
EDA得出，训练集视频总数57820，人脸1705010，每个视频人脸特征数目集中在100以下，平均出现29.5次。大多数label只在少于10个视频中出现（在2个视频中出现的最多），是个few-shot的问题。人脸共10034个label。人脸检测分基本都在0.8以上，绝大多数哦0.9以上。质量得分在-30以上，绝大多数在0以上，类似均值80+的正态分布。
此外： 验证集视频数目72798， 但gt中只出现了45472，只有62.5%的视频是出现在训练集中的，剩下的是噪声标签。这是一个open set的问题。
