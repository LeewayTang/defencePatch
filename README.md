# defencePatch
## 概述

### 0 来源

论文：[Adversarial Patch](https://arxiv.org/abs/1712.09665)

叙述了对抗补丁的生成思路，在VGG，inceptionV3，ResNet等分类模型上可以做到用贴补丁的方式定向误导模型识别结果。

### 1 迁移

产生对抗补丁的原理是迭代改变补丁的RGB值，使得识别结果靠近目标。

同样的思路可以用于防御：将图片置于对抗场景中。随机生成的patch随机贴在原图上，使得模型识别的效果提升。

