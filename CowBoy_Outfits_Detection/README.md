## 配置环境
按照YOLOX官网首先安装YOLOX，然后安装apex和pycocotools
## 数据预处理
由于image_id太大，直接使用数据集将报错，因此需要将数据集中的image_id替换为序列号，并创建装备类别到序号1-5的映射
## 在预训练过的模型上进行训练
使用的是YOLOX_M的模型，自定义了exp文件来定义模型以及dataloader需要加载的数据路径，还有训练的epoch数量，以及自定义的dataloader
## 定义预测函数
