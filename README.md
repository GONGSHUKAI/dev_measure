# 基于卫星图像的人机协同经济发展水平测量

## 代码仓库解释

1. concat文件夹：
    - concat.py：使用pycuda编写核函数进行图像拼接(GPU加速)
    - concat_np.py：使用numpy进行图像拼接(CPU)
    - csv.split：将csv文件按照指定行数分割
    - sichuan_granules_scores_part1.csv：储存四川省各区块经济发展水平得分
    - sichuan_granules_scores_part2.csv：储存四川省各区块经济发展水平得分
    - img_output文件夹：60个区块的经济发展热力图
2. graph_config文件夹：
    - sichuan.txt：四川省的POG
3. Stage1文件夹：数据的爬取，分割，标注，以及一阶段模型的训练
    - crawler_stac_visual.py：使用Planetary ComputerAPI爬取卫星图像数据
    - vlm_labeling.py：使用大模型API对卫星图像进行标注
    - pic_seg_mpi.cpp：使用MPI加速卫星图像的分割
    - Dcluster.ipynb：使用DeepCluster对图像进行聚类
    - pretrain_classifier.py：训练一个ResNet对卫星图像进行城市、农村、山地、高原的四分类
4. Stage2文件夹：POG的生成与加聚
    - vlm_ranking.py：使用大模型API对卫星图像聚类进行排序，生成多个POG
    - pog_aggregation.py：对多个POG进行并行加聚
5. Stage3文件夹：经济发展水平得分模型的训练
    - backbone.py：卫星图像打分模型的定义，以及损失函数的定义
    - scoring.py：训练卫星图像打分模型的脚本
    - test.py：测试卫星图像打分模型的脚本
6. Evaluation文件夹：对经济发展水平热力图可行性的评估
    - night light comparison.ipynb：对经济发展水平热力图与夜光图算相关系数
7. utils文件夹：一些工具函数
    - graph.py：定义POG
    - parameters.py：定义一些超参数
    - siScore_utils.py：定义图像数据集的类