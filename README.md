## GCN-BSC & GraphSAGE-BPR by PyTorch

Building patterns are important components of urban structures and functions, and their accurate recognition is the foundation of urban spatial analysis, cartographic generalization, and other
tasks. Current building pattern recognition methods are often based on a shape index that can only characterize shape features from one aspect, resulting in significant errors. In this study, a building pattern recognition method based on a graph neural network is proposed to enhance shape cognition
and focus on recognizing collinear patterns. First, a building shape classification model that integrates global shape and graph node structure features was constructed to quantitatively study shape cognition. Subsequently, a collinear pattern recognition (CPR) model was established based on a dual building graph. The shape cognition results were integrated into the model to enhance its recognition ability. The results show that the shape classification model can be used to effectively distinguish different shape categories and support building pattern recognition tasks. Based on the CPR model, false recognitions can be avoided, and recognition results similar to those of visual cognition can be obtained. Compared with the comparative methods, both models have significant advantages in terms of statistical results and implementation.



### Requirements

```
python            3.7.13
scikit-learn      1.0.2
torch             1.12.0+cu116
torch-geometric   2.3.1
numpy             1.21.6+mkl
pandas            1.3.5
matplotlib        3.5.3
shapely           2.0.1
geopandas         0.9.0
```

### Reference

If you used the code, please cite this paper.

```
Zhang, F., Sun, Q., Huang, W., Su, Y., Ma, J., & Xing, R. (2024). Enhancing the Recognition of Collinear Building Patterns by Shape Cognition Based on Graph Neural Networks. Applied Artificial Intelligence, 38(1). https://doi.org/10.1080/08839514.2024.2439611
```

