# Bio_GMM_VGAE
Integrating gaussian mixture model and variational graph autoencoder for gene imputation in spatial transcriptomics
集成高斯混合模型和变分图自编码器的空间转录组基因填充研究

------
相较单细胞测序技术而言，空间转录组在生成基因表达谱的同时，能保留细胞的空间位置信息。其不足是基因测序深度不够，且数据噪声较大。
本文提出集成高斯混合模型和变分图自编码器的空间转录组基因填充方法，以实现数据的基因表达谱还原、去噪。
首先利用高斯混合模型对空间转录组数据进行预聚类，形成预训练模型；随后利用变分图自编码器在预训练模型基础上进行低维表征推断和特征矩阵重构
，分别由变分图自编码器中的编码器和解码器来实现。对重构的特征矩阵进行高度可变基因提取，描绘其在基因填充前后的空间表达模式，以验证本文提出方法的准确性和稳定性。
在多个空间转录组学数据上进行验证，进而证明所提方法的泛化性和有效性。
## Workflow of spatial clustering task
![](https://github.com/narutoten520/Bio_GMM_VGAE/blob/b0f8ceea752be2c01063217a98abdd353bf39eea/%E5%9B%BE%E7%89%871.png)

## Contents
* [Prerequisites](https://github.com/narutoten520/Bio_GMM_VGAE/tree/main#prerequisites)
* [Example usage](https://github.com/narutoten520/Bio_GMM_VGAE/tree/main#example-usage)
* [Datasets Availability](https://github.com/narutoten520/Bio_GMM_VGAE/tree/main#datasets-availability)
* [License](https://github.com/narutoten520/Bio_GMM_VGAE/tree/main#license)
* [Trouble shooting](https://github.com/narutoten520/Bio_GMM_VGAE/tree/main#trouble-shooting)

### Prerequisites

1. Python (>=3.8)
2. Scanpy
3. Squidpy
4. Pytorch_pyG
5. Pandas
6. Numpy
7. Sklearn
8. Seaborn
9. Matplotlib
10. Torch_geometric

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Example usage
* Spatial clustering for human DLPFC data using VGAE_SGC method
  ```sh
    running gene_imputation.ipynb to choose idetify the spatial domains for human breast cancner data
  ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Datasets Availability

* [Human DLPFC](https://github.com/LieberInstitute/spatialLIBD)
* [Mouse brain](https://squidpy.readthedocs.io/en/stable/auto_tutorials/tutorial_visium_hne.html)
* [Slide-seqV2](https://squidpy.readthedocs.io/en/stable/auto_tutorials/tutorial_slideseqv2.html)
* [Stereo-seq](https://stagate.readthedocs.io/en/latest/T4_Stereo.html)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Trouble shooting

* data files<br>
Please down load the spatial transcriptomics data from the provided links.

* Porch_pyg<br>
Please follow the instruction to install pyG and geometric packages.
