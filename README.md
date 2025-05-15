# [TGRS 2022] Dual-Frequency Autoencoder for Anomaly Detection in Transformed Hyperspectral Imagery

## Abstract:
Hyperspectral anomaly detection (HAD) is a challenging task since samples are unavailable for training. Although unsupervised learning methods have been developed, they often train the model using an original hyperspectral image (HSI) and require retraining on different HSIs, which may limit the feasibility of HAD methods in practical applications. To tackle this problem, we propose a dual-frequency autoencoder (DFAE) detection model in which the original HSI is transformed into high-frequency components (HFCs) and low-frequency components (LFCs) before detection. A novel spectral rectification is first proposed to alleviate the spectral variation problem and generate the LFCs of HSI. Meanwhile, the HFCs are extracted by the Laplacian operator. Subsequently, the proposed DFAE model is learned to detect anomalies from the LFCs and HFCs in parallel. Finally, the learned model is well-generalized for anomaly detection from other hyperspectral datasets. While breaking the dilemma of limited generalization in the sample-free HAD task, the proposed DFAE can enhance the background–anomaly separability, providing a better performance gain. Experiments on real datasets demonstrate that the DFAE method exhibits competitive performance compared with other advanced HAD methods.


## DFAE Code Structure and Workflow:
The original HSI is first split into **low-frequency** and **high-frequency** components, and then fed separately into the networks.
Thus, the code can be roughly understood as consisting of **three main parts**:
1. **Preprocessing** (in MATLAB)
2. **Neural network processing** (in Python)
3. **Postprocessing** (in MATLAB)

## Detailed Code Steps

1. `./HSIs` contains the five original HSI images used in the paper.

2. Use the code in `./matlab/DFAE_low` to extract the first 188 bands from the original HSI images. The extracted images are stored in `./data_ori`.

3. `./matlab/DFAE_low` is used to extract the **low-frequency** part of the images. The resulting data is saved in `./data_low`.

4. `./matlab/DFAE_high` is used to extract the **high-frequency** part of the images. The resulting data is saved in `./data_high`.

5. `./DFAE_low_train` and `./DFAE_high_train` are used to train the networks on the low-frequency and high-frequency data, respectively. In the paper, the training set is `sandiego1`, which means these Python training scripts take `sandiego1` from `./data_low` and `./data_high` as input.

6. `./DFAE_low_test` and `./DFAE_high_test` are the test scripts. They take other datasets from `./data_low` and `./data_high` as input.

7. The output results of the Python code are saved in `./data_low_res` and `./data_high_res`.

8. `./matlab/DFAE_final_detection` performs the final detection. Relevant parameters (i.e., the area for attribute filtering) should be adjusted based on the size of anomalies. The final detection results are saved in `./data_res`.

## Citation
If your find our work are helpful for your research, please cite our paper.
```
@article{liu2022dual,
  title={Dual-frequency autoencoder for anomaly detection in transformed hyperspectral imagery},
  author={Liu, Yidan and Xie, Weiying and Li, Yunsong and Li, Zan and Du, Qian},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--13},
  year={2022},
  publisher={IEEE}
}
```
