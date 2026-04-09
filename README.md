Reproduced Weights
SELLM-Weights
https://pan.baidu.com/s/1ca0Azm3CyNFuJ6RzbaC9mw?pwd=x4j5 code: x4j5 

The TSCC module retains several layers that are not explicitly used in the forward pass. These layers originate from the original experimental code used in the paper. We observed that removing them changes parameter initialization, particularly affecting the VAE module, which leads to noticeable differences between reproduced results and those reported in the paper. To ensure reproducibility, these layers are preserved in the released implementation. These layers should be regarded as implementation artifacts rather than essential components of the proposed method. Importantly, their presence does not affect the relative trends observed in ablation studies, nor does it influence the effectiveness of the proposed approach in improving LLM performance on time-series forecasting tasks.
