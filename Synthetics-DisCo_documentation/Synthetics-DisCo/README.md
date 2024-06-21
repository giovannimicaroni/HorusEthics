# Data Package for "Synthetic Face Datasets Generation via Latent Space Exploration from Brownian Identity Diffusion"

This data package contains a selection of datasets generated for the above-mentioned paper, as well as intermediate data required to reproduce the results. The three datasets present in this package are the three best performing datasets, presented in the first table of the paper. We do not release the full ensemble of synthetic datasets due to the volume of data involved and since they can easily be reproduced with the information present both here, in the code repository and in the paper. We are open of sharing the full data upon reasonable request.

## Package Content

This package contain:

1) Two sets of synthetic reference identities, 10k and 30k respectively, created with the *Langevin* algorithms: ```sg2_n10k_arc_r14_lang_v1``` and ```sg2_n30k_arc_r14_lang_v2```.

2) Three synthetic face datasets, each containing 64 variations of the synthetic reference identities. One dataset, ```sg2_n10k_arc_r14_disp_n64_rl12``` is created with the *Dispersion* algorithm from the ```sg2_n10k_arc_r14_lang_v1``` references. The two other datasets, ```sg2_n10k_arc_r14_disco_n64_rl14```and ```sg2_n30k_arc_r14_disco_n64_rl14``` are created with the *DisCo* algorithm from the ```sg2_n10k_arc_r14_lang_v1``` and ```sg2_n30k_arc_r14_lang_v2```references, respectively.

3) Face recognition models trained on these three datasets (`trained_models`). The face recognition models have iResNet50 backbone and are trained with the AdaFace loss function using our generated synthetic datasets. For each model, a checkpoint file (i.e., `.ckpt`) is provided. In addition, training log and hyper-parameters are also provided in `results` subfolder. 

4) SQLite database index files, containing samples metadata, for the *FFHQ* and *MultiPIE* genuine face databases. The data must be obtained separately.

5) Linear SVM trained on these projection providing latent directions, used for instance by the *DisCo* algorithm.

For each *Langevin*, *Dispersion* and *DisCo* ensembles, are provided:

* A sample collection HDF5 file ```samples.h5```containing the latent and embedding vectors for all synthetic samples. The code to read such a file is provided by the ```SampleCollection```class in the companion software package.

* One PNG image per sample, aligned according to the ArcFace 112x112 standard. Other alignment standards, as well as un-cropped images, can be re-generated from the sample collection HDF5 file with the  ```synthetics generate-database generate-images``` command. For data volume reasons, the references images are not duplicated in the *Dispersion* and *DisCo* ensembles.

* A statistics HDF5 file ```stats.h5``` containing dynamical per time-step values such as average forces, distances, etc.. These values can be plotted with the ```synthetics plot``` command from the companion software package.

* A ```command.yml``` file that show detailed information for reproducing the dataset, i.e. every CLI parameters and git revision hash.

## Reference

Please cite the following reference when referencing to this data:

```
@article{geissbuhler2024synthetic,
  title={Synthetic Face Datasets Generation via Latent Space Exploration from Brownian Identity Diffusion},
  author={Geissb{\"u}hler, David and Shahreza, Hatef Otroshi and Marcel, S{\'e}bastien},
  journal={arXiv preprint arXiv:2405.00228},
  year={2024}
}
```