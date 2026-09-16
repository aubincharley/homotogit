# Literature and citation scope

The paper uses primary research papers, their publisher/PMLR/CVF records, author manuscripts or arXiv records. Continuation guarantees are attributed only with their assumptions; spatial filtering is not identified with convolution of the parameter-space loss. CBS and SDPoint are direct methodological antecedents, not experimentally reproduced baselines in this package. Frequency bias of the learned function, spatial feature-map filtering, and spatial spectra of an input Jacobian are distinguished.

Li et al. motivates normalized weight slices and PCA; Dinh et al. supports the reparameterization caveat. Pearlmutter supports matrix-free Hessian-vector products and Ghorbani et al. the neural-net spectral context. Sokolic et al. supplies context for Jacobian/margin analyses; the measured average logit Jacobians are not substituted for that paper's uniform bounds. Garipov et al. supplies the counterpoint to interpreting a straight-segment barrier as disconnected basins. Hutchinson is cited for the stochastic trace estimator, and ROF for the TV reconstruction tradition, with no claim that its original formulation equals our constrained preview specification.

Dataset, architecture, normalization and optimizer citations identify the source methods. Their papers do not certify that the transferred experimental recipes are canonical or optimally tuned. The current paper's numerical conclusions are supported by the experiment exports, not by citations to those methods.

The compiled manuscript cites 34 distinct bibliography entries. Uncited existing entries are retained in `references.bib` for author continuity but do not appear in the PDF.

| Key | Primary record / manuscript | Year |
|---|---|---|
| `allgower1990continuation` | [Numerical Continuation Methods: An Introduction](https://link.springer.com/book/10.1007/978-3-642-61257-2) | 1990 |
| `bengio2009curriculum` | [Curriculum Learning](https://icml.cc/2009/papers/119.pdf) | 2009 |
| `coates2011analysis` | [An Analysis of Single-Layer Networks in Unsupervised Feature Learning](https://proceedings.mlr.press/v15/coates11a.html) | 2011 |
| `dinh2017sharp` | [Sharp Minima Can Generalize for Deep Nets](https://proceedings.mlr.press/v70/dinh17b.html) | 2017 |
| `garipov2018surfaces` | [Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs](https://arxiv.org/abs/1802.10026) | 2018 |
| `ghorbani2019hessian` | [An Investigation into Neural Net Optimization via Hessian Eigenvalue Density](https://proceedings.mlr.press/v97/ghorbani19b.html) | 2019 |
| `gulcehre2016mollifying` | [Mollifying Networks](https://arxiv.org/abs/1608.04980) | 2016 |
| `hazan2016graduated` | [On Graduated Optimization for Stochastic Non-Convex Problems](https://proceedings.mlr.press/v48/hazanb16.html) | 2016 |
| `he2016resnet` | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | 2016 |
| `hendrycks2016gelu` | [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415) | 2016 |
| `hoffer2019mixmatch` | [Mix & Match: Training Convnets with Mixed Image Sizes for Improved Accuracy, Speed and Scale Resiliency](https://arxiv.org/abs/1908.08986) | 2019 |
| `hutchinson1989trace` | [A Stochastic Estimator of the Trace of the Influence Matrix for Laplacian Smoothing Splines](https://doi.org/10.1080/03610918908812806) | 1989 |
| `ioffe2015batch` | [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://proceedings.mlr.press/v37/ioffe15.html) | 2015 |
| `kingma2015adam` | [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) | 2015 |
| `krizhevsky2009learning` | [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) | 2009 |
| `kuen2018stochastic` | [Stochastic Downsampling for Cost-Adjustable Inference and Improved Regularization in Convolutional Networks](https://openaccess.thecvf.com/content_cvpr_2018/html/Kuen_Stochastic_Downsampling_for_CVPR_2018_paper.html) | 2018 |
| `li2018visualizing` | [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913) | 2018 |
| `liu2020radam` | [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265) | 2020 |
| `loshchilov2019decoupled` | [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) | 2019 |
| `michaeli2023aliasfree` | [Alias-Free Convnets: Fractional Shift Invariance via Polynomial Activations](https://arxiv.org/abs/2303.08085) | 2023 |
| `mobahi2015gaussian` | [A Theoretical Analysis of Optimization by Gaussian Continuation](https://ojs.aaai.org/index.php/AAAI/article/view/9356) | 2015 |
| `netzer2011reading` | [Reading Digits in Natural Images with Unsupervised Feature Learning](https://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf) | 2011 |
| `pearlmutter1994hessian` | [Fast Exact Multiplication by the Hessian](https://www.bcl.hamilton.ie/~barak/papers/nc-hessian.pdf) | 1994 |
| `rahaman2019spectral` | [On the Spectral Bias of Neural Networks](https://proceedings.mlr.press/v97/rahaman19a.html) | 2019 |
| `ramachandran2017searching` | [Searching for Activation Functions](https://arxiv.org/abs/1710.05941) | 2017 |
| `rudin1992tv` | [Nonlinear Total Variation Based Noise Removal Algorithms](https://doi.org/10.1016/0167-2789(92)90242-F) | 1992 |
| `simonyan2015very` | [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) | 2015 |
| `sinha2020curriculum` | [Curriculum by Smoothing](https://proceedings.neurips.cc/paper_files/paper/2020/hash/f6a673f09493afcd8b129a0bcf1cd5bc-Abstract.html) | 2020 |
| `sokolic2017robust` | [Robust Large Margin Deep Neural Networks](https://arxiv.org/abs/1605.08254) | 2017 |
| `stergiou2021softpool` | [Refining Activation Downsampling with SoftPool](https://arxiv.org/abs/2101.00440) | 2021 |
| `tan2021efficientnetv2` | [EfficientNetV2: Smaller Models and Faster Training](https://proceedings.mlr.press/v139/tan21a.html) | 2021 |
| `vasconcelos2021aliasing` | [Impact of Aliasing on Generalization in Deep Convolutional Networks](https://arxiv.org/abs/2108.03489) | 2021 |
| `wu2018group` | [Group Normalization](https://arxiv.org/abs/1803.08494) | 2018 |
| `zhang2019shift` | [Making Convolutional Networks Shift-Invariant Again](https://proceedings.mlr.press/v97/zhang19a.html) | 2019 |
