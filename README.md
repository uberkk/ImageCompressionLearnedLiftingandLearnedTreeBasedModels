# IMAGE-COMPRESSION-WITH-LEARNED-LIFTING-BASED-DWT-AND-LEARNED-TREE-BASED-ENTROPY-MODELS

This repo includes the codes for IMAGE COMPRESSION WITH LEARNED LIFTING-BASED DWT AND LEARNED TREE-BASED ENTROPY MODELS.


Required libraries can be found in requirements.txt
pytorch-wavelets should be installed to use CDF97 filters with torch tensors.
For experiments few weight files are uploaded can be downloaded via the link in weight_link.txt

It should be noted that this repo includes the Version1 files for the proposed work.

**Abstract**

This paper explores learned image compression based on traditional and learned discrete wavelet transform (DWT) architectures and learned entropy models for coding DWT subband coefficients. A learned DWT is obtained through the lifting scheme with learned nonlinear predict and update filters. Several learned entropy models are proposed to exploit inter and intra-DWT subband coefficient dependencies, akin to traditional EZW, SPIHT, or EBCOT algorithms. Experimental results show that when the proposed learned entropy models are combined with traditional wavelet filters, such as the CDF 9/7 filters, compression performance that far exceeds that of JPEG2000 can be achieved. When the learned entropy models are combined with the learned DWT, compression performance increases further. The computations in the learned DWT and all entropy models, except one, can be simply parallelized, and the systems provide practical encoding and decoding times on GPUs.


**Citation**
    
@misc{https://doi.org/10.48550/arxiv.2212.03616,
  doi = {10.48550/ARXIV.2212.03616},
  url = {https://arxiv.org/abs/2212.03616},
  author = {Sahin, Ugur Berk and Kamisli, Fatih},
  keywords = {Image and Video Processing (eess.IV), Multimedia (cs.MM), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Image Compression With Learned Lifting-Based DWT and Learned Tree-Based Entropy Models},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}

