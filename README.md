A small project I am using to test and fine-tune news summarization models.

```
Conda packages installed:
# Name                    Version                   Build  Channel
abseil-cpp                20211102.0           hd77b12b_0
aiohttp                   3.9.0           py311h2bbff1b_0
aiosignal                 1.2.0              pyhd3eb1b0_0
arrow-cpp                 11.0.0               ha81ea56_2
attrs                     23.1.0          py311haa95532_0
aws-c-common              0.6.8                h2bbff1b_1
aws-c-event-stream        0.1.6                hd77b12b_6
aws-checksums             0.1.11               h2bbff1b_2
aws-sdk-cpp               1.8.185              hd77b12b_1
blas                      1.0                         mkl
boost-cpp                 1.82.0               h59b6b97_2
bottleneck                1.3.5           py311h5bb9823_0
brotli-python             1.0.9           py311hd77b12b_7
bzip2                     1.0.8                he774522_0
c-ares                    1.19.1               h2bbff1b_0
ca-certificates           2023.12.12           haa95532_0
certifi                   2023.11.17      py311haa95532_0
cffi                      1.16.0          py311h2bbff1b_0
charset-normalizer        2.0.4              pyhd3eb1b0_0
colorama                  0.4.6           py311haa95532_0
cryptography              41.0.7          py311h89fc84f_0
cuda-cccl                 12.3.101                      0    nvidia
cuda-cudart               12.1.105                      0    nvidia
cuda-cudart-dev           12.1.105                      0    nvidia
cuda-cupti                12.1.105                      0    nvidia
cuda-libraries            12.1.0                        0    nvidia
cuda-libraries-dev        12.1.0                        0    nvidia
cuda-nvrtc                12.1.105                      0    nvidia
cuda-nvrtc-dev            12.1.105                      0    nvidia
cuda-nvtx                 12.1.105                      0    nvidia
cuda-opencl               12.3.101                      0    nvidia
cuda-opencl-dev           12.3.101                      0    nvidia
cuda-profiler-api         12.3.101                      0    nvidia
cuda-runtime              12.1.0                        0    nvidia
datasets                  2.12.0          py311haa95532_0
dill                      0.3.6           py311haa95532_0
filelock                  3.13.1          py311haa95532_0
freetype                  2.12.1               ha860e81_0
frozenlist                1.4.0           py311h2bbff1b_0
fsspec                    2023.10.0       py311haa95532_0
gflags                    2.2.2                ha925a31_0
giflib                    5.2.1                h8cc25b3_3
glog                      0.5.0                hd77b12b_0
gmpy2                     2.1.2           py311h7f96b67_0
grpc-cpp                  1.48.2               hfe90ff0_1
huggingface_hub           0.17.3          py311haa95532_0
idna                      3.4             py311haa95532_0
importlib-metadata        7.0.1           py311haa95532_0
intel-openmp              2023.1.0         h59b6b97_46320
jinja2                    3.1.2           py311haa95532_0
jpeg                      9e                   h2bbff1b_1
lerc                      3.0                  hd77b12b_0
libboost                  1.82.0               h3399ecb_2
libbrotlicommon           1.0.9                h2bbff1b_7
libbrotlidec              1.0.9                h2bbff1b_7
libbrotlienc              1.0.9                h2bbff1b_7
libcublas                 12.1.0.26                     0    nvidia
libcublas-dev             12.1.0.26                     0    nvidia
libcufft                  11.0.2.4                      0    nvidia
libcufft-dev              11.0.2.4                      0    nvidia
libcurand                 10.3.4.107                    0    nvidia
libcurand-dev             10.3.4.107                    0    nvidia
libcurl                   8.5.0                h86230a5_0
libcusolver               11.4.4.55                     0    nvidia
libcusolver-dev           11.4.4.55                     0    nvidia
libcusparse               12.0.2.55                     0    nvidia
libcusparse-dev           12.0.2.55                     0    nvidia
libdeflate                1.17                 h2bbff1b_1
libevent                  2.1.12               h56d1f94_1
libffi                    3.4.4                hd77b12b_0
libjpeg-turbo             2.0.0                h196d8e1_0
libnpp                    12.0.2.50                     0    nvidia
libnpp-dev                12.0.2.50                     0    nvidia
libnvjitlink              12.1.105                      0    nvidia
libnvjitlink-dev          12.1.105                      0    nvidia
libnvjpeg                 12.1.1.14                     0    nvidia
libnvjpeg-dev             12.1.1.14                     0    nvidia
libpng                    1.6.39               h8cc25b3_0
libprotobuf               3.20.3               h23ce68f_0
libssh2                   1.10.0               he2ea4bf_2
libthrift                 0.15.0               h4364b78_2
libtiff                   4.5.1                hd77b12b_0
libuv                     1.44.2               h2bbff1b_0
libwebp                   1.3.2                hbc33d0d_0
libwebp-base              1.3.2                h2bbff1b_0
lz4-c                     1.9.4                h2bbff1b_0
markupsafe                2.1.3           py311h2bbff1b_0
mkl                       2023.1.0         h6b88ed4_46358
mkl-service               2.4.0           py311h2bbff1b_1
mkl_fft                   1.3.8           py311h2bbff1b_0
mkl_random                1.2.4           py311h59b6b97_0
mpc                       1.1.0                h7edee0f_1
mpfr                      4.0.2                h62dcd97_1
mpir                      3.0.0                hec2e145_1
mpmath                    1.3.0           py311haa95532_0
multidict                 6.0.4           py311h2bbff1b_0
multiprocess              0.70.14         py311haa95532_0
networkx                  3.1             py311haa95532_0
numexpr                   2.8.7           py311h1fcbade_0
numpy                     1.26.3          py311hdab7c0b_0
numpy-base                1.26.3          py311hd01c5d8_0
nvidia-ml-py3             7.352.0                  pypi_0    pypi
openjpeg                  2.4.0                h4fc8c34_0
openssl                   3.0.12               h2bbff1b_0
orc                       1.7.4                h623e30f_1
packaging                 23.1            py311haa95532_0
pandas                    2.1.4           py311hf62ec03_0
pillow                    10.0.1          py311h045eedc_0
pip                       23.3.1          py311haa95532_0
pyarrow                   11.0.0          py311h8a3a540_1
pycparser                 2.21               pyhd3eb1b0_0
pyopenssl                 23.2.0          py311haa95532_0
pysocks                   1.7.1           py311haa95532_0
python                    3.11.7               he1021f5_0
python-dateutil           2.8.2              pyhd3eb1b0_0
python-tzdata             2023.3             pyhd3eb1b0_0
python-xxhash             2.0.2           py311h2bbff1b_1
pytorch                   2.1.2           py3.11_cuda12.1_cudnn8_0    pytorch
pytorch-cuda              12.1                 hde6ce7c_5    pytorch
pytorch-mutex             1.0                        cuda    pytorch
pytz                      2023.3.post1    py311haa95532_0
pyyaml                    6.0.1           py311h2bbff1b_0
re2                       2022.04.01           hd77b12b_0
regex                     2023.10.3       py311h2bbff1b_0
requests                  2.31.0          py311haa95532_0
responses                 0.13.3             pyhd3eb1b0_0
safetensors               0.4.0           py311hcbdf901_0
sentencepiece             0.1.99          py311h59b6b97_0
setuptools                68.2.2          py311haa95532_0
six                       1.16.0             pyhd3eb1b0_1
snappy                    1.1.10               h6c2663c_1
sqlite                    3.41.2               h2bbff1b_0
sympy                     1.12            py311haa95532_0
tbb                       2021.8.0             h59b6b97_0
tk                        8.6.12               h2bbff1b_0
tokenizers                0.13.3          py311h49fca51_0
torchaudio                2.1.2                    pypi_0    pypi
torchvision               0.16.2                   pypi_0    pypi
tqdm                      4.65.0          py311h746a85d_0
transformers              4.32.1          py311haa95532_0
typing-extensions         4.9.0           py311haa95532_1
typing_extensions         4.9.0           py311haa95532_1
tzdata                    2023d                h04d1e81_0
urllib3                   1.26.18         py311haa95532_0
utf8proc                  2.6.1                h2bbff1b_0
vc                        14.2                 h21ff451_1
vs2015_runtime            14.27.29016          h5e58377_2
wheel                     0.41.2          py311haa95532_0
win_inet_pton             1.1.0           py311haa95532_0
xxhash                    0.8.0                h2bbff1b_3
xz                        5.4.5                h8cc25b3_0
yaml                      0.2.5                he774522_0
yarl                      1.9.3           py311h2bbff1b_0
zipp                      3.17.0          py311haa95532_0
zlib                      1.2.13               h8cc25b3_0
zstd                      1.5.5                hd43e919_0
```
