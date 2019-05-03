# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#
# Multistage build.
#

ARG BASE_IMAGE=nvcr.io/nvidia/tensorrtserver:19.04-py3
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:19.04-py3

############################################################################
## Caffe2 stage: Use PyTorch container to get Caffe2 backend
############################################################################
FROM ${PYTORCH_IMAGE} AS trtserver_caffe2

# We cannot just pull libraries from the PyTorch container... we need
# to:
#   - copy over netdef_backend_c2 interface so it can build with other
#     C2 sources
#   - need to patch as explained below

# Copy netdef_backend_c2 into Caffe2 core so it builds into the
# libcaffe2 library. We want netdef_backend_c2 to build against the
# Caffe2 protobuf since it interfaces with that code.
COPY src/backends/caffe2/netdef_backend_c2.* \
     /opt/pytorch/pytorch/caffe2/core/

# Avoid failure when peer access already enabled for CUDA device
COPY tools/patch/caffe2 /tmp/patch/caffe2
RUN sha1sum -c /tmp/patch/caffe2/checksums && \
    patch -i /tmp/patch/caffe2/core/context_gpu.cu \
          /opt/pytorch/pytorch/caffe2/core/context_gpu.cu

# Build same as in pytorch container... except for the NO_DISTRIBUTED
# line where we turn off features not needed for trtserver
WORKDIR /opt/pytorch
RUN pip uninstall -y torch
RUN cd pytorch && \
    TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5+PTX" \
      CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
      NCCL_INCLUDE_DIR="/usr/include/" \
      NCCL_LIB_DIR="/usr/lib/" \
      NO_DISTRIBUTED=1 NO_TEST=1 NO_MIOPEN=1 USE_MKLDNN=0 USE_OPENCV=OFF USE_LEVELDB=OFF \
      python setup.py install && python setup.py clean

############################################################################
## Build stage: Build inference server
############################################################################
FROM ${BASE_IMAGE} AS trtserver_build

ARG TRTIS_VERSION=1.3.0dev
ARG TRTIS_CONTAINER_VERSION=19.06dev

# libgoogle-glog0v5 is needed by caffe2 libraries.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            software-properties-common \
            autoconf \
            automake \
            build-essential \
            cmake \
            libgoogle-glog0v5 \
            libre2-dev \
            libssl-dev \
            libtool

# Caffe2 library requirements...
COPY --from=trtserver_caffe2 \
     /opt/conda/lib/python3.6/site-packages/torch/lib/libcaffe2_detectron_ops_gpu.so \
     /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 \
     /opt/conda/lib/python3.6/site-packages/torch/lib/libcaffe2.so \
     /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 \
     /opt/conda/lib/python3.6/site-packages/torch/lib/libcaffe2_gpu.so \
     /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 \
     /opt/conda/lib/python3.6/site-packages/torch/lib/libc10.so \
     /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 \
     /opt/conda/lib/python3.6/site-packages/torch/lib/libc10_cuda.so \
     /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 /opt/conda/lib/libmkl_avx2.so /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 /opt/conda/lib/libmkl_core.so /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 /opt/conda/lib/libmkl_def.so /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 /opt/conda/lib/libmkl_gnu_thread.so /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 /opt/conda/lib/libmkl_intel_lp64.so /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 /opt/conda/lib/libmkl_rt.so /opt/tensorrtserver/lib/
COPY --from=trtserver_caffe2 /opt/conda/lib/libmkl_vml_def.so /opt/tensorrtserver/lib/

# Copy entire repo into container even though some is not needed for
# build itself... because we want to be able to copyright check on
# files that aren't directly needed for build.
WORKDIR /workspace
RUN rm -fr *
COPY . .

# Build the server. The build can fail the first time due to a
# protobuf_generate_cpp error that doesn't repeat on subsequent
# builds, which is why there are 2 make invocations below.
RUN cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DTRTIS_ENABLE_METRICS=ON \
          -DTRTIS_ENABLE_CUSTOM=ON \
          -DTRTIS_ENABLE_TENSORFLOW=OFF \
          -DTRTIS_ENABLE_TENSORRT=OFF \
          -DTRTIS_ENABLE_CAFFE2=OFF && \
    (make -j16 trtis || true) && \
    make -j16 trtis && \
    mkdir -p /opt/tensorrtserver && \
    cp -r trtis/install/* /opt/tensorrtserver/. && \
    (cd /opt/tensorrtserver && ln -s /workspace/qa qa)

ENV TENSORRT_SERVER_VERSION ${TRTIS_VERSION}
ENV NVIDIA_TENSORRT_SERVER_VERSION ${TRTIS_CONTAINER_VERSION}

ENV LD_LIBRARY_PATH /opt/tensorrtserver/lib:${LD_LIBRARY_PATH}
ENV PATH /opt/tensorrtserver/bin:${PATH}

COPY nvidia_entrypoint.sh /opt/tensorrtserver
ENTRYPOINT ["/opt/tensorrtserver/nvidia_entrypoint.sh"]

############################################################################
##  Production stage: Create container with just inference server executable
############################################################################
FROM ${BASE_IMAGE}

ARG TRTIS_VERSION=1.3.0dev
ARG TRTIS_CONTAINER_VERSION=19.06dev

ENV TENSORRT_SERVER_VERSION ${TRTIS_VERSION}
ENV NVIDIA_TENSORRT_SERVER_VERSION ${TRTIS_CONTAINER_VERSION}
LABEL com.nvidia.tensorrtserver.version="${TENSORRT_SERVER_VERSION}"

ENV LD_LIBRARY_PATH /opt/tensorrtserver/lib:${LD_LIBRARY_PATH}
ENV PATH /opt/tensorrtserver/bin:${PATH}

ENV TF_ADJUST_HUE_FUSED         1
ENV TF_ADJUST_SATURATION_FUSED  1
ENV TF_ENABLE_WINOGRAD_NONFUSED 1
ENV TF_AUTOTUNE_THRESHOLD       2

# Create a user that can be used to run the tensorrt-server as
# non-root. Make sure that this user to given ID 1000.
ENV TENSORRT_SERVER_USER=tensorrt-server
RUN id -u $TENSORRT_SERVER_USER > /dev/null 2>&1 || \
    useradd $TENSORRT_SERVER_USER && \
    [ `id -u $TENSORRT_SERVER_USER` -eq 1000 ] && \
    [ `id -g $TENSORRT_SERVER_USER` -eq 1000 ]

# libgoogle-glog0v5 is needed by caffe2 libraries.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            libgoogle-glog0v5 \
            libre2-1v5

WORKDIR /opt/tensorrtserver
RUN rm -fr /opt/tensorrtserver/*
COPY LICENSE .
COPY --from=trtserver_caffe2 /opt/pytorch/pytorch/LICENSE LICENSE.pytorch
COPY --from=trtserver_build /opt/tensorrtserver/bin/trtserver bin/
COPY --from=trtserver_build /opt/tensorrtserver/lib lib
COPY --from=trtserver_build /opt/tensorrtserver/include include

RUN chmod ugo-w+rx /opt/tensorrtserver/lib/*.so

# Extra defensive wiring for CUDA Compat lib
RUN ln -sf ${_CUDA_COMPAT_PATH}/lib.real ${_CUDA_COMPAT_PATH}/lib \
 && echo ${_CUDA_COMPAT_PATH}/lib > /etc/ld.so.conf.d/00-cuda-compat.conf \
 && ldconfig \
 && rm -f ${_CUDA_COMPAT_PATH}/lib

COPY nvidia_entrypoint.sh /opt/tensorrtserver
ENTRYPOINT ["/opt/tensorrtserver/nvidia_entrypoint.sh"]

ARG NVIDIA_BUILD_ID
ENV NVIDIA_BUILD_ID ${NVIDIA_BUILD_ID:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
ARG NVIDIA_BUILD_REF
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"
