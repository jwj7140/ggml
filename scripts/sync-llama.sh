#!/bin/bash

cp -rpv ../llama.cpp/ggml.c       src/ggml.c
cp -rpv ../llama.cpp/ggml-cuda.cu src/ggml-cuda.cu
cp -rpv ../llama.cpp/ggml-cuda.h  src/ggml-cuda.h
cp -rpv ../llama.cpp/ggml.h       include/ggml/ggml.h
