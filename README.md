Exemplar Word Spotting
==

Welcome to the Exemplar Word Spotting library, a software for the localization of words in document images.

This code is the basis of the following project:

## [Jon Almazán](http://www.cvc.uab.es/~almazan), [Albert Gordo], [Alicia Fornés], [Ernest Valveny]. **Efficient Exemplar Word Spotting** In BMVC, 2012. | [Project Page](http://almazan.github.com/ews/) 


Abstract

In this paper we propose an unsupervised segmentation-free method for word spotting in document images. Documents are represented with a grid of HOG descriptors, and a sliding window approach is used to locate the document regions that are most similar to the query. We use the exemplar SVM framework to produce a better representation of the query in an unsupervised way. Finally, the document descriptors are precomputed and compressed with Product Quantization. This offers two advantages: first, a large number of documents can be kept in RAM memory at the same time. Second, the sliding window becomes significantly faster since distances between quantized HOG descriptors can be precomputed.  Our results significantly outperform other segmentation-free methods in the literature, both in accuracy and in speed and memory usage.

---

This word spotting software uses some great open-source software:

* Fast blas convolution code (from [voc-release-4.0](http://www.cs.brown.edu/~pff/latent/)), 

* HOG feature code (31-D) (from [voc-release-3.1](http://www.cs.brown.edu/~pff/latent/)), 

* [Yael library](http://www.cs.brown.edu/~pff/latent/)), 

* [JSGD library] (http://lear.inrialpes.fr/src/jsgd/),

----

# MATLAB Quick Start Guide

To get started, you need to install MATLAB and download the code from Github. This code has been tested on Mac OS X and Linux.  Pre-compiled Mex files for Mac OS X and Linux are included.

## Download Exemplar-SVM Library source code (MATLAB and C++) and compile it
``` sh
$ cd ~/projects/
$ git clone git@github.com:quantombone/exemplarsvm.git
$ cd ~/projects/exemplarsvm
$ matlab
>> esvm_compile
```

## Download and load pre-trained VOC2007 model(s)
``` sh
$ matlab
>> addpath(genpath(pwd))
>> [models, M, test_set] = esvm_download_models('voc2007-bus');
```

or

``` sh
$ wget http://people.csail.mit.edu/~tomasz/exemplarsvm/voc2007-models.tar
$ tar -xf voc2007-models.tar
$ matlab
>> load voc2007_bus.mat
>> [models, M, test_set] = esvm_download_models('voc2007-bus.mat');
```

You can alternatively download the pre-trained models individually from [http://people.csail.mit.edu/tomasz/exemplarsvm/models/](http://people.csail.mit.edu/tomasz/exemplarsvm/models/) or a tar file of all models [voc2007-models.tar](http://people.csail.mit.edu/tomasz/exemplarsvm/models/voc2007-models.tar) (NOTE: tar file is 450MB)

## Demo: Apply models to a set of images

``` sh
>> esvm_demo_apply;
```

See the file [tutorial/esvm_demo_apply.html](http://people.csail.mit.edu/tomasz/exemplarsvm/tutorial/esvm_demo_apply.html) for a step-by-step tutorial on what esvm_demo_apply.m produces


# Training an Ensemble of Exemplar-SVMs

## Demo: Synthetic-data training and testing

``` sh
>> esvm_demo_train_synthetic;
```

See the file [tutorial/esvm_demo_train_synthetic.html](http://people.csail.mit.edu/tomasz/exemplarsvm/tutorial/esvm_demo_train_synthetic.html) for a step-by-step tutorial on what esvm_demo_train_synthetic.m produces

The training scripts are designed to work with the PASCAL VOC 2007
dataset, so we need to download that first.

## Prerequsite: Install PASCAL VOC 2007 trainval/test sets
``` sh
$ mkdir /nfs/baikal/tmalisie/pascal #Make a directory for the PASCAL VOC data
$ cd /nfs/baikal/tmalisie/pascal
$ wget http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
$ wget http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCtest_06-Nov-2007.tar
$ tar xf VOCtest_06-Nov-2007.tar 
$ tar xf VOCtrainval_06-Nov-2007.tar 
``` 
You can also get the VOC 2007 dataset tar files manually, [VOCtrainval_06-Nov-2007.tar](http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) and [VOCtest_06-Nov-2007.tar](http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCtest_06-Nov-2007.tar)

## Demo: Training and Evaluating an Ensemble of "bus" Exemplar-SVMs quick-demo
``` sh
>> data_dir = '/your/directory/to/pascal/VOCdevkit/';
>> dataset = 'VOC2007';
>> results_dir = '/your/results/directory/';
>> [models,M] = esvm_demo_train_voc_class_fast('car', data_dir, dataset, results_dir);
# All output (models, M-matrix, AP curve) has been written to results_dir
```

See the file [tutorial/esvm_demo_train_voc_class_fast.html](http://people.csail.mit.edu/tomasz/exemplarsvm/tutorial/esvm_demo_train_voc_class_fast.html) for a step-by-step tutorial on what esvm_demo_train_voc_class_fast.m produces


## Script: Training and Evaluating an Ensemble of "bus" Exemplar-SVMs full script
``` sh
>> data_dir = '/your/directory/to/pascal/VOCdevkit/';
>> dataset = 'VOC2007';
>> results_dir = '/your/results/directory/';
>> [models,M] = esvm_script_train_voc_class('bus', data_dir, dataset, results_dir);
# All output (models, M-matrix, AP curve) has been written to results_dir
```

# Extra: How to run the Exemplar-SVM framework on a cluster

This library was meant to run on a cluster with a shared NFS/AFS file
structure where all nodes can read/write data from a common data
source/target.  The PASCAL VOC dataset must be installed on such a
shared resource and the results directory as well.  The idea is that
results are written as .mat files and intermediate work is protected
via lock files. Lock files are temporary files (they are directories
actually) which are deleted once something has finished process.  This
means that the entire voc training script can be replicated across a
cluster, you can run the script 200x times and the training will
happen in parallel.

To run ExemplarSVM on a cluster, first make sure you have a cluster,
use an ssh-based launcher such as my
[warp_scripts](https://github.com/quantombone/warp_scripts) github
project.  I have used warp_starter.sh at CMU (using WARP cluster)
and sc.sh at MIT (using the continents).


--- 
**Copyright (C) 2012 by Jon Almazán**

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
