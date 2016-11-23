Exemplar Word Spotting
==

Welcome to the Exemplar Word Spotting library, a software for the localization of words in document images.

This code is the basis of the following project:

###### [Jon Almazán](http://www.cvc.uab.es/~almazan), [Albert Gordo](http://dag.cvc.uab.es/content/albert-gordo), [Alicia Fornés](http://dag.cvc.uab.es/content/alicia-forn%C3%A9s), [Ernest Valveny](http://www.cvc.uab.es/personal2.asp?id=73). **Efficient Exemplar Word Spotting**. In BMVC, 2012. | [Project Page](http://almazan.github.com/ews/) 


Abstract

In this paper we propose an unsupervised segmentation-free method for word spotting in document images. Documents are represented with a grid of HOG descriptors, and a sliding window approach is used to locate the document regions that are most similar to the query. We use the exemplar SVM framework to produce a better representation of the query in an unsupervised way. Finally, the document descriptors are precomputed and compressed with Product Quantization. This offers two advantages: first, a large number of documents can be kept in RAM memory at the same time. Second, the sliding window becomes significantly faster since distances between quantized HOG descriptors can be precomputed.  Our results significantly outperform other segmentation-free methods in the literature, both in accuracy and in speed and memory usage.

---

This word spotting library uses some great open-source software:

* [Yael library](https://gforge.inria.fr/frs/?group_id=2151&release_id=6971#yael-v277-title-content) 

* [JSGD library] (http://lear.inrialpes.fr/src/jsgd/)

* Fast blas convolution code (from [voc-release-4.0](http://www.cs.brown.edu/~pff/latent/))

* HOG feature code (31-D) (from [voc-release-3.1](http://www.cs.brown.edu/~pff/latent/))

* [Exemplar-SVM](http://github.com/quantombone/exemplarsvm)


----

# MATLAB Quick Start Guide

To get started, you need to install MATLAB and download the code from Github. This code has been tested on Linux and pre-compiled Mex files are included.

## Download Exemplar Word Spotting Library source code (MATLAB and C++) and compile it
``` sh
$ cd ~/your_projects/
$ git clone git://github.com/almazan/ews.git
$ cd ~/ews/util
$ ./compileAll.sh
```

## Download and uncompress datasets

Download the GW dataset from [here] (https://www.dropbox.com/s/ewz5c2gmxwjyokm/GW.zip?dl=0) and unzip its content in ews/datasets.


## Script for parameters validation

``` sh
>> validation_script
```



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

=======

