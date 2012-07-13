Here are some util functions used in the Exemplar Word Spotting framework.

Compile with **compileAll.sh** bash script

** compute_scores[L2,PQ_L2].m: a function that given a word and a document image representation, performs a sliding window and returns the score of all the patches.

** hogDraw.m: a Matlab function that creates a visualization of the HOG descriptor. Credits to Piotr Dollar.

** nms_C.c: a function that computed the Non-Maxima Supression algorithm. Returns the indices of the selected regions.

** visualize_box.m: a Matlab function that shows the given bounding box of an image.
