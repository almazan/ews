/opt/matlab2009a/bin/mex -largeArrayDims -O compute_scores_L2.c CFLAGS="\$CFLAGS -O3 -funroll-loops -ffast-math -fexpensive-optimizations"
/opt/matlab2009a/bin/mex -largeArrayDims -O compute_scores_PQ_L2.c CFLAGS="\$CFLAGS -O3 -funroll-loops -ffast-math -fexpensive-optimizations"
/opt/matlab2009a/bin/mex -largeArrayDims -O nms_C.c CFLAGS="\$CFLAGS -O3 -funroll-loops -ffast-math -fexpensive-optimizations"
