
#include <stdlib.h>
#include <mex.h>   
#include <math.h>


#define getPos(cr,cc,rows,cols) ( cols*cr + cc)

float dp(int N, float *x, float *y)
{
        int i;
        float total=0;
        for (i=0; i < N;i++) 
        {
                total+=x[i]*y[i];
        }
        return total;
}
/*
float dp_sse(int n, float *x, float *y)
{
        int i,p;
        float total=0;
        float tmp;
        __m128 a, b, res;
        int N;
        N = (n/4)*4;
        for (p=0; p < N; p+=4)
        {
                a = _mm_loadu_ps(x+p);
                b = _mm_loadu_ps(y+p);
                res = _mm_dp_ps(a, b, 0xff);     
                _mm_store_ss(&tmp,res); 
                total+=tmp;
        }
        for (i=N; i < n;i++) 
        {
                total+=x[i]*y[i];
        }
        return total;
}
*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{   
	
	/*
	 * IN:
	 * 0 - flat (bH * bW * 1)
	 * 1 - lookup (256 x (nbinsH*nbinsW))
     * 2 - partial norm (1x256)
	 * 3 - bH
	 * 4 - bW
	 * 5 - nbinsH
	 * 6 - nbinsW
	 * 7 - step
	 * 8 - numWindows
	 * OUT:
	 * 0 - scr
	 */
	int i,j,k;
	int *flat;
	float *lookup;
    float *squares;
	int bH,bW,dim, nbinsH,nbinsW,step,numWindows;
    int wsize;
	double *scr;
	double partialNorm;
	int pp, hh,jj,ii;
    int idx;
	int pos;
	
	flat =(int*) mxGetData(prhs[0]);
	lookup = (float*)mxGetData(prhs[1]);
    squares = (float*)mxGetData(prhs[2]);
	bH = mxGetScalar(prhs[3]);
	bW = mxGetScalar(prhs[4]);
	nbinsH= mxGetScalar(prhs[5]);
	nbinsW = mxGetScalar(prhs[6]);
	step = mxGetScalar(prhs[7]);
	numWindows = mxGetScalar(prhs[8]);
    
    wsize = nbinsH * nbinsW;
	
    /* Output scores */
	plhs[0] = mxCreateDoubleMatrix(numWindows ,1 , mxREAL);
    scr = mxGetPr(plhs[0]);
		
	k=0;
    /* For each possible window */
	for (i=0; i <= bH-nbinsH; i+=step)
	{
		for (j=0; j <= bW-nbinsW; j+=step)
		{
            /*printf("Doing window %d. Starting point at %d,%d\n", k, i, j);*/
            scr[k]=0;
            partialNorm=0;
			pp=0;
            /* Explore the window */
            for (ii=i;ii < i+nbinsH;ii++)
            {
                for (jj=j;jj < j + nbinsW; jj++)
                {
                    /* Get the PQ index of that doc cell */
                    pos = getPos(ii,jj,bH,bW);
                    idx = flat[pos];
                    /* Get the distance from the lookup table, according to the position in the subwindow (p) */
                    scr[k] +=  lookup[pp*256 + idx];
                    partialNorm+=squares[idx];
                    /*printf("%dx%d (%d) - %d - %d - %.4f\n",ii,jj,pp,pos,idx,lookup[pp*256 + idx]);*/
                    pp++;
                }
            }
            scr[k]=scr[k]/sqrt(partialNorm);
			k++;
		}
	}	
    return;
}
