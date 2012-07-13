#include <stdlib.h>
#include <mex.h>   
#include <math.h>

#define getPos(cr,cc,ch,rows,cols, dims) ( cols*dims*cr + dims*cc + ch  )

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
	 * 0 - flat
	 * 1 - model.root
	 * 2 - bH
	 * 3 - bW
	 * 4 - dim
	 * 5 - nbinsH
	 * 6 - nbinsW
	 * 7 - step
	 * 8 - numWindows
	 * OUT:
	 * 0 - scr
	 */
	int i,j,k,l;
    float tmp;
	float *flat;
    int Nflat;
	float *w;
    int Nw;
	int bH,bW,dim, nbinsH,nbinsW,step,numWindows;
	double *scr;
	
	int pp, hh,ii,jj;
	int pos;
	float norm;
	flat =(float*) mxGetData(prhs[0]);
    Nflat = mxGetM(prhs[0]);
	w = (float*)mxGetData(prhs[1]);
    Nw = mxGetN(prhs[1]);
	bH = mxGetScalar(prhs[2]);
	bW = mxGetScalar(prhs[3]);
	dim = mxGetScalar(prhs[4]);
	nbinsH= mxGetScalar(prhs[5]);
	nbinsW = mxGetScalar(prhs[6]);
	step = mxGetScalar(prhs[7]);
	numWindows = mxGetScalar(prhs[8]);
	
	plhs[0] = mxCreateDoubleMatrix(numWindows ,1 , mxREAL);
    scr = mxGetPr(plhs[0]);
		
	k=0;

    
	for (i=0; i <= bH-nbinsH; i+=step)
	{
		for (j=0; j <= bW-nbinsW; j+=step)
		{
			scr[k]=0;
			pp=0;
            norm=0;
            
            for (ii=i;ii < i+nbinsH;ii++)
            {
                pos = getPos(ii,j,0,bH,bW,dim);
                scr[k] = scr[k] + dp(nbinsW*dim, &flat[pos], &w[pp]);
                for (l=0;l < nbinsW*dim;l++)
                {
                        tmp = flat[pos+l];
                        norm+=(tmp*tmp);
                }
                pp+=nbinsW*dim;
            }
            scr[k]=scr[k]/sqrt(norm);
			k++;
		}
	}
	
    return;
}
