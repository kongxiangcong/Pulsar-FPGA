#ifndef FFT_H_
#define FFT_H_

typedef float DTYPE;
typedef int INTTYPE;
#define M 14 			/* Number of Stages = Log2N */
#define SIZE 16384 		/* SIZE OF FFT */
#define SIZE2 SIZE>>1	/* SIZE/2 */


//pulsar coefficients
#define FCOORD 1024
#define D 4.148808*1e+15
#define BW 1024
#define fs 32768
#define fo 16384

//#define S1_baseline
//#define S2_Unroll
//#define S3_LUT
#define S4_DATAFLOW
//#define S5_Effect_Improve

#ifdef S1_baseline
void fft(DTYPE X_R[SIZE], DTYPE X_I[SIZE]);
#endif

#ifdef S2_Unroll
void fft(DTYPE X_R[SIZE], DTYPE X_I[SIZE], DTYPE OUT_R[SIZE], DTYPE OUT_I[SIZE]);
#endif

#ifdef S4_DATAFLOW
void fft(DTYPE X_R[SIZE], DTYPE X_I[SIZE], DTYPE OUT_R[SIZE], DTYPE OUT_I[SIZE]);
#endif

void co_dedisp(int DM, DTYPE FFT_R[SIZE], DTYPE FFT_I[SIZE], DTYPE OUT_R[SIZE], DTYPE OUT_I[SIZE]);

#include "tw_r.h"
#include "tw_i.h"
//#include "coefficients1024.h"
#include "coefficients16384.h"

#endif
