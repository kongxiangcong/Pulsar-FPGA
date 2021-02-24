#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include "fft.h"

DTYPE In_R[SIZE], In_I[SIZE];
DTYPE OUT_R[SIZE], OUT_I[SIZE];
DTYPE WW_R[SIZE], WW_I[SIZE];
/*
DTYPE f_coord[FCOORD], f_coord1[FCOORD];
DTYPE dedisp_seq[SIZE];
*/
int main()
{
	FILE *fp;
	FILE *fp_r, *fp_i;

    // reading raw file
	printf("READING INPUT PULSAR DATAS FROM RAW FILE....\n");
	DTYPE *h5buf;
	FILE *rawfile;
	rawfile=fopen("pulsar_S0.raw","r");

	h5buf=(float *) malloc(sizeof(float) * SIZE);
	fread(h5buf, sizeof(float), SIZE, rawfile);
    for(int i=0; i<16384; i++){
		In_R[i] = h5buf[i];
		In_I[i] = 0.0;
    }
	/*
    //TEST raw file:
	fp=fopen("read_raw_cpp.dat", "w");
	for(int i=0; i<16384; i++){
		In_R[i] = h5buf[i];
		In_I[i] = 0.0;
		fprintf(fp, "%4d\t%f\t%f\n", i, In_R[i], In_I[i]);
	}
	fclose(fp);
	*/


	//Twiddle factor is calculated here and saved in fft.h to be used in offline.
	double	e = -6.2831853071795864769;
	printf("GENERATING %d TWIDDLE FACTORS\n", SIZE);
	fp_r=fopen("tw_r.h", "w");
	fp_i=fopen("tw_i.h", "w");
	fprintf(fp_r, "const DTYPE W_real[]={");
	fprintf(fp_i, "const DTYPE W_imag[]={");
	for(int i=0; i<SIZE2; i++)
	{
		//COMPLEX W;	// e^(-j 2 pi/ N)
	  double w = e*double(i)/double(SIZE);
	  WW_R[i]=cos(w);
	  WW_I[i]=sin(w);
	  //printf("%4d\t%f\t%f\n",i,WW_R[i],WW_I[i]);
		fprintf(fp_r, "%.20f,",WW_R[i]);
		fprintf(fp_i, "%.20f,",WW_I[i]);
		if(i%16==0)
			{
				fprintf(fp_r, "\n");
				fprintf(fp_i, "\n");
			}
	}
	fprintf(fp_r, "};\n");
	fprintf(fp_i, "};\n");
	fclose(fp_r);
	fclose(fp_i);


	//Perform FFT
    #ifdef S1_baseline
		//
        //printf("%.4f\t%.4f\t%.4f\t%.4f\n", In_R[10], In_R[20], In_R[35], In_R[2560]);
		fft(In_R, In_I);
        
		//Print output
		fp=fopen("out.fft.dat", "w");
		printf("Printing FFT Output\n");
		for(int i=0; i<SIZE; i++){
		  //printf("%4d\t%f\t%f\n",i,In_R[i],In_I[i]);
			fprintf(fp, "%4d\t%f\t%f\n",i,In_R[i],In_I[i]);
		}
		fclose(fp);
	#endif

	#ifdef S2_Unroll
		fft(In_R, In_I, OUT_R, OUT_I);
        printf("%4f\n",3.0+OUT_R[25]);
		//Print output
		fp=fopen("out.fft.dat", "w");
		printf("Printing FFT Output\n");
		for(int i=0; i<SIZE; i++){
		  //printf("%4d\t%f\t%f\n",i,In_R[i],In_I[i]);
			fprintf(fp, "%4d\t%f\t%f\n",i,OUT_R[i],OUT_I[i]);
		}
		fclose(fp);
	#endif

	#ifdef S4_DATAFLOW
		fft(In_R, In_I, OUT_R, OUT_I);
        printf("%4f\n",3.0+OUT_R[25]);
		//Print output
		fp=fopen("out.fft.dat", "w");
		printf("Printing FFT Output\n");
		for(int i=0; i<SIZE; i++){
		  //printf("%4d\t%f\t%f\n",i,In_R[i],In_I[i]);
			fprintf(fp, "%4d\t%f\t%f\n",i,OUT_R[i],OUT_I[i]);
		}
		fclose(fp);
	#endif
	
	printf ("Comparing against output data \n");
	// std::ifstream golden("out.fft.gold.dat");
    std::ifstream golden("scipy_fft.dat");
	DTYPE error = 0.0;
	DTYPE maxerror = 0.0;
	for(int i=0; i<SIZE; i++) {
	  DTYPE rx, ix;
	  int j;
	  golden >> j >> rx >> ix;

    #ifdef S1_baseline
	  	  DTYPE newerror = fabs(rx-In_R[i]) + fabs(ix-In_I[i]);
	  #endif

      #ifdef S2_Unroll
		  DTYPE newerror = fabs(rx-OUT_R[i]) + fabs(ix-OUT_I[i]);
	  #endif

	  #ifdef S4_DATAFLOW
		  DTYPE newerror = fabs(rx-OUT_R[i]) + fabs(ix-OUT_I[i]);
	  #endif

	  error += newerror;
	  if(newerror > maxerror) {
	    maxerror = newerror; 
	    fprintf(stdout, "Max Error@%d: %f\n", i, maxerror);
	  }
	}

//	fprintf(stdout, "Average Error: %f\n", error/SIZE);
	
	if ((error/SIZE) > .08 || maxerror > 2) { // This is somewhat arbitrary.  Should do proper error analysis.
	  fprintf(stdout, "*******************************************\n");
	  fprintf(stdout, "FAIL: Output DOES NOT match the golden output\n");
      printf("avg error = %f\n", (error/SIZE));
	  fprintf(stdout, "*******************************************\n");
	  return 1;
	}
	else {
	  fprintf(stdout, "*******************************************\n");
	  fprintf(stdout, "PASS: The output matches the golden output!\n");
      printf("avg error = %f\n", (error/SIZE));
	  fprintf(stdout, "*******************************************\n");
	  return 0;
	}
	
}
