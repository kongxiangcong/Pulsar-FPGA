#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include "fft.h"

int main() {
    FILE *fp;
    DTYPE WW_R[SIZE];

    //Twiddle factor is calculated here and saved in fft.h to be used in offline.
	double	e = -6.2831853071795864769;
	printf("GENERATING %d TWIDDLE FACTORS\n", SIZE);
	fp=fopen("2coefficients16384.h", "w");
	fprintf(fp, "const DTYPE sin_coefficients_table[]={");

	for(int i=0; i<SIZE2; i++)
	{
		//COMPLEX W;	// e^(-j 2 pi/ N)
	  double w = e*double(i)/double(SIZE);
	  WW_R[i]=sin(w);
	  //WW_I[i]=sin(w);
	  
		fprintf(fp, "%.6f,",WW_R[i]);

	}
	fprintf(fp, "};\n");

	fclose(fp);

}