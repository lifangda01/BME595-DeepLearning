#include <stdio.h>
#include <stdlib.h>

/* Data of a tensor is stored as an 1-D ARRAY */
#define ARR_IDX(r,c,w) ((r)*(w)+c)

double pixel_conv(double *x, double *k, int row, int col, int x_c, int k_s)
{
	int h_s = k_s / 2;
	double result = 0;
	for (int m = 0; m < k_s; ++m)
	{
		for (int n = 0; n < k_s; ++n)
		{
			result += x[ARR_IDX(row+m, col+n, x_c)] * k[k_s*k_s-1 - ARR_IDX(m, n, k_s)];
		}
	}
	return result;
}

void conv(double *x, double *k, double *r, int x_c, int r_r, int r_c, int k_s)
{
	#pragma omp parallel for
	for (int row = 0; row < r_r; ++row)
	{
		for (int col = 0; col < r_c; ++col)
		{
			r[ARR_IDX(row, col , r_c)] = pixel_conv(x, k, row, col, x_c, k_s);
		}
	}
}