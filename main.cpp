/*
 * main.cpp
 *
 *  Created on: Jul 25, 2014
 *      Author: Abuenameh
 */

#include <ctime>

#include "L-BFGS/lbfgsb.h"

#define L 50
#define nmax 7

real* f_tb_host;
real* f_tb_dev;

extern void initProb(real* x, int* nbd, real* l, real* u, int dim);
extern void energy(real* x, real* f, real* g, int L_, int nmax_, real U, real J, real mu,
		const cudaStream_t& stream);

cublasHandle_t cublasHd;
real stpscal;

real U;
real J;
real mu;

void funcgrad(real* x, real& f, real* g, const cudaStream_t& stream) {
	energy(x, f_tb_dev, g, L, nmax, U, J, mu, stream);
	f = *f_tb_host;
}

int main() {

	time_t start = time(NULL);

	int dim = L * (nmax + 1);

	const real epsg = EPSG;
	const real epsf = EPSF;
	const real epsx = EPSX;
	const int maxits = MAXITS;
	stpscal = 0.5;
	int info;

	real* x;
	int* nbd;
	real* l;
	real* u;
	memAlloc<real>(&x, dim);
	memAlloc<int>(&nbd, dim);
	memAlloc<real>(&l, dim);
	memAlloc<real>(&u, dim);
	memAllocHost<real>(&f_tb_host, &f_tb_dev, 1);

	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaGLSetGLDevice(0);
	cublasCreate_v2(&cublasHd);

	U = 1;
	J = 0.001;
	mu = 0.5;

	initProb(x, nbd, l, u, dim);
	lbfgsbminimize(dim, 4, x, epsg, epsf, epsx, maxits, nbd, l, u, info);
	printf("info: %d\n", info);

	printf("f: %e\n", *f_tb_host);
	real* x_host = new real[dim];
	memCopy(x_host, x, dim * sizeof(real), cudaMemcpyDeviceToHost);
	printf("x: ");
	for (int i = 0; i < dim; i++) {
		printf("%f, ", x_host[i]);
	}
	printf("\n");

	memFreeHost(f_tb_host);
	memFree(x);
	memFree(nbd);
	memFree(l);
	memFree(u);

	cublasDestroy_v2(cublasHd);

	cudaDeviceReset();

	time_t end = time(NULL);

	printf("Runtime: %ld", end-start);
}
