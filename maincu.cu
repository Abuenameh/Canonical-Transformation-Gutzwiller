#include "L-BFGS/cutil_inline.h"
#include "L-BFGS/lbfgsbcuda.h"
#include "L-BFGS/lbfgsb.h"

__device__ double atomicAdd(double* address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*) address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__global__ void initProbKer(real* x, int* nbd, real* l, real* u, int dim) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= dim)
		return;

	x[i] = 0.5;
	nbd[i] = 0;

}

extern void initProb(real* x, int* nbd, real* l, real* u, int dim) {
	initProbKer<<<lbfgsbcuda::iDivUp(dim, 64), 64>>>(x, nbd, l, u, dim);
}

__global__ void energyKer(real* x, real* f, real* g, int L, int nmax, real U,
		real J, real mu, real* norm2, real norm2s) {

	if(threadIdx.x > 0) {
		return;
	}

	const int i = blockIdx.x;
	if (i >= L) {
		return;
	}

	real os = 0;
	real hop1 = 0;
	real hop2 = 0;
	for (int n = 0; n <= nmax; n++) {
		int k = i * (nmax + 1) + n;
		os = (0.5 * U * n * (n - 1) * x[k] * x[k] - mu * n * x[k] * x[k])
				* norm2s / norm2[i];
		atomicAdd(f, os);

		atomicAdd(&g[k],
				(U * n * (n - 1) * x[k] - 2 * mu * n * x[k]) * norm2s
						/ norm2[i]);
		for (int j = 0; j < L; j++) {
			if (i != j) {
				for (int m = 0; m <= nmax; m++) {
					int l = j * (nmax + 1) + m;
					atomicAdd(&g[l], os * 2 * x[l] / norm2[j]);
				}
			}
		}
	}

	int i1 = (i == 0) ? L - 1 : i - 1;
	int i2 = (i == L - 1) ? 0 : i + 1;
	for (int n = 0; n < nmax; n++) {
		int k = i * (nmax + 1) + n;
		for (int m = 0; m < nmax; m++) {
			int k1 = i1 * (nmax + 1) + m;
			int k2 = i2 * (nmax + 1) + m;
			hop1 = -J * sqrt(1.0 * (n + 1) * (m + 1)) * x[k + 1] * x[k] * x[k1]
					* x[k1 + 1] * norm2s / (norm2[i] * norm2[i1]);
			hop2 = -J * sqrt(1.0 * (n + 1) * (m + 1)) * x[k + 1] * x[k] * x[k2]
					* x[k2 + 1] * norm2s / (norm2[i] * norm2[i2]);
			atomicAdd(f, hop1 + hop2);

			atomicAdd(&g[k],
					-J * sqrt(1.0 * (n + 1) * (m + 1)) * x[k + 1] * x[k1]
							* x[k1 + 1] * norm2s / (norm2[i] * norm2[i1]));
			atomicAdd(&g[k],
					-J * sqrt(1.0 * (n + 1) * (m + 1)) * x[k + 1] * x[k2]
							* x[k2 + 1] * norm2s / (norm2[i] * norm2[i2]));
			atomicAdd(&g[k + 1],
					-J * sqrt(1.0 * (n + 1) * (m + 1)) * x[k] * x[k1]
							* x[k1 + 1] * norm2s / (norm2[i] * norm2[i1]));
			atomicAdd(&g[k + 1],
					-J * sqrt(1.0 * (n + 1) * (m + 1)) * x[k] * x[k2]
							* x[k2 + 1] * norm2s / (norm2[i] * norm2[i2]));
			atomicAdd(&g[k1],
					-J * sqrt(1.0 * (n + 1) * (m + 1)) * x[k + 1] * x[k]
							* x[k1 + 1] * norm2s / (norm2[i] * norm2[i1]));
			atomicAdd(&g[k2],
					-J * sqrt(1.0 * (n + 1) * (m + 1)) * x[k + 1] * x[k]
							* x[k2 + 1] * norm2s / (norm2[i] * norm2[i2]));
			atomicAdd(&g[k1 + 1],
					-J * sqrt(1.0 * (n + 1) * (m + 1)) * x[k + 1] * x[k] * x[k1]
							* norm2s / (norm2[i] * norm2[i1]));
			atomicAdd(&g[k2 + 1],
					-J * sqrt(1.0 * (n + 1) * (m + 1)) * x[k + 1] * x[k] * x[k2]
							* norm2s / (norm2[i] * norm2[i2]));
			for (int q = 0; q < L; q++) {
				if (q != i) {
					for (int p = 0; p <= nmax; p++) {
						int l = q * (nmax + 1) + p;
						if (q != i1) {
							atomicAdd(&g[l], hop1 * 2 * x[l] / norm2[q]);
						}
						if (q != i2) {
							atomicAdd(&g[l], hop2 * 2 * x[l] / norm2[q]);
						}
					}
				}
			}
		}
	}

}

extern void energy(real* x, real* f, real* g, int L, int nmax, real U, real J,
		real mu, const cudaStream_t& stream) {
//	printf("Energy 1: %ld\n", time(NULL));

	int dim = L * (nmax + 1);

	real f_host = 0;
	real* g_host = new real[dim];
	for (int k = 0; k < dim; k++) {
		g_host[k] = 0;
	}
	memCopy(f, &f_host, sizeof(real), cudaMemcpyHostToDevice);
	memCopy(g, g_host, dim * sizeof(real), cudaMemcpyHostToDevice);

	real* x_host = new real[dim];
	memCopy(x_host, x, dim * sizeof(real), cudaMemcpyDeviceToHost);

	real* norm2_host = new real[L];
	real norm2s = 1;
	for (int j = 0; j < L; j++) {
		norm2_host[j] = 0;
		for (int n = 0; n <= nmax; n++) {
			int k = j * (nmax + 1) + n;
			norm2_host[j] += x_host[k] * x_host[k];
		}
		norm2s *= norm2_host[j];
	}
	real* norm2;
	memAlloc<real>(&norm2, L * sizeof(real));
	memCopy(norm2, norm2_host, L * sizeof(real), cudaMemcpyHostToDevice);

	energyKer<<<L, 1>>>(x, f, g, L, nmax, U, J, mu, norm2, norm2s);
	cutilSafeCall(cudaDeviceSynchronize());

	memCopy(&f_host, f, sizeof(real), cudaMemcpyDeviceToHost);
	memCopy(g_host, g, dim * sizeof(real), cudaMemcpyDeviceToHost);

	for (int i = 0; i < L; i++) {
		for (int n = 0; n <= nmax; n++) {
			int k = i * (nmax + 1) + n;
			g_host[k] = (g_host[k] * norm2s
					- f_host * 2 * x_host[k] * norm2s / norm2_host[i])
					/ (norm2s * norm2s);
		}
	}
	f_host /= norm2s;

	memCopy(f, &f_host, sizeof(real), cudaMemcpyHostToDevice);
	memCopy(g, g_host, dim * sizeof(real), cudaMemcpyHostToDevice);

//	printf("Energy 2: %ld\n", time(NULL));

}
