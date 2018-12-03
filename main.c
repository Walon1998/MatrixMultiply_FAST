//Compile with -DCLS=$(getconf LEVEL1_DCACHE_LINESIZE)
#include <stdlib.h>
#include <stdio.h>
#include <emmintrin.h>
#include <assert.h>
#include <x86intrin.h>

//#define N 10
//double res[N][N] __attribute__ ((aligned (64)));
//double mul1[N][N] __attribute__ ((aligned (64)));
//double mul2[N][N] __attribute__ ((aligned (64)));


#define SM (CLS / sizeof (double))

static void printresult(double *A, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
//            printf("%g ", A[i * size + j]);
        }
//        printf("\n");
    }
}

static void fast_multiply(double *A, double *B, double *result, int size) {

    int i, i2, j, j2, k, k2;
    double *restrict rres;
    double *restrict rmul1;
    double *restrict rmul2;
    for (i = 0; i < size; i += SM)
        for (j = 0; j < size; j += SM)
            for (k = 0; k < size; k += SM)
                for (i2 = 0, rres = &result[i * size + j], rmul1 = &A[i * size + k]; i2 < SM;
                     ++i2, rres += size, rmul1 += size) {
                    _mm_prefetch (&rmul1[8], _MM_HINT_NTA);
                    for (k2 = 0, rmul2 = &B[k * size + j]; k2 < SM; ++k2, rmul2 += size) {
                        __m128d m1d = _mm_load_sd(&rmul1[k2]);
                        m1d = _mm_unpacklo_pd(m1d, m1d);
                        for (j2 = 0; j2 < SM; j2 += 2) {
                            __m128d m2 = _mm_load_pd(&rmul2[j2]);
                            __m128d r2 = _mm_load_pd(&rres[j2]);
                            _mm_store_pd(&rres[j2],
                                         _mm_add_pd(_mm_mul_pd(m2, m1d), r2));
                        }
                    }
                }

}

static double *randmatrix(int size) {
    double *matrix = malloc(size * size * sizeof(double));
    assert(matrix != NULL);

    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            matrix[row * size + col] = random() / (RAND_MAX / 100);
        }
    }

    return matrix;
}

unsigned long long readTSC() {
    // _mm_lfence();  // optionally wait for earlier insns to retire before reading the clock
    return __rdtsc();
    // _mm_lfence();  // optionally block later instructions until rdtsc retires
}

static double *mmul(int size, double *A, double *B) {
    double *result = malloc(size * size * sizeof(int));
    assert(result != NULL);

    long long t = readTSC();

    fast_multiply(A, B, result, size);


    long long resultcycles = readTSC();
    long long resultcycles2 = resultcycles - t;

    if (resultcycles2 < 0) {
        printf("0\n");
    } else {
        printf("%d\n", resultcycles2);
        printf("%d\n", resultcycles2);
        printf("%d\n", resultcycles2);
        printf("%d\n", resultcycles2);
    }


    return result;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("USAGE: mmul <matrix_size>\n");
        return 1;
    }
    int size = atoi(argv[1]);

    for (int i = 320; i < size; i += 64) {
//        printf("%d: ", i);


        double *A = randmatrix(i);
        //printmatrix(size, A, 'A');

        double *B = randmatrix(i);
        //printmatrix(size, B, 'B');


        double *C = mmul(i, A, B);


        // the result needs to be used, otherwise C will not be computed
        // redirect the output with ./mmul 5000 > output
        printresult(C, i);

        free(A);
        free(B);
        free(C);
    }

    return 0;
}