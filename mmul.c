#include <stdlib.h>
#include <stdio.h>
#include <emmintrin.h>
#include <x86intrin.h>

#define N 1000
double res[N][N] __attribute__ ((aligned (64)));
double mul1[N][N] __attribute__ ((aligned (64)));
double mul2[N][N] __attribute__ ((aligned (64)));
#define SM (CLS / sizeof (double))

unsigned long long readTSC() {
    // _mm_lfence();  // optionally wait for earlier insns to retire before reading the clock
    return __rdtsc();
    // _mm_lfence();  // optionally block later instructions until rdtsc retires
}

int main (void)
{
    long long t = readTSC();
// ... Initialize mul1 and mul2
    int i, i2, j, j2, k, k2;
    double *restrict rres;
    double *restrict rmul1;
    double *restrict rmul2;
    for (i = 0; i < N; i += SM)
        for (j = 0; j < N; j += SM)
            for (k = 0; k < N; k += SM)
                for (i2 = 0, rres = &res[i][j], rmul1 = &mul1[i][k]; i2 < SM;
                     ++i2, rres += N, rmul1 += N)
                {
                    _mm_prefetch (&rmul1[8], _MM_HINT_NTA);
                    for (k2 = 0, rmul2 = &mul2[k][j]; k2 < SM; ++k2, rmul2 += N)
                    {
                        __m128d m1d = _mm_load_sd (&rmul1[k2]);
                        m1d = _mm_unpacklo_pd (m1d, m1d);
                        for (j2 = 0; j2 < SM; j2 += 2)
                        {
                            __m128d m2 = _mm_load_pd (&rmul2[j2]);
                            __m128d r2 = _mm_load_pd (&rres[j2]);
                            _mm_store_pd (&rres[j2],
                                          _mm_add_pd (_mm_mul_pd (m2, m1d), r2));
                        }
                    }
                }

    long long resultcycles = readTSC();
    long long resultcycles2 = resultcycles - t;
    printf("%d\n", resultcycles2);
// ... use res matrix
    return 0;
}
