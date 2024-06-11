#include <iostream>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <mkl.h>
#include <windows.h>

#define MATRIX_SIZE 4096
#define BLOCK_SIZE 32


using namespace std;

float* matrix_create(const int N)
{
    float* matrix = (float*)calloc(N * N, sizeof(float));
    return matrix;
}


void matrix_fill(float* matrix, const int N)
{
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[i * N + j] = (float)rand() / RAND_MAX * 10;
        }
    }
}


void matrix_trans(float* matrixB, float* matrixBT, const int N)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            matrixBT[j * N + i] = matrixB[i * N + j];
        }
    }
}

void linearMatrixMultiplication(const float* matrixA, const float* matrixB, float* matrixC, const int N)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float result = 0.0f;
            for (int k = 0; k < N; k++)
            {
                result += matrixA[i * N + k] * matrixB[j * N + k];
            }
            matrixC[i * N + j] = result;
        }
    }
}

float dot_prod(const float* matrixA, const float* matrixB, const int N)
{
    float s[16] = { 0.0f };

    int n16 = (N / 16) * 16;

    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();

    for (int i = 0; i < n16; i += 16)
    {
        __m256 va1 = _mm256_loadu_ps(&matrixA[i]);
        __m256 vb1 = _mm256_loadu_ps(&matrixB[i]);
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(va1, vb1));

        __m256 va2 = _mm256_loadu_ps(&matrixA[i + 8]);
        __m256 vb2 = _mm256_loadu_ps(&matrixB[i + 8]);
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(va2, vb2));
    }

    _mm256_storeu_ps(&s[0], sum1);
    _mm256_storeu_ps(&s[8], sum2);

    float result = 0.0f;
    for (int i = 0; i < 16; i++)
    {
        result += s[i];
    }

    for (int i = n16; i < N; i++)
    {
        result += matrixA[i] * matrixB[i];
    }

    return result;
}

void matrix_mult_optimization(const float* matrixA, const float* matrixB, float* matrixC, const int N, const int blockSize) {
#pragma omp parallel for num_threads(32)
    for (int ii = 0; ii < N; ii += blockSize)
    {
        for (int jj = 0; jj < N; jj += blockSize)
        {
            for (int i = 0; i < blockSize; ++i)
            {
                for (int j = 0; j < blockSize; ++j)
                {
                    matrixC[(ii + i) * N + (jj + j)] = dot_prod(&matrixA[(ii + i) * N], &matrixB[(jj + j) * N], N);
                }
            }
        }
    }
}

bool is_equal(const float matrixA, const float matrixB, float epsilon = 0.001f) {
    return abs(matrixA - matrixB) < epsilon;
}

bool is_equal_matrix(const float* matrixA, const float* matrixB, int N, float epsilon = 0.001f) {
    for (int i = 0; i < N; ++i) {
        if (!is_equal(matrixA[i], matrixB[i], epsilon)) {
            return false;
        }
    }
    return true;
}

int main()
{
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);

    srand(time(NULL));


    float* matrixA = matrix_create(MATRIX_SIZE);
    float* matrixB = matrix_create(MATRIX_SIZE);
    float* matrixBT = matrix_create(MATRIX_SIZE);
    float* matrixC1 = matrix_create(MATRIX_SIZE);
    float* matrixC2 = matrix_create(MATRIX_SIZE);
    float* matrixC3 = matrix_create(MATRIX_SIZE);

    clock_t start, end;
    double elapsed_secs;

    float alpha(1.0);
    float beta(0.0);

    long long c = 2 * (double)pow(MATRIX_SIZE, 3);

    matrix_fill(matrixA, MATRIX_SIZE);
    matrix_fill(matrixB, MATRIX_SIZE);

    cout << "Выполнил: " << endl;
    cout << "Сложность алгоритма составляет: "<< c << endl;
    cout << "Значения в заполненых ячейках матрицы:\n";
    cout << "Матрица matrixA[15] = " << matrixA[15] << endl;
    cout << "Матрица matrixB[15] = " << matrixB[15] << endl;

    start = clock();
    //linearMatrixMultiplication(matrixA, matrixB, matrixC1, MATRIX_SIZE);
    end = clock();
    elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
    double p1 = c / elapsed_secs * 1.e-6;

    cout << "\nНативный алгоритм умножения матриц: " << elapsed_secs
        << " секунд, производительность: " << p1 << " MFlops\n";
    cout << "Результат умножения матриц matrixA на matrixB:\n";
    cout << "Матрица matrixС1[15] = " << matrixC1[15] << endl << endl;

    start = clock();
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, alpha, matrixA,
        MATRIX_SIZE, matrixB, MATRIX_SIZE, beta, matrixC2, MATRIX_SIZE);
    end = clock();
    elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
    double p2 = c / elapsed_secs * 1.e-6;

    cout << "Умножение матриц используя блиотеку MKL: " << elapsed_secs
        << " секунд, производительность: " << p2 << " MFlops\n";
    cout << "Результат умножения матриц matrixA на matrixB:\n";
    cout << "Матрица matrixС2[15] = " << matrixC2[15] << endl << endl;

    start = clock();
    matrix_trans(matrixB, matrixBT, MATRIX_SIZE);
    matrix_mult_optimization(matrixA, matrixBT, matrixC3, MATRIX_SIZE, BLOCK_SIZE);
    end = clock();
    elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
    double p3 = c / elapsed_secs * 1.e-6;

    cout << "Умножение матриц используя блочный алгоритм: " << elapsed_secs
        << " секунд, производительность: " << p3 << " MFlops\n";
    cout << "Результат умножения матриц matrixA на matrixB:\n";
    cout << "Матрица matrixС3[15] = " << matrixC3[15] << endl << endl;
    cout << "Проверка на равенство матриц:\nМатрицы matrixC2 и matrixС3: "
        << (is_equal_matrix(matrixC2, matrixC3, MATRIX_SIZE, 0.1) ? "равны" : "не равны") << endl;

    cout << "\nСравнение производительности алгоритмов:\n"
        << "Производительность оптимизированного алгоритма относительно функции из MKL составляет: " << (p3 / p2 * 100) << "%" << endl;

    free(matrixA);
    free(matrixB);
    free(matrixBT);
    free(matrixC1);
    free(matrixC2);
    free(matrixC3);

    return 1;
}
