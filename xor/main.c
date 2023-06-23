#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef float sample_t[3];

typedef struct
{
	float or_w1;
	float or_w2;
	float or_b;

	float nand_w1;
	float nand_w2;
	float nand_b;

	float and_w1;
	float and_w2;
	float and_b;
} xor_t;

sample_t or_gate[] = {
	{0, 0, 0},
	{1, 0, 1},
	{0, 1, 1},
	{1, 1, 1},
};

sample_t and_gate[] = {
	{0, 0, 0},
	{1, 0, 0},
	{0, 1, 0},
	{1, 1, 1},
};

sample_t nand_gate[] = {
	{0, 0, 1},
	{1, 0, 1},
	{0, 1, 1},
	{1, 1, 0},
};

sample_t xor_gate[] = {
	// (x|y) & ~(x&y)
	{0, 0, 0},
	{1, 0, 1},
	{0, 1, 1},
	{1, 1, 0},
};

sample_t *train = xor_gate;

#define TRAIN_COUNT 4

float rand_float(void)
{
	return (float)rand() / (float)RAND_MAX;
}

xor_t rand_xor(void)
{
	xor_t m;
	m.or_w1 = rand_float();
	m.or_w2 = rand_float();
	m.or_b = rand_float();
	m.nand_w1 = rand_float();
	m.nand_w2 = rand_float();
	m.nand_b = rand_float();
	m.and_w1 = rand_float();
	m.and_w2 = rand_float();
	m.and_b = rand_float();

	return m;
}

void print_xor(xor_t m)
{
	printf("or_w1 = %f\n", m.or_w1);
	printf("or_w2 = %f\n", m.or_w2);
	printf("or_b = %f\n", m.or_b);
	printf("nand_w1 = %f\n", m.nand_w1);
	printf("nand_w2 = %f\n", m.nand_w2);
	printf("nand_b = %f\n", m.nand_b);
	printf("and_w1 = %f\n", m.and_w1);
	printf("and_w2 = %f\n", m.and_w2);
	printf("and_b = %f\n", m.and_b);
}

float sigmoidf(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

float forward(xor_t m, float x1, float x2)
{
	float a = sigmoidf(m.or_w1 * x1 + m.or_w2 * x2 + m.or_b);
	float b = sigmoidf(m.nand_w1 * x1 + m.nand_w2 * x2 + m.nand_b);
	return sigmoidf(a * m.and_w1 + b * m.and_w2 + m.and_b);
}

float cost(xor_t m)
{
	float mse = 0.0f;
	for (size_t i = 0; i < TRAIN_COUNT; ++i)
	{
		float x1 = train[i][0];
		float x2 = train[i][1];
		float y = forward(m, x1, x2);
		float d = y - train[i][2];
		mse += d * d;
	}

	mse /= TRAIN_COUNT;
	return mse;
}

xor_t finite_diff(xor_t m, float eps)
{
	xor_t g;

	float saved;
	float c = cost(m);

	saved = m.or_w1;
	m.or_w1 += eps;
	g.or_w1 = (cost(m) - c) / eps;
	m.or_w1 = saved;

	saved = m.or_w2;
	m.or_w2 += eps;
	g.or_w2 = (cost(m) - c) / eps;
	m.or_w2 = saved;

	saved = m.or_b;
	m.or_b += eps;
	g.or_b = (cost(m) - c) / eps;
	m.or_b = saved;

	saved = m.nand_w1;
	m.nand_w1 += eps;
	g.nand_w1 = (cost(m) - c) / eps;
	m.nand_w1 = saved;

	saved = m.nand_w2;
	m.nand_w2 += eps;
	g.nand_w2 = (cost(m) - c) / eps;
	m.nand_w2 = saved;

	saved = m.nand_b;
	m.nand_b += eps;
	g.nand_b = (cost(m) - c) / eps;
	m.nand_b = saved;

	saved = m.and_w1;
	m.and_w1 += eps;
	g.and_w1 = (cost(m) - c) / eps;
	m.and_w1 = saved;

	saved = m.and_w2;
	m.and_w2 += eps;
	g.and_w2 = (cost(m) - c) / eps;
	m.and_w2 = saved;

	saved = m.and_b;
	m.and_b += eps;
	g.and_b = (cost(m) - c) / eps;
	m.and_b = saved;

	return g;
}

xor_t learn(xor_t m, xor_t g, float rate)
{
	m.or_w1 -= rate * g.or_w1;
	m.or_w2 -= rate * g.or_w2;
	m.or_b -= rate * g.or_b;
	m.nand_w1 -= rate * g.nand_w1;
	m.nand_w2 -= rate * g.nand_w2;
	m.nand_b -= rate * g.nand_b;
	m.and_w1 -= rate * g.and_w1;
	m.and_w2 -= rate * g.and_w2;
	m.and_b -= rate * g.and_b;

	return m;
}

int main(void)
{
	srand(time(0));
	xor_t m = rand_xor();

	float eps = 1e-1;
	float rate = 1e-1;

	for (size_t i = 0; i < 100000; ++i)
	{
		xor_t g = finite_diff(m, eps);
		m = learn(m, g, rate);
	}

	printf("c = %f\n", cost(m));

	for (size_t i = 0; i < 2; ++i)
	{
		for (size_t j = 0; j < 2; ++j)
		{
			printf("%zu ^ %zu = %f\n", i, j, forward(m, i, j));
		}
	}

	printf("-----\n\"OR\" gate\n");
	for (size_t i = 0; i < 2; ++i)
	{
		for (size_t j = 0; j < 2; ++j)
		{
			printf("%zu | %zu = %f\n", i, j, sigmoidf(m.or_w1 * i + m.or_w2 * j + m.or_b));
		}
	}

	printf("-----\n\"NAND\" gate\n");
	for (size_t i = 0; i < 2; ++i)
	{
		for (size_t j = 0; j < 2; ++j)
		{
			printf("~(%zu & %zu) = %f\n", i, j, sigmoidf(m.nand_w1 * i + m.nand_w2 * j + m.nand_b));
		}
	}

	printf("-----\n\"AND\" gate\n");
	for (size_t i = 0; i < 2; ++i)
	{
		for (size_t j = 0; j < 2; ++j)
		{
			printf("%zu & %zu = %f\n", i, j, sigmoidf(m.and_w1 * i + m.and_w2 * j + m.nand_b));
		}
	}

	return 0;
}