#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef float sample[3];

sample or_gate[] = {
	{0, 0, 0},
	{1, 0, 1},
	{0, 1, 1},
	{1, 1, 1},
};

sample and_gate[] = {
	{0, 0, 0},
	{1, 0, 0},
	{0, 1, 0},
	{1, 1, 1},
};

sample nand_gate[] = {
	{0, 0, 1},
	{1, 0, 1},
	{0, 1, 1},
	{1, 1, 0},
};

sample xor_gate[] = {
	{0, 0, 0},
	{1, 0, 1},
	{0, 1, 1},
	{1, 1, 0},
};

sample *train = or_gate;

#define TRAIN_COUNT (sizeof(or_gate) / sizeof(or_gate[0]))

float rand_float(void)
{
	return (float)rand() / (float)RAND_MAX;
}

// activation function
float sigmoidf(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

float cost(float w1, float w2, float b)
{
	float mse = 0.0f; // mean squared error
	for (size_t i = 0; i < TRAIN_COUNT; ++i)
	{
		float x1 = train[i][0];
		float x2 = train[i][1];
		float y = sigmoidf((x1 * w1) + (x2 * w2) + b);
		float d = y - train[i][2];
		mse += d * d;
	}

	mse /= TRAIN_COUNT;
	return mse;
}

int main(void)
{
	srand(time(0));
	float w1 = rand_float();
	float w2 = rand_float();
	float b = rand_float();

	float eps = 1e-1;
	float rate = 1e-1;

	for (size_t i = 0; i < 10000; ++i)
	{
		float c = cost(w1, w2, b);
		printf("w1: %f, w2: %f, b: %f, c: %f\n", w1, w2, b, c);

		float dw1 = (cost(w1 + eps, w2, b) - c) / eps;
		float dw2 = (cost(w1, w2 + eps, b) - c) / eps;
		float db = (cost(w1, w2, b + eps) - c) / eps;

		w1 -= dw1 * rate;
		w2 -= dw2 * rate;
		b -= db * rate;
	}

	printf("-----\n");
	printf("w1: %f, w2: %f, b: %f, c: %f\n", w1, w2, b, cost(w1, w2, b));

	for (size_t i = 0; i < 2; ++i)
	{
		for (size_t j = 0; j < 2; ++j)
		{
			printf("%zu | %zu = %f\n", i, j, sigmoidf((i * w1) + (j * w2) + b));
		}
	}
	return 0;
}