#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float train[][2] = {
	{0, 0},
	{1, 2},
	{2, 4},
	{3, 6},
	{4, 8},
};

#define TRAIN_COUNT (sizeof(train) / sizeof(train[0]))

float rand_float(void)
{
	return (float)rand() / (float)RAND_MAX;
}

float cost(float w, float b)
{
	float mse = 0.0f; // mean squared error
	for (size_t i = 0; i < TRAIN_COUNT; ++i)
	{
		float x = train[i][0];
		float y = (x * w) + b;
		float d = y - train[i][1];
		mse += d * d;
	}

	mse /= TRAIN_COUNT;
	return mse;
}

int main(void)
{
	// y = (x * w) + b

	srand(34); // srand(time(0));
	float w = rand_float();
	float b = rand_float();

	// magical values (https://xkcd.com/1838/)
	float eps = 1e-3;
	float rate = 1e-3;

	for (size_t i = 0; i < 10000; ++i)
	{
		// derivative (https://en.wikipedia.org/wiki/Derivative)
		// finite difference (https://en.wikipedia.org/wiki/Finite_difference)
		float c = cost(w, b);

		float dw = (cost(w + eps, b) - c) / eps;
		float db = (cost(w, b + eps) - c) / eps;

		w -= dw * rate;
		b -= db * rate;
	}

	printf("-----\n");
	printf("w: %f, b: %f\n", w, b);

	for (size_t i = 0; i < TRAIN_COUNT; ++i)
	{
		float y = (train[i][0] * w) + b;
		printf("got: %f, wanted: %f\n", y, train[i][1]);
	}
	return 0;
}