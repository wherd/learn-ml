#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <limits.h>
#include <float.h>

#define WIDTH 50
#define HEIGHT 50
#define PPM_SCALER 25
#define BIAS 10
#define SAMPLE_SIZE 1000
#define TRAIN_PASSES 200

typedef float layer_t[HEIGHT][WIDTH];

static layer_t inputs;
static layer_t weights;

static inline int clampi(int x, int low, int high)
{
	return (x < low ? low : (x > high ? high : x));
}

void layer_save_as_ppm(layer_t layer, const char *filename)
{
	float min = FLT_MAX;
	float max = FLT_MIN;

	for (int y = 0; y < HEIGHT; ++y)
	{
		for (int x = 0; x < WIDTH; ++x)
		{
			if (layer[y][x] < min) min = layer[y][x];
			if (layer[y][x] > max) max = layer[y][x];
		}
	}

	FILE *f = fopen(filename, "wb");
	if (f == NULL)
	{
		fprintf(stderr, "ERROR: could not open file %s: %m\n", filename);
		exit(0);
	}

	fprintf(f, "P6\n%d %d 255\n", WIDTH * PPM_SCALER, HEIGHT * PPM_SCALER);

	for (int y = 0; y < HEIGHT * PPM_SCALER; ++y)
	{
		for (int x = 0; x < WIDTH * PPM_SCALER; ++x)
		{
			float s = (layer[y/PPM_SCALER][x/PPM_SCALER] - min) / (max - min);
			char pixel[3] = {
				(char)floorf(255 * (1.0f - s)),
				(char)floorf(255 * s),
				0
			};
			fwrite(pixel, sizeof(pixel), 1, f);
		}
	}

	fclose(f);
}

void layer_save_as_bin(layer_t layer, const char *filename)
{
	FILE *f = fopen(filename, "wb");
	if (f == NULL)
	{
		fprintf(stderr, "ERROR: could not open file %s: %m", filename);
		exit(0);
	}

	fwrite(layer, sizeof(layer_t), 1, f);
	fclose(f);
}


void layer_fill_rect(layer_t layer, int x, int y, int w, int h, float value)
{
	assert(w > 0);
	assert(h > 0);

	int x0 = clampi(x, 0, WIDTH - 1);
	int y0 = clampi(y, 0, HEIGHT - 1);
	int x1 = clampi(x0 + w - 1, 0, WIDTH - 1);
	int y1 = clampi(y0 + h - 1, 0, HEIGHT - 1);

	for (int y = 0; y <= y1; ++y)
	{
		for (int x = 0; x <= x1; ++x)
		{
			layer[y][x] = value;
		}
	}
}

void layer_fill_circle(layer_t layer, int cx, int cy, int r, float value)
{
	assert(r > 0);

	int x0 = clampi(cx - r, 0, WIDTH - 1);
	int y0 = clampi(cy - r, 0, HEIGHT - 1);
	int x1 = clampi(cx + r, 0, WIDTH - 1);
	int y1 = clampi(cy + r, 0, HEIGHT - 1);

	for (int y = y0; y <= y1; ++y)
	{
		for (int x = x0; x <= x1; ++x)
		{
			int dx = x - cx;
			int dy = y - cy;
			if (dx*dx + dy*dy <= r*r)
			{
				layer[y][x] = value;
			}
		}
	}
}

float forward(layer_t inputs, layer_t weights)
{
	float output = 0.f;

	for (size_t y = 0; y < HEIGHT; ++y)
	{
		for (size_t x = 0; x < WIDTH; ++x)
		{
			output += inputs[y][x] * weights[y][x];
		}
	}

	return output;
}

int rand_range(int low, int high)
{
	assert(low < high);
	return rand() % (high - low) + low;
}

void layer_random_rect(layer_t layer)
{
	int x = rand_range(0, WIDTH);
	int y = rand_range(0, HEIGHT);

	int w = WIDTH - x;
	if (w < 2) w = 2;
	w = rand_range(1, w);

	int h = HEIGHT - y;
	if (h < 2) h = 2;
	h = rand_range(1, h);

	layer_fill_rect(layer, 0, 0, WIDTH, HEIGHT, 0.0f);
	layer_fill_rect(layer, x, y, w, h, 1.0f);
}

void layer_random_circle(layer_t layer)
{
	int cx = rand_range(0, WIDTH);
	int cy = rand_range(0, HEIGHT);

	int r = cx;
	if (r > cy) r = cy;
	if (r > WIDTH - cx) r = WIDTH - cx;
	if (r > HEIGHT - cy) r = HEIGHT - cy;
	if (r < 2) r = 2;
	r = rand_range(1, r);

	layer_fill_rect(layer, 0, 0, WIDTH, HEIGHT, 0.0f);
	layer_fill_circle(layer, cx, cy, r, 1.0f);
}

void add_inputs_to_weights(layer_t inputs, layer_t weights)
{
	for (int y = 0; y < HEIGHT; ++y)
	{
		for (int x = 0; x < WIDTH; ++x)
		{
			weights[y][x] += inputs[y][x];
		}
	}
}

void sub_inputs_from_weights(layer_t inputs, layer_t weights)
{
	for (int y = 0; y < HEIGHT; ++y)
	{
		for (int x = 0; x < WIDTH; ++x)
		{
			weights[y][x] -= inputs[y][x];
		}
	}

}

int train_pass(layer_t inputs, layer_t weights)
{
	int adjusted = 0;

	for (int i=0; i<SAMPLE_SIZE; ++i)
	{
		layer_random_rect(inputs);
		if (forward(inputs, weights) > BIAS)
		{
			sub_inputs_from_weights(inputs, weights);
			adjusted += 1;
		}

		layer_random_circle(inputs);
		if (forward(inputs, weights) < BIAS)
		{
			add_inputs_to_weights(inputs, weights);
			adjusted += 1;
		}
	}

	return adjusted;
}

int check_pass(layer_t inputs, layer_t weights)
{
	int failed = 0;

	for (int i=0; i<SAMPLE_SIZE; ++i)
	{
		layer_random_rect(inputs);
		if (forward(inputs, weights) > BIAS)
		{
			failed += 1;
		}

		layer_random_circle(inputs);
		if (forward(inputs, weights) < BIAS)
		{
			failed += 1;
		}
	}

	return failed;
}


int main(void)
{

	srand(420);
	int failed = check_pass(inputs, weights);
	printf("The fail rate of untrained model is %f\n", failed / (SAMPLE_SIZE * 2.0f));

	for (int i=0; i<TRAIN_PASSES; ++i)
	{
		srand(69);
		int adjusted = train_pass(inputs, weights);
		if (adjusted <= 0) break;
	}

	srand(420);
	failed = check_pass(inputs, weights);
	printf("The fail rate of trained model is %f\n", failed / (SAMPLE_SIZE * 2.0f));

	return 0;
}
