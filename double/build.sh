#!/bin/sh

set -xe
clang -Wall -Wextra main.c -o main -lm
