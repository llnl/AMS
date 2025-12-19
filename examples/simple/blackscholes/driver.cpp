// Copyright (c) 2007 Intel Corp.

// Black-Scholes
// Analytical method for calculating European Options
//
//
// Reference Source: Options, Futures, and Other Derivatives, 3rd Edition, Prentice
// Hall, John C. Hull,

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <cassert>
#include <cmath>
#include <iostream>

#include "constants.hpp"
#include "profile.hpp"

#define DOUBLE 0
#define FLOAT 1
#define INT 2

int compute(fptype *sptprice,
            fptype *strike,
            fptype *rate,
            fptype *volatility,
            fptype *otime,
            int *otype,
            fptype *prices,
            size_t numOptions);

void readData(FILE *fd, double **data, size_t *numElements)
{
  assert(fd && "File pointer is not valid\n");
  fread(numElements, sizeof(size_t), 1, fd);
  size_t elements = *numElements;
  double *ptr = (double *)malloc(sizeof(double) * elements);
  assert(ptr && "Could Not allocate pointer\n");
  *data = ptr;
  size_t i;
  int type;
  fread(&type, sizeof(int), 1, fd);
  if (type == DOUBLE) {
    fread(ptr, sizeof(double), elements, fd);
  } else if (type == FLOAT) {
    float *tmp = (float *)malloc(sizeof(float) * elements);
    fread(tmp, sizeof(float), elements, fd);
    for (i = 0; i < elements; i++) {
      ptr[i] = (double)tmp[i];
    }
    free(tmp);
  } else if (type == INT) {
    int *tmp = (int *)malloc(sizeof(int) * elements);
    fread(tmp, sizeof(int), elements, fd);
    for (i = 0; i < elements; i++) {
      ptr[i] = (double)tmp[i];
    }
    free(tmp);
  }
  return;
}

void readData(FILE *fd, float **data, size_t *numElements)
{
  assert(fd && "File pointer is not valid\n");
  fread(numElements, sizeof(size_t), 1, fd);
  size_t elements = *numElements;

  float *ptr = (float *)malloc(sizeof(float) * elements);
  assert(ptr && "Could Not allocate pointer\n");
  *data = ptr;

  size_t i;
  int type;
  fread(&type, sizeof(int), 1, fd);
  if (type == FLOAT) {
    fread(ptr, sizeof(float), elements, fd);
  } else if (type == DOUBLE) {
    double *tmp = (double *)malloc(sizeof(double) * elements);
    fread(tmp, sizeof(double), elements, fd);
    for (i = 0; i < elements; i++) {
      ptr[i] = (float)tmp[i];
    }
    free(tmp);
  } else if (type == INT) {
    int *tmp = (int *)malloc(sizeof(int) * elements);
    fread(tmp, sizeof(int), elements, fd);
    for (i = 0; i < elements; i++) {
      ptr[i] = (float)tmp[i];
    }
    free(tmp);
  }
  return;
}

void readData(FILE *fd, int **data, size_t *numElements)
{
  assert(fd && "File pointer is not valid\n");
  fread(numElements, sizeof(size_t), 1, fd);
  size_t elements = *numElements;

  int *ptr = (int *)malloc(sizeof(int) * elements);
  assert(ptr && "Could Not allocate pointer\n");
  *data = ptr;

  size_t i;
  int type;
  fread(&type, sizeof(int), 1, fd);
  if (type == INT) {
    fread(ptr, sizeof(int), elements, fd);
  } else if (type == DOUBLE) {
    double *tmp = (double *)malloc(sizeof(double) * elements);
    fread(tmp, sizeof(double), elements, fd);
    for (i = 0; i < elements; i++) {
      ptr[i] = (int)tmp[i];
    }
    free(tmp);
  } else if (type == FLOAT) {
    float *tmp = (float *)malloc(sizeof(float) * elements);
    fread(tmp, sizeof(float), elements, fd);
    for (i = 0; i < elements; i++) {
      ptr[i] = (int)tmp[i];
    }
    free(tmp);
  }
  return;
}

int main(int argc, char **argv)
{

  fptype *prices;
  size_t numOptions;

  int *otype;
  fptype *sptprice;
  fptype *strike;
  fptype *rate;
  fptype *volatility;
  fptype *otime;

  FILE *file;
  int i;
  int loopnum;
  int rv;

  fflush(NULL);
  if (argc != 2) {
    printf("Usage:\n\t%s <inputFile>\n", argv[0]);
    exit(1);
  }
  char *inputFile = argv[1];
  char *outputFile = argv[2];

  //Read input data from file
  file = fopen(inputFile, "rb");
  if (file == NULL) {
    printf("ERROR: Unable to open file `%s'.\n", inputFile);
    exit(1);
  }
#define PAD 256
#define LINESIZE 64
  readData(file, &otype, &numOptions);
  readData(file, &sptprice, &numOptions);
  readData(file, &strike, &numOptions);
  readData(file, &rate, &numOptions);
  readData(file, &volatility, &numOptions);
  readData(file, &otime, &numOptions);
  prices = (fptype *)malloc(sizeof(fptype) * numOptions);

  {
    std::cout << "Total NumOptions Computed:" << numOptions << "\n";
    ScopedTimer<size_t> t("compute", numOptions);
    compute(
        sptprice, strike, rate, volatility, otime, otype, prices, numOptions);
  }

  free(sptprice);
  free(strike);
  free(rate);
  free(volatility);
  free(otime);
  free(otype);
  free(prices);

  return 0;
}
