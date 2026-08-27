#include <cmath>

#include "constants.hpp"
#include "device_traits.hpp"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Cumulative Normal Distribution Function
// See Hull, Section 11.8, P.243-244

__device__ fptype CNDF(fptype InputX)
{
  int sign;

  fptype OutputX;
  fptype xInput;
  fptype xNPrimeofX;
  fptype expValues;
  fptype xK2;
  fptype xK2_2, xK2_3;
  fptype xK2_4, xK2_5;
  fptype xLocal, xLocal_1;
  fptype xLocal_2, xLocal_3;
  fptype temp;

  // Check for negative value of InputX
  if (InputX < zero) {
    InputX = -InputX;
    sign = 1;
  } else
    sign = 0;

  xInput = InputX;

  // Compute NPrimeX term common to both four & six decimal accuracy calcs
  temp = -half * InputX * InputX;

  expValues = exp(temp);

  xNPrimeofX = expValues;
  xNPrimeofX = xNPrimeofX * inv_sqrt_2xPI;

  xK2 = const1 * xInput;
  xK2 = one + xK2;
  xK2 = one / xK2;
  xK2_2 = xK2 * xK2;
  xK2_3 = xK2_2 * xK2;
  xK2_4 = xK2_3 * xK2;
  xK2_5 = xK2_4 * xK2;

  xLocal_1 = xK2 * const2;
  xLocal_2 = xK2_2 * (-const3);
  xLocal_3 = xK2_3 * const4;
  xLocal_2 = xLocal_2 + xLocal_3;
  xLocal_3 = xK2_4 * (-const5);
  xLocal_2 = xLocal_2 + xLocal_3;
  xLocal_3 = xK2_5 * const6;
  xLocal_2 = xLocal_2 + xLocal_3;

  xLocal_1 = xLocal_2 + xLocal_1;
  xLocal = xLocal_1 * xNPrimeofX;
  xLocal = one - xLocal;

  OutputX = xLocal;

  if (sign) {
    OutputX = one - OutputX;
  }

  return OutputX;
}

__device__ fptype BlkSchlsEqEuroNoDiv(fptype sptprice,
                                      fptype strike,
                                      fptype rate,
                                      fptype volatility,
                                      fptype time,
                                      int otype,
                                      float timet)
{
  fptype OptionPrice;

  // local private working variables for the calculation
  fptype xStockPrice;
  fptype xStrikePrice;
  fptype xRiskFreeRate;
  fptype xVolatility;

  fptype xTime;
  fptype xSqrtTime;

  fptype logValues;
  fptype xLogTerm;
  fptype xD1;
  fptype xD2;
  fptype xPowerTerm;
  fptype xDen;
  fptype d1;
  fptype d2;
  fptype FutureValueX;
  fptype NofXd1;
  fptype NofXd2;
  fptype NegNofXd1;
  fptype NegNofXd2;
  fptype temp;

  xStockPrice = sptprice;
  xStrikePrice = strike;
  xRiskFreeRate = rate;
  xVolatility = volatility;

  xTime = time;
  xSqrtTime = sqrt(xTime);

  temp = sptprice / strike;

  logValues = log(sptprice / strike);

  xLogTerm = logValues;


  xPowerTerm = xVolatility * xVolatility;
  xPowerTerm = xPowerTerm * half;

  xD1 = xRiskFreeRate + xPowerTerm;
  xD1 = xD1 * xTime;
  xD1 = xD1 + xLogTerm;

  xDen = xVolatility * xSqrtTime;
  xD1 = xD1 / xDen;
  xD2 = xD1 - xDen;

  d1 = xD1;
  d2 = xD2;

  //@APPROX LABEL("CNDF_1") IN(d1) OUT(NofXd1) APPROX_TECH(MEMO_IN|MEMO_OUT)
  NofXd1 = CNDF(d1);

  //@APPROX LABEL("CNDF_2") IN(d2) OUT(NofXd2) APPROX_TECH(MEMO_IN|MEMO_OUT)
  NofXd2 = CNDF(d2);

  temp = -(rate * time);

  FutureValueX = (exp(temp));

  FutureValueX *= strike;

  if (otype == 0) {
    OptionPrice = (sptprice * NofXd1) - (FutureValueX * NofXd2);
  } else {
    NegNofXd1 = (one - NofXd1);
    NegNofXd2 = (one - NofXd2);
    OptionPrice = (FutureValueX * NegNofXd2) - (sptprice * NegNofXd1);
  }

  return OptionPrice;
}

__global__ void gpu_run(fptype *sptprice,
                        fptype *strike,
                        fptype *rate,
                        fptype *volatility,
                        fptype *otime,
                        int *otype,
                        fptype *prices,
                        size_t numOptions)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid > numOptions) return;
  prices[gid] = BlkSchlsEqEuroNoDiv(sptprice[gid],
                                    strike[gid],
                                    rate[gid],
                                    volatility[gid],
                                    otime[gid],
                                    otype[gid],
                                    0);
}


int compute(fptype *sptprice,
            fptype *strike,
            fptype *rate,
            fptype *volatility,
            fptype *otime,
            int *otype,
            fptype *prices,
            size_t numOptions)
{
  constexpr int BLOCK_SIZE = 256;
  fptype *d_sptprice;
  fptype *d_strike;
  fptype *d_rate;
  fptype *d_volatility;
  fptype *d_otime;
  int *d_otype;
  fptype *d_prices;

  DEVICE_CHECK((Device::deviceMalloc(reinterpret_cast<void **>(&d_sptprice),
                                     numOptions * sizeof(fptype))));
  DEVICE_CHECK(Device::deviceMalloc(reinterpret_cast<void **>(&d_strike),
                                    numOptions * sizeof(fptype)));
  DEVICE_CHECK(Device::deviceMalloc(reinterpret_cast<void **>(&d_rate),
                                    numOptions * sizeof(fptype)));
  DEVICE_CHECK(Device::deviceMalloc(reinterpret_cast<void **>(&d_volatility),
                                    numOptions * sizeof(fptype)));
  DEVICE_CHECK(Device::deviceMalloc(reinterpret_cast<void **>(&d_otime),
                                    numOptions * sizeof(fptype)));
  DEVICE_CHECK(Device::deviceMalloc(reinterpret_cast<void **>(&d_otype),
                                    numOptions * sizeof(int)));
  DEVICE_CHECK(Device::deviceMalloc(reinterpret_cast<void **>(&d_prices),
                                    numOptions * sizeof(fptype)));

  DEVICE_CHECK(Device::deviceCopy(reinterpret_cast<void *>(d_sptprice),
                                  sptprice,
                                  numOptions * sizeof(fptype),
                                  Device::memcpyHostToDeviceKind()));

  DEVICE_CHECK(Device::deviceCopy(reinterpret_cast<void *>(d_strike),
                                  strike,
                                  numOptions * sizeof(fptype),
                                  Device::memcpyHostToDeviceKind()));

  DEVICE_CHECK(Device::deviceCopy(reinterpret_cast<void *>(d_rate),
                                  rate,
                                  numOptions * sizeof(fptype),
                                  Device::memcpyHostToDeviceKind()));

  DEVICE_CHECK(Device::deviceCopy(reinterpret_cast<void *>(d_volatility),
                                  volatility,
                                  numOptions * sizeof(fptype),
                                  Device::memcpyHostToDeviceKind()));
  DEVICE_CHECK(Device::deviceCopy(reinterpret_cast<void *>(d_otime),
                                  otime,
                                  numOptions * sizeof(fptype),
                                  Device::memcpyHostToDeviceKind()));

  DEVICE_CHECK(Device::deviceCopy(reinterpret_cast<void *>(d_otype),
                                  otype,
                                  numOptions * sizeof(int),
                                  Device::memcpyHostToDeviceKind()));


  int grid = (numOptions + BLOCK_SIZE - 1) / BLOCK_SIZE;
  gpu_run<<<grid, BLOCK_SIZE>>>(d_sptprice,
                                d_strike,
                                d_rate,
                                d_volatility,
                                d_otime,
                                d_otype,
                                d_prices,
                                numOptions);
  DEVICE_CHECK(Device::deviceCopy(reinterpret_cast<void *>(prices),
                                  d_prices,
                                  numOptions * sizeof(fptype),
                                  Device::memcpyDeviceToHostKind()));

  return 0;
}
