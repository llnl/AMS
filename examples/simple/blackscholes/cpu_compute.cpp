#include <cmath>

#include "constants.hpp"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Cumulative Normal Distribution Function
// See Hull, Section 11.8, P.243-244
//

fptype CNDF(fptype InputX)
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

fptype BlkSchlsEqEuroNoDiv(fptype sptprice,
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


int compute(fptype *sptprice,
            fptype *strike,
            fptype *rate,
            fptype *volatility,
            fptype *otime,
            int *otype,
            fptype *prices,
            size_t numOptions)
{
  int i, j, k;
  fptype priceDelta;
  for (i = 0; i < numOptions; i++) {
    prices[i] = BlkSchlsEqEuroNoDiv(
        sptprice[i], strike[i], rate[i], volatility[i], otime[i], otype[i], 0);
  }

  return 0;
}
