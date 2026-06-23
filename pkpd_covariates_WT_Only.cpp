$PROB
#A simple model to get started.
#Date: September 24, 2015

$PARAM 
KA=0.5, CL=1, VC=10
KIN=100, KOUT=2, IC50=2, FBIO=0.92, TLAG=0
WT = 70, SEX = 0, AGE = 40

$CMT GUT CENT RESPONSE

$SET end=36, delta=0.5

$GLOBAL
#define CP (CENT/VCi)
#define INH (CP/(CP+IC50))

$MAIN

RESPONSE_0 = KIN/KOUT;
double CLi = CL*exp(ETA(1)) * pow(WT/70, 0.75);
double VCi = VC*exp(ETA(2)) * (WT/70);
double KAi = KA*exp(ETA(3));
_ALAG(1) = TLAG; _F(1) = FBIO;
$OMEGA 0 0
$OMEGA 0
$SIGMA 0
$ODE
dxdt_GUT = -KA*GUT;
dxdt_CENT = KA*GUT - (CLi/VCi)*CENT;
dxdt_RESPONSE = KIN*(1-INH) - KOUT*RESPONSE;
$CAPTURE CP
$TABLE 
double GOAL = (RESPONSE<25);
$CAPTURE GOAL
