# Load required libraries
library(mrgsolve)
library(ggplot2)
library(dplyr)
library(dmutate)


mod <- mread('pkpd_covariates')

#mod <- mread("pkpd_covariates") %>%
#  omat(list(cmat(0.1, 0.8, 0.3),dmat(0.5)))
#mod

set.seed(20202)
idata <- 
  tibble(ID=1:300) %>% 
  mutate_random(WT[50,110] ~ rnorm(70,30))

head(idata)

out <- mod %>% param(TLAG=0) %>%
  ev(amt=100, ii=24, addl=4) %>%
  idata_set(idata) %>%
  mrgsim(delta=0.1, end=300)

head(out)
plot(out, CP + RESPONSE~.,)

out_df = as.data.frame(out)
covariates_df = as.data.frame(idata)
write.csv(covariates_df, 'Covariates_WT.csv')
write.csv(out_df, 'PKPD_Covariates_WT.csv')
