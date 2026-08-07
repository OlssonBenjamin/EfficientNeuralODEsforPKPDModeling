# Load required libraries
library(mrgsolve)
library(ggplot2)
library(dplyr)
library(dmutate)

mod <- mread('PKPD R and CPP files/pkpd_covariates')
set.seed(1000)
#mod <- mread("PKPD R and CPP files/pkpd_covariates") %>%
#  omat(list(cmat(0.1, 0.8, 0.3),dmat(0.5)))
#mod

rtnorm <- function(n, mu, sigma, lo, hi) {
  x <- numeric(n)
  i <- 0
  while (i < n) {
    s <- rnorm(1, mu, sigma)
    if (s >= lo && s <= hi) { i <- i + 1; x[i] <- s }
  }
  x
}

idata <- tibble(ID=1:100) %>%
  mutate(WT  = rtnorm(n(), 70, 15, 40, 120)) %>%
  mutate(AGE = rtnorm(n(), 40, 10, 20, 80)) %>%
  mutate(SEX = rbinom(n(), 1, 0.5))


out <- mod %>% param(TLAG=0) %>%
  ev(amt=75, ii=24, addl=14) %>%
  idata_set(idata) %>%
  mrgsim(delta=0.1, end=408)

head(out)
plot(out, CP + RESPONSE~.,)

out_df = as.data.frame(out)
covariates_df = as.data.frame(idata)
write.csv(covariates_df, 'Num Dose Data/75_24_15.csv')
write.csv(out_df, 'Num Dose Data/PKPD_75_24_15.csv')

