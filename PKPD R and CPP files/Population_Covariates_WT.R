# Load required libraries
library(mrgsolve)
library(ggplot2)
library(dplyr)
library(dmutate)


mod <- mread('PKPD R and CPP files/pkpd_covariates_WT_Only')
set.seed(12)
mod <- mread("PKPD R and CPP files/pkpd_covariates_WT_Only") %>%
  omat(list(cmat(0.1, 0.8, 0.3),dmat(0.5)))
mod

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

head(idata)

out <- mod %>% param(TLAG=0) %>%
  ev(amt=100, ii=24, addl=4) %>%
  idata_set(idata) %>%
  mrgsim(delta=0.1, end=408)

head(out)
plot(out, CP + RESPONSE~.,)

out_df = as.data.frame(out)
covariates_df = as.data.frame(idata)
write.csv(covariates_df, 'Random Data/PK_PD_100_24_5_100_WT.csv')
write.csv(out_df, 'Random Data/100_24_5_100_WT.csv')
