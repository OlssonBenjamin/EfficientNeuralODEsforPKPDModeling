# Load required libraries
library(mrgsolve)
library(ggplot2)
library(dplyr)
library(dmutate)

mod <- mread('pkpd_covariates')
set.seed(100)
#mod <- mread("pkpd_covariates") %>%
#  omat(list(cmat(0.1, 0.8, 0.3),dmat(0.5)))
#mod

idata <- tibble(ID=1:100) %>%
mutate(WT = 70)%>%
mutate(AGE = 40) %>%
mutate(SEX = 1.5)


out <- mod %>% param(TLAG=0) %>%
  ev(amt=150, ii=48, addl=5) %>%
  idata_set(idata) %>%
  mrgsim(delta=0.1, end=408)

head(out)
plot(out, CP + RESPONSE~.,)

out_df = as.data.frame(out)
covariates_df = as.data.frame(idata)
write.csv(covariates_df, 'Mean_150_48_6_100.csv')
write.csv(out_df, 'PK_PD_Mean_150_48_6_100.csv')

