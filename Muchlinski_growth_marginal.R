# setwd("~/Dropbox/dbart_mid/Civil_War_PA")
# setwd("/scratch/qg251/dbart_mid/Civial_War_PA")

data.path <- "/scratch/qg251/dbart_mid/Civil_War_PA"
script.path <- "/scratch/qg251/dbart_mid/muchlinski_rep"

options(java.parameters = "-Xmx300g")

data=read.csv(file=paste0(data.path, "/SambnisImp.csv")) # data for prediction
data2<-read.csv(file=paste0(data.path, "/Amelia.Imp3.csv")) # data for causal machanisms

# data=read.csv(file="SambnisImp.csv") # data for prediction
# data2<-read.csv(file="Amelia.Imp3.csv") # data for causal machanisms



library(foreign)
library(dbarts)
library(ggplot2)
library(gridExtra)
library(dplyr)
library(randomForest) #for random forests
library(caret) # for CV folds and data splitting
library(ROCR) # for diagnostics and ROC plots/stats
library(pROC) # same as ROCR
#library(stepPlr) # Firth;s logit implemented thru caret library
# library(doMC) # for using multipe processor cores 
library(bartMachine)
###Using only the 88 variables specified in Sambanis (2006) Appendix###
data.full<-data[,c("warstds", "ager", "agexp", "anoc", "army85", "autch98", "auto4",
        "autonomy", "avgnabo", "centpol3", "coldwar", "decade1", "decade2",
        "decade3", "decade4", "dem", "dem4", "demch98", "dlang", "drel",
        "durable", "ef", "ef2", "ehet", "elfo", "elfo2", "etdo4590",
        "expgdp", "exrec", "fedpol3", "fuelexp", "gdpgrowth", "geo1", "geo2",
        "geo34", "geo57", "geo69", "geo8", "illiteracy", "incumb", "infant",
        "inst", "inst3", "life", "lmtnest", "ln_gdpen", "lpopns", "major", "manuexp", "milper",
        "mirps0", "mirps1", "mirps2", "mirps3", "nat_war", "ncontig",
        "nmgdp", "nmdp4_alt", "numlang", "nwstate", "oil", "p4mchg",
        "parcomp", "parreg", "part", "partfree", "plural", "plurrel",
        "pol4", "pol4m", "pol4sq", "polch98", "polcomp", "popdense",
        "presi", "pri", "proxregc", "ptime", "reg", "regd4_alt", "relfrac", "seceduc",
        "second", "semipol3", "sip2", "sxpnew", "sxpsq", "tnatwar", "trade",
        "warhist", "xconst")]

###Converting DV into Factor with names for Caret Library###
data.full$warstds<-factor(
  data.full$warstds,
  levels=c(0,1),
  labels=c("peace", "war"))

# registerDoMC(cores=3) # distributing workload over multiple cores for faster computaiton

# set.seed(666) #the most metal seed for CV
# 
# model.btmchine <- bartMachine(X = data.full[,-1], y = factor(data.full$warstds), num_trees = 1000, num_burn_in = 2000, num_iterations_after_burn_in = 10000, serialize = T)
# save(model.btmchine, file = paste0(data.path, "/model.btmchine.RData"))

load(paste0(data.path, "/model.btmchine.RData"))


##############################################################
###### run dbarts cross validation to get the best model #####
##############################################################
data.full$warstds <- ifelse(data.full$warstds == "peace", 0, 1)

quant_prob <- c(.01, seq(0.05, 0.95, 0.05), .99)

# marginal effects plot trade
create_test_data <- function(choice){
  if (choice == "high"){
    covariates <- data.full[,-1] 
    x <- covariates[which(data.full$warstds == 1),] 
  }
  if (choice == "low"){
    covariates <- data.full[,-1] 
    x <- covariates[which(data.full$warstds == 0),] 
  }
  
  q <- apply(x[,1:length(covariates)], 2, median)
  test <- apply(matrix(q, nrow = 1), 2, function(x) rep(x, length(quant_prob)))
  test[,which(names(covariates) == "gdpgrowth")] <- quantile(covariates[,which(names(covariates) == "gdpgrowth")], probs = quant_prob)
  test <- as.data.frame(test)
  names(test) <- names(covariates)
  return(test)
}

test <- lapply(c("high", "low"), create_test_data)


# dbart_formula <- as.formula(paste0("data.full$warstds ~ ", paste(names(data.full)[-1], collapse = "+")))
# model.dbart <- dbarts(dbartsData(dbart_formula, data.full, test = test[[1]]), control = dbartsControl(n.burn = 2000L, n.tree = 1000L, n.samples = 1000L, n.thin = 10))
# dbart_result <- model.dbart$run()
# save(dbart_result, file = paste0(data.path, "/model.dbart.growth.RData"))
 

load(paste0(data.path, "/model.dbart.growth.RData"))


dbart_phat <- apply(dbart_result$test, 1, function(x) mean(pnorm(x)))
dbart.hpc <- apply(dbart_result$test, 1, function(x) quantile(pnorm(x), probs = c(0.05, 0.95)))
dbart.plot.data <- as.data.frame(cbind(dbart_phat, dbart.hpc[1,], dbart.hpc[2,], test[[1]]$gdpgrowth))
names(dbart.plot.data) <- c("p_hat", "ci_lower_bd", "ci_upper_bd", "gdpgrowth")

dbart_posterior_draws <- as.data.frame(t(pnorm(dbart_result$test)))
print(dim(dbart_posterior_draws))
names(dbart_posterior_draws) <- unlist(lapply(seq(5, 95, by = 5), function(x) paste0("pct_", x)))
print(names(dbart_posterior_draws))
write.dta(dbart_posterior_draws, "dbart_posterior_draws.dta")

# in bartMachine result the probability is reversed for 0 and 1, so when calculating the y_hat and confidence intervals we have to correct the problem using the yhat to subtract from 1
btmachine.matrix.full <- 1 - bart_machine_get_posterior(model.btmchine, new_data = test[[1]])$y_hat_posterior_samples
btmachine.matrix <- btmachine.matrix.full[, 1:1000*10]
btmchine_phat <- apply(btmachine.matrix, 1, function(x) mean(x))
btmchine.hpc <- apply(btmachine.matrix, 1, function(x) quantile(x, probs = c(0.05, 0.95)))
btmchine.plot.data <- as.data.frame(cbind(btmchine_phat, btmchine.hpc[1,], btmchine.hpc[2,], test[[1]]$gdpgrowth))
names(btmchine.plot.data) <- c("p_hat", "ci_lower_bd", "ci_upper_bd", "gdpgrowth")

btmachine_posterior_draws <- as.data.frame(t(btmachine.matrix.full))
names(btmachine_posterior_draws) <- unlist(lapply(seq(5, 95, by = 5), function(x) paste0("pct_", x)))
write.dta(btmachine_posterior_draws, "btmachine_posterior_draws.dta")


create_marginal_plot_gdpgrowth <- function(df){
  g <- ggplot(df, aes(x = gdpgrowth, y = p_hat))
  g <- g + geom_line() +  ylim(0, 1)
  g <- g + geom_errorbar(aes(ymax = ci_upper_bd, ymin = ci_lower_bd), width = 0.02)
  g <- g + theme_bw() + theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5))
  g <- g + xlab("GDPgrowth") + ylab("Probability")
  g
}

btmchine.plot <- create_marginal_plot_gdpgrowth(btmchine.plot.data)
dbart.plot <- create_marginal_plot_gdpgrowth(dbart.plot.data)



# probit model
model.probit<-glm(as.factor(warstds)~sxpnew+sxpsq+ln_gdpen+gdpgrowth+warhist+lmtnest+ef+popdense
+lpopns+coldwar+seceduc+ptime+trade,
data=data.full, family=binomial(link = "probit"))

model.probit



create_test_data_probit <- function(choice){
  if (choice == "high"){
    covariates <- data.full[,c("sxpnew", "sxpsq", "ln_gdpen", "gdpgrowth", "warhist", "lmtnest", "ef", "popdense",
                               "lpopns", "coldwar", "seceduc", "ptime", "trade")]
    x <- covariates[which(data.full$warstds == 1),]
  }
  if (choice == "low"){
    covariates <- data.full[,c("sxpnew", "sxpsq", "ln_gdpen", "gdpgrowth", "warhist", "lmtnest", "ef", "popdense",
                               "lpopns", "coldwar", "seceduc", "ptime", "trade")]
    x <- covariates[which(data.full$warstds == 0),]
  }

  q <- apply(x[,1:length(covariates)], 2, median)
  test <- apply(matrix(q, nrow = 1), 2, function(x) rep(x, length(quant_prob)))
  test[,which(names(covariates) == "gdpgrowth")] <- quantile(covariates[,which(names(covariates) == "gdpgrowth")], probs = quant_prob)
  test <- as.data.frame(test)
  names(test) <- names(covariates)
  return(test)
}

test <- lapply(c("high", "low"), create_test_data_probit)

probit.pred <- predict(model.probit, newdata = test[[1]], type = "response", se.fit = TRUE)
ci_probit_upper <- probit.pred$fit + 1.96*probit.pred$se.fit
ci_probit_lower <- probit.pred$fit - 1.96*probit.pred$se.fit


probit.plot.data <- as.data.frame(cbind(ci_probit_upper, ci_probit_lower, probit.pred$fit, test[[1]]$gdpgrowth))
names(probit.plot.data) <- c("ci_upper_bd", "ci_lower_bd", "p_hat", "gdpgrowth")
probit.plot <- create_marginal_plot_gdpgrowth(probit.plot.data)


pp <- grid.arrange(btmchine.plot, dbart.plot, probit.plot, ncol = 2)
ggsave(paste0(script.path, "/btmachine_vs_dbart_vs_probit_marginal_gdpgrowth.pdf"), pp, height = 10, width = 10)

