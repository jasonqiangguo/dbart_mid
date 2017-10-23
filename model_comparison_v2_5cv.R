data.path <- "/scratch/qg251/dbart_mid/Civil_War_PA"
script.path <- "/scratch/qg251/dbart_mid/muchlinski_rep"

options(java.parameters = "-Xmx300g")
data <- read.csv(file=paste0(data.path, "/SambnisImp.csv")) # data for prediction
data2 <- read.csv(file=paste0(data.path, "/Amelia.Imp3.csv")) # data for causal machanisms

# library(foreign)
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
library(MASS)
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

# registerDoMC(cores=7) # distributing workload over multiple cores for faster computaiton

set.seed(666) #the most metal seed for CV

# data.full <- sample_frac(data.full, size = 0.1, replace = FALSE)

#This method of data slicing - or CV - will be used for all logit models - uncorrected and corrected
tc<-trainControl(method="cv",
                 number=5,#creates CV folds - 5 for this data
                 summaryFunction=twoClassSummary, # provides ROC summary stats in call to model
                 classProb=T)

#Fearon and Laitin Model Specification###
model.fl.1<-train(as.factor(warstds)~warhist+ln_gdpen+lpopns+lmtnest+ncontig+oil+nwstate
             +inst3+pol4+ef+relfrac, #FL 2003 model spec
             metric="ROC", method="glm", family="binomial", #uncorrected logistic model
             trControl=tc, data=data.full)

summary(model.fl.1) #provides coefficients & traditional R model output
model.fl.1 # provides CV summary stats # keep in mind caret takes first class (here that's 0)
#as refernce class so sens and spec are backwards


###Now doing Collier and Hoeffler (2004) uncorrected logistic specification###
model.ch.1<-train(as.factor(warstds)~sxpnew+sxpsq+ln_gdpen+gdpgrowth+warhist+lmtnest+ef+popdense
                  +lpopns+coldwar+seceduc+ptime, #CH 2004 model spec
                  metric="ROC", method="glm", family="binomial",
                  trControl=tc, data=data.full)
model.ch.1


bartGrid <- expand.grid(num_trees = c(500, 1000), k = 2, alpha = 0.95, beta = 2, nu = 3)
model.bt <- train(as.factor(warstds)~., data=data.full, metric = "ROC", method = "bartMachine",
                  tuneGrid = bartGrid, trControl = tc,  num_burn_in = 2000, num_iterations_after_burn_in = 2000, serialize = T)


# model.bt <- bartMachine(X = data.full[,-1], y = factor(data.full$warstds), num_trees = 1000, num_burn_in = 2000, num_iterations_after_burn_in = 2000, alpha = 0.95, beta = 2, k = 2, q = 0.9, nu = 3, serialize = T)


#####################################################################
######## partial dependence plot conditioning on y = 1 ##############
#####################################################################

# x1 <- rpois(500, 30)
# x2 <- rpois(500, 20)
# Sigma <- matrix(c(0.04,0,0,0.04),2,2)
# 
# b <- c(0.01, 0.01)
# X <- cbind(x1, x2)
# 
# X %*% as.matrix(b)
# y <- rbinom(500, 1, prob = pnorm(X %*% as.matrix(b)))
# 
# dt <- as.data.frame(cbind(y, X))
# dt$y<-factor(dt$y, levels=c(0,1),
#              labels=c("peace", "war"))
# 
# bartGrid <- expand.grid(num_trees = c(500, 1000), k = 2, alpha = 0.95, beta = 2, nu = 3)
# model.bt <- train(as.factor(y)~ ., data=dt, metric = "ROC", method = "bartMachine",
#                   tuneGrid = bartGrid, trControl = tc,  num_burn_in = 100, num_iterations_after_burn_in = 100, serialize = T)
# 
# model.bt2 <- bartMachine(y=factor(dt[,1]), X=as.data.frame(dt$x), num_trees = 100, num_burn_in = 200, num_iterations_after_burn_in = 200, alpha = 0.95, beta = 2, k = 2, q = 0.9, nu = 3, serialize = T)

md <- model.bt$finalModel

save(md, file=paste0(data.path, "/final_model_5cv.RData"))

##partial dependence plot using covariates for those y = 1 
pd_plot_prob_case_select = function(bart_machine, j, levs = c(0.05, seq(from = 0.10, to = 0.90, by = 0.10), 0.95), lower_ci = 0.025, upper_ci = 0.975, prop_data = 1){
  # check_serialization(bart_machine) #ensure the Java object exists and fire an error if not
  if (class(j) == "integer"){
    j = as.numeric(j)
  }
  if (class(j) == "numeric" && (j < 1 || j > bart_machine$p)){
    stop(paste("You must set j to a number between 1 and p =", bart_machine$p))
  } else if (class(j) == "character" && !(j %in% bart_machine$training_data_features)){
    stop("j must be the name of one of the training features (see \"<bart_model>$training_data_features\")")
  } else if (!(class(j) == "numeric" || class(j) == "character")){
    stop("j must be a column number or column name")
  }
  
  x_j = bart_machine$model_matrix_training_data[, j]
  
  #fail with a warning if there's only one value (we don't want an error because it would fail on loops).
  if (length(unique(na.omit(x_j))) <= 1){
    warning("There must be more than one unique value in this training feature. PD plot not generated.")
    return()
  }
  x_j_quants = unique(quantile(x_j, levs, na.rm = TRUE))
  
  #fail with a warning if there's only one value (we don't want an error because it would fail on loops).
  if (length(unique(x_j_quants)) <= 1){
    warning("There must be more than one unique value among the quantiles selected. PD plot not generated.")
    return()
  }
  
  # n_pd_plot = round(bart_machine$n * prop_data)
  n_pd_plot = sum(bart_machine$y == "war")
  bart_predictions_by_quantile = array(NA, c(length(x_j_quants), n_pd_plot, bart_machine$num_iterations_after_burn_in))
  
  for (q in 1 : length(x_j_quants)){
    #pull out a certain proportion of the data randomly
    # indices = sample(1 : bart_machine$n, n_pd_plot)
    indices = which(bart_machine$y == "war")
    #now create test data matrix
    test_data = bart_machine$X[indices, ]
    test_data[, j] = rep(x_j_quants[q], n_pd_plot)
    
    bart_predictions_by_quantile[q, , ] = 1 - bart_machine_get_posterior(bart_machine, test_data)$y_hat_posterior_samples
    cat(".")
  }
  cat("\n")
  
  if (bart_machine$pred_type == "classification"){ ##convert to probits
    bart_predictions_by_quantile = bart_predictions_by_quantile
  }
  
  bart_avg_predictions_by_quantile_by_gibbs = array(NA, c(length(x_j_quants), bart_machine$num_iterations_after_burn_in))
  for (q in 1 : length(x_j_quants)){
    for (g in 1 : bart_machine$num_iterations_after_burn_in){
      bart_avg_predictions_by_quantile_by_gibbs[q, g] = mean(bart_predictions_by_quantile[q, , g])
    }		
  }
  
  bart_avg_predictions_by_quantile = apply(bart_avg_predictions_by_quantile_by_gibbs, 1, mean)
  bart_avg_predictions_lower = apply(bart_avg_predictions_by_quantile_by_gibbs, 1, quantile, probs = lower_ci)
  bart_avg_predictions_upper = apply(bart_avg_predictions_by_quantile_by_gibbs, 1, quantile, probs = upper_ci)
  
  var_name = ifelse(class(j) == "character", j, bart_machine$training_data_features[j])
  ylab_name = ifelse(bart_machine$pred_type == "classification", "Partial Effect (Probabilities)", "Partial Effect")
  plot(x_j_quants, bart_avg_predictions_by_quantile, 
       type = "o", 
       main = "Partial Dependence Plot",
       # ylim = c(min(bart_avg_predictions_lower, bart_avg_predictions_upper), max(bart_avg_predictions_lower, bart_avg_predictions_upper)),
       ylim = c(0, 0.5),
       ylab = ylab_name,
       xlab = paste(var_name, "plotted at specified quantiles"))
  polygon(c(x_j_quants, rev(x_j_quants)), c(bart_avg_predictions_upper, rev(bart_avg_predictions_lower)), col = "gray87", border = NA)
  lines(x_j_quants, bart_avg_predictions_lower, type = "o", col = "blue", lwd = 1)
  lines(x_j_quants, bart_avg_predictions_upper, type = "o", col = "blue", lwd = 1)
  lines(x_j_quants, bart_avg_predictions_by_quantile, type = "o", lwd = 2)
  
  invisible(list(x_j_quants = x_j_quants, bart_avg_predictions_by_quantile = bart_avg_predictions_by_quantile, prop_data = prop_data))
}


pdf(paste0(script.path, "/partial_growth_case_select_5cv.pdf"))
pd_plot_prob_case_select(md, "gdpgrowth")
dev.off()


##partial dependence plot using covariates for those y = 1 
pd_plot_prob = function(bart_machine, j, levs = c(0.05, seq(from = 0.10, to = 0.90, by = 0.10), 0.95), lower_ci = 0.025, upper_ci = 0.975, prop_data = 1){
  # check_serialization(bart_machine) #ensure the Java object exists and fire an error if not
  if (class(j) == "integer"){
    j = as.numeric(j)
  }
  if (class(j) == "numeric" && (j < 1 || j > bart_machine$p)){
    stop(paste("You must set j to a number between 1 and p =", bart_machine$p))
  } else if (class(j) == "character" && !(j %in% bart_machine$training_data_features)){
    stop("j must be the name of one of the training features (see \"<bart_model>$training_data_features\")")
  } else if (!(class(j) == "numeric" || class(j) == "character")){
    stop("j must be a column number or column name")
  }
  
  x_j = bart_machine$model_matrix_training_data[, j]
  
  #fail with a warning if there's only one value (we don't want an error because it would fail on loops).
  if (length(unique(na.omit(x_j))) <= 1){
    warning("There must be more than one unique value in this training feature. PD plot not generated.")
    return()
  }
  x_j_quants = unique(quantile(x_j, levs, na.rm = TRUE))
  
  #fail with a warning if there's only one value (we don't want an error because it would fail on loops).
  if (length(unique(x_j_quants)) <= 1){
    warning("There must be more than one unique value among the quantiles selected. PD plot not generated.")
    return()
  }
  
  n_pd_plot = round(bart_machine$n * prop_data)
  # n_pd_plot = sum(bart_machine$y == "war")
  bart_predictions_by_quantile = array(NA, c(length(x_j_quants), n_pd_plot, bart_machine$num_iterations_after_burn_in))
  
  for (q in 1 : length(x_j_quants)){
    #pull out a certain proportion of the data randomly
    indices = sample(1 : bart_machine$n, n_pd_plot)
    # indices = which(bart_machine$y == "war")
    #now create test data matrix
    test_data = bart_machine$X[indices, ]
    test_data[, j] = rep(x_j_quants[q], n_pd_plot)
    
    bart_predictions_by_quantile[q, , ] = 1 - bart_machine_get_posterior(bart_machine, test_data)$y_hat_posterior_samples
    cat(".")
  }
  cat("\n")
  
  if (bart_machine$pred_type == "classification"){ ##convert to probits
    bart_predictions_by_quantile = bart_predictions_by_quantile
  }
  
  bart_avg_predictions_by_quantile_by_gibbs = array(NA, c(length(x_j_quants), bart_machine$num_iterations_after_burn_in))
  for (q in 1 : length(x_j_quants)){
    for (g in 1 : bart_machine$num_iterations_after_burn_in){
      bart_avg_predictions_by_quantile_by_gibbs[q, g] = mean(bart_predictions_by_quantile[q, , g])
    }		
  }
  
  bart_avg_predictions_by_quantile = apply(bart_avg_predictions_by_quantile_by_gibbs, 1, mean)
  bart_avg_predictions_lower = apply(bart_avg_predictions_by_quantile_by_gibbs, 1, quantile, probs = lower_ci)
  bart_avg_predictions_upper = apply(bart_avg_predictions_by_quantile_by_gibbs, 1, quantile, probs = upper_ci)

  var_name = ifelse(class(j) == "character", j, bart_machine$training_data_features[j])
  ylab_name = ifelse(bart_machine$pred_type == "classification", "Partial Effect (Probabilities)", "Partial Effect")
  plot(x_j_quants, bart_avg_predictions_by_quantile, 
       type = "o", 
       main = "Partial Dependence Plot",
       # ylim = c(min(bart_avg_predictions_lower, bart_avg_predictions_upper), max(bart_avg_predictions_lower, bart_avg_predictions_upper)),
       ylim = c(0, 0.5),
       ylab = ylab_name,
       xlab = paste(var_name, "plotted at specified quantiles"))
  polygon(c(x_j_quants, rev(x_j_quants)), c(bart_avg_predictions_upper, rev(bart_avg_predictions_lower)), col = "gray87", border = NA)
  lines(x_j_quants, bart_avg_predictions_lower, type = "o", col = "blue", lwd = 1)
  lines(x_j_quants, bart_avg_predictions_upper, type = "o", col = "blue", lwd = 1)
  lines(x_j_quants, bart_avg_predictions_by_quantile, type = "o", lwd = 2)
  
  invisible(list(x_j_quants = x_j_quants, bart_avg_predictions_by_quantile = bart_avg_predictions_by_quantile, prop_data = prop_data))
}


pdf(paste0(script.path, "/partial_growth_5cv.pdf"))
pd_plot_prob(md, "gdpgrowth")
dev.off()

# ###Gathering info for ROC Plots ###
# FL.1.pred<-predict(model.fl.1, type="prob")
# RF.1.pred<-predict(model.rf, type="prob")
# BT.1.pred<-predict(model.bt, type="prob")
# # for bartMachine, the predicted probabilities for war and peace are reversed so need to make corrections for this situation
# names(BT.1.pred) <- names(BT.1.pred)[order(names(BT.1.pred), decreasing = T)]
# 
# produce_ROC_AUC <- function(pred){
#   pred <- prediction(pred$war, data.full$warstds)
#   perf <- performance(pred,"tpr","fpr")
#   AUC <- round(performance(pred,"auc")@y.values[[1]], digits = 3)
#   return(list(perf, AUC))
# }
# 
# FL <- produce_ROC_AUC(FL.1.pred)
# RF <- produce_ROC_AUC(RF.1.pred)
# BT <- produce_ROC_AUC(BT.1.pred)
# 
# pdf(paste0(script.path, "/roc_compare_5_fold.pdf"), height = 8, width = 8)
# plot(FL[[1]], main="Logits, Random Forests and BART")
# plot(RF[[1]], add=T, lty=2)
# plot(BT[[1]], add=T, lty=3)
# legend(0.32, 0.25, c(paste0("Fearon and Laitin (2003) Logit ", FL[[2]]), paste0("Random Forest ", RF[[2]]),  paste0("BART ", BT[[2]])), lty=c(1,2,3), bty="n",
#        cex = .75)
# dev.off()
# 
# # save predicted probabilities from 3 models into a dataset together with actual war onset
# data.full$warstds <- ifelse(data.full$warstds == "peace", 0, 1)
# out.pred <- as.data.frame(cbind(data.full$warstds, FL.1.pred$war,  RF.1.pred$war,  BT.1.pred$war))
# names(out.pred) <- c("onset", "logit", "RandomF", "BART")
# 
# #######################################
# # produce marginal effects for growth #
# #######################################
# 
# quant_prob <- c(.01, seq(0.05, 0.95, 0.05), .99)
# 
# # marginal effects plot of gdpgrowth
# create_test_data <- function(choice){
#   if (choice == "high"){
#     covariates <- data.full[,-1]
#     x <- covariates[which(data.full$warstds == 1),]
#   }
#   if (choice == "low"){
#     covariates <- data.full[,-1]
#     x <- covariates[which(data.full$warstds == 0),]
#   }
# 
#   q <- apply(x[,1:length(covariates)], 2, median)
#   test <- apply(matrix(q, nrow = 1), 2, function(x) rep(x, length(quant_prob)))
#   test[,which(names(covariates) == "gdpgrowth")] <- quantile(covariates[,which(names(covariates) == "gdpgrowth")], probs = quant_prob)
#   test <- as.data.frame(test)
#   names(test) <- names(covariates)
#   return(test)
# }
# 
# test <- lapply(c("high", "low"), create_test_data)
# 
# btmchine_phat <- predict(model.bt, test[[1]], type = "prob")$peace
# btmchine.hpc <- 1 - calc_credible_intervals(model.bt$finalModel, test[[1]])
# btmchine.plot.data <- as.data.frame(cbind(btmchine_phat, btmchine.hpc[,2], btmchine.hpc[,1], test[[1]]$gdpgrowth))
# names(btmchine.plot.data) <- c("p_hat", "ci_lower_bd", "ci_upper_bd", "gdpgrowth")
# 
# 
# 
# create_marginal_plot_gdpgrowth <- function(df){
#   g <- ggplot(df, aes(x = gdpgrowth, y = p_hat))
#   g <- g + geom_line() +  ylim(0, 1)
#   g <- g + geom_errorbar(aes(ymax = ci_upper_bd, ymin = ci_lower_bd), width = 0.02)
#   g <- g + theme_bw() + theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5))
#   g <- g + xlab("GDPgrowth") + ylab("Probability")
#   g
# }
# 
# btmchine.plot <- create_marginal_plot_gdpgrowth(btmchine.plot.data)
# 
# 
# data.full$warstds<-factor(
#   data.full$warstds,
#   levels=c(0,1),
#   labels=c("peace", "war"))
# 
# model.logit<- train(as.factor(warstds)~sxpnew+sxpsq+ln_gdpen+gdpgrowth+warhist+lmtnest+ef+popdense
#                     +lpopns+coldwar+seceduc+ptime+trade,
#                     metric="ROC", method="glm", family="binomial", #uncorrected logistic model
#                     trControl=tc, data=data.full)
# 
# model.logit
# 
# 
# data.full$warstds <- ifelse(data.full$warstds == "peace", 0, 1)
# 
# create_test_data_logit <- function(choice){
#   if (choice == "high"){
#     covariates <- data.full[,c("sxpnew", "sxpsq", "ln_gdpen", "gdpgrowth", "warhist", "lmtnest", "ef", "popdense",
#                                "lpopns", "coldwar", "seceduc", "ptime", "trade")]
#     x <- covariates[which(data.full$warstds == 1),]
#   }
#   if (choice == "low"){
#     covariates <- data.full[,c("sxpnew", "sxpsq", "ln_gdpen", "gdpgrowth", "warhist", "lmtnest", "ef", "popdense",
#                                "lpopns", "coldwar", "seceduc", "ptime", "trade")]
#     x <- covariates[which(data.full$warstds == 0),]
#   }
# 
#   q <- apply(x[,1:length(covariates)], 2, median)
#   test <- apply(matrix(q, nrow = 1), 2, function(x) rep(x, length(quant_prob)))
#   test[,which(names(covariates) == "gdpgrowth")] <- quantile(covariates[,which(names(covariates) == "gdpgrowth")], probs = quant_prob)
#   test <- as.data.frame(test)
#   names(test) <- names(covariates)
#   return(test)
# }
# 
# test <- lapply(c("high", "low"), create_test_data_logit)
# 
# 
# summary(model.logit$finalModel)
# 
# logit.pred <- predict(model.logit$finalModel, newdata = test[[1]], type = "response", se.fit = T)
# ci_logit_upper <- logit.pred$fit + 1.96*logit.pred$se.fit
# ci_logit_lower <- logit.pred$fit - 1.96*logit.pred$se.fit
# 
# 
# logit.plot.data <- as.data.frame(cbind(ci_logit_upper, ci_logit_lower, logit.pred$fit, test[[1]]$gdpgrowth))
# names(logit.plot.data) <- c("ci_upper_bd", "ci_lower_bd", "p_hat", "gdpgrowth")
# logit.plot <- create_marginal_plot_gdpgrowth(logit.plot.data)
# 
# pp <- grid.arrange(btmchine.plot, logit.plot, ncol = 2)
# 
# ggsave(paste0(script.path, "/logit_vs_bart_marginal_gdpgrowth_5_fold.pdf"), pp, height = 8, width = 16)
# 
# # save all the data files in the "dbart_mid" dataset
# setwd(data.path)
# save(model.bt, file = "model.bt.5.fold.RData")
# write.csv(out.pred, file = "onset.prediction.5.fold.CSV")


# pdf(paste0(script.path, "/partial_growth.pdf"))
# pd_plot(model.bt, "gdpgrowth")
# dev.off()

# partial dependence plot conditioning on y = 1
# can variance be caused by low density of covariate values in the support
# generate predicted values for y, take top 25% and use their covariates
# which bill becomes law
# regime change 
# http://www.cs.nyu.edu/~mohri/pub/

