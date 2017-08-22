# setwd("~/Dropbox/dbart_mid/Civil_War_PA")
# setwd("/scratch/qg251/dbart_mid/Civial_War_PA")
options(java.parameters = "-Xmx300g")
getwd()
data=read.csv(file="SambnisImp.csv") # data for prediction
data2<-read.csv(file="Amelia.Imp3.csv") # data for causal machanisms

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

set.seed(666) #the most metal seed for CV

# data.full <- sample_frac(data.full, size = 0.05, replace = FALSE)

#This method of data slicing - or CV - will be used for all logit models - uncorrected and corrected
# tc<-trainControl(method="cv",
#                  number=10,#creates CV folds - 10 for this data
#                  summaryFunction=twoClassSummary, # provides ROC summary stats in call to model
#                  classProb=T)
#                  
# 
# ###Implementing RF (with CV) on entirety of data###
# model.rf<-train(as.factor(warstds)~., metric = "ROC",
#                   method="rf",
#                   sampsize=c(30,90), #Downsampling the class-imbalanced DV
#                     importance=T, # Variable importance measures retained
#                    proximity=F, ntree=1000, # number of trees grown
#                 trControl = tc, data=data.full)



# model.rf.pred <- predict(model.rf$finalModel, type = "prob")
# rf.p_hat <- model.rf.pred[,2]
# plot(performance(prediction(rf.p_hat, data$warstds), "tpr", "fpr"))
# performance(prediction(rf.p_hat, data$warstds), "auc")
# 
# 
# # run bartmachine cross-validation to get the best model
# # bartGrid <- expand.grid(num_trees = c(500, 1000), k = 2, alpha = 0.95, beta = 2, nu = 3)
# # model.bt <- train(as.factor(warstds)~., data=data.full, metric = "ROC", method = "bartMachine", tuneGrid = bartGrid, trControl = tc,  num_burn_in = 2000, num_iterations_after_burn_in = 1000)
# model.btmchine <- bartMachine(X = data.full[,-1], y = factor(data.full$warstds), num_trees = 1000, num_burn_in = 2000, num_iterations_after_burn_in = 10000, serialize = T)
# save(model.btmchine, file = "model.btmchine.RData")
# 
# load("model.btmchine.RData")

# pdf("convergence_diagnostic.pdf", height = 6, width = 10)
# plot_convergence_diagnostics(model.btmchine)
# dev.off()

# BT.1.pred <- 1 - model.btmchine$p_hat_train
# pred.BT.1<-prediction(BT.1.pred, data.full$warstds)
# perf.BT.1 <- performance(pred.BT.1, "tpr", "fpr")
# BT.1.AUC <- round(performance(pred.BT.1, "auc")@y.values[[1]], digits = 3)
# bt_posterior_draws <- bart_machine_get_posterior(model.btmchine, data.full[,-1])


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
  test[,which(names(covariates) == "trade")] <- quantile(covariates[,which(names(covariates) == "trade")], probs = quant_prob)
  test <- as.data.frame(test)
  names(test) <- names(covariates)
  return(test)
}

test <- lapply(c("high", "low"), create_test_data)


# dbart_formula <- as.formula(paste0("data.full$warstds ~ ", paste(names(data.full)[-1], collapse = "+")))
# model.dbart <- dbarts(dbartsData(dbart_formula, data.full, test = test[[1]]), control = dbartsControl(n.burn = 2000L, n.tree = 1000L, n.samples = 1000L, n.thin = 10))
# dbart_result <- model.dbart$run()
# save(dbart_result, file = "model.dbart.RData")
# 

load("model.dbart.RData")

dbart_phat <- apply(dbart_result$test, 1, function(x) mean(pnorm(x)))
dbart.hpc <- apply(dbart_result$test, 1, function(x) quantile(pnorm(x), probs = c(0.05, 0.95)))
dbart.plot.data <- as.data.frame(cbind(dbart_phat, dbart.hpc[1,], dbart.hpc[2,], test[[1]]$trade))
names(dbart.plot.data) <- c("p_hat", "ci_lower_bd", "ci_upper_bd", "trade")

# dbart.1.pred <- apply(dbart_result$train, 1, function(x) mean(pnorm(x)))
# pred.dbart.1 <- prediction(dbart.1.pred, data.full$warstds)
# perf.dbart.1 <- performance(pred.dbart.1, "tpr", "fpr")
# dbart.1.AUC <- round(performance(pred.dbart.1, "auc")@y.values[[1]], digits = 3)

# pdf("dbart_v_btmchine.pdf", width = 8, height = 8)
# plot(perf.BT.1, main="dbart vs. bartMachine")
# plot(perf.dbart.1, add=T, lty=2)
# # plot(perf.HS, add=T, lty=3)
# # plot(perf.RF.1, add=T, lty=4)
# # plot(pref.BT.1, add=T, lty=5)
# legend(0.32, 0.25, c(paste0("bartMachine ", BT.1.AUC), paste0("dbart ", dbart.1.AUC)), lty=c(1,2), bty="n", 
#        cex = .75)
# dev.off()


marginal.mean.pred.high <- 1 - bart_machine_get_posterior(model.btmchine, new_data = test[[1]])$y_hat

# marginal.mean.pred.low <- 1 - bart_machine_get_posterior(model.btmchine, new_data = test[[2]])$y_hat

# in bartMachine result the probability is reversed for 0 and 1, so when calculating the y_hat and confidence intervals we have to correct the problem using the yhat to subtract from 1
# calculate credible interval for each observation
hpc_95_high <- 1 - calc_credible_intervals(model.btmchine, new_data = test[[1]])

btmchine.plot.data <- as.data.frame(cbind(hpc_95_high[,1], hpc_95_high[,2], marginal.mean.pred.high, test[[1]]$trade))
names(btmchine.plot.data) <- c("ci_upper_bd", "ci_lower_bd", "p_hat", "trade")

qqplot(btmchine.plot.data$p_hat, dbart.plot.data$p_hat, xlim = c(0,1), ylim = c(0, 1))
abline(a = 0, b = 1, col = "red")


create_marginal_plot_trade <- function(df){
  g <- ggplot(df, aes(x = trade, y = p_hat))
  g <- g + geom_line() +  ylim(0, 1)
  g <- g + geom_errorbar(aes(ymax = ci_upper_bd, ymin = ci_lower_bd), width = 0.02)
  g <- g + theme_bw() + theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5))
  g <- g + xlab("Trade") + ylab("Probability")
  g
}

btmchine.plot <- create_marginal_plot_trade(btmchine.plot.data)
dbart.plot <- create_marginal_plot_trade(dbart.plot.data)


pp <- grid.arrange(btmchine.plot, dbart.plot, ncol = 2)
ggsave("btmachine_vs_dbart_marginal.pdf", pp, height = 8, width = 12)
#confusionMatrix(model.rf, norm="average")


###Running Random Forest without CV to get OOB Error Rate - results not shown in paper###
# RF.out<-randomForest(as.factor(warstds)~., sampsize=c(30, 90),
#        importance=T, proximity=F, ntree=1000, confusion=T, err.rate=T, data=data.full)
# print(RF.out)


###Random Forests on Amelia Imputed Data for Variable Importance Plot###
###Data Imputed only for Theoretically Important Variables### Done to analyze causal mechanisms

# myvars <- names(data2) %in% c("X", "country", "year", "atwards") 
# newdata <- data2[!myvars]
# 
# RF.out.am<-randomForest(as.factor(warstds)~.,sampsize=c(30, 90),
#           importance=T, proximity=F, ntree=1000, confusion=T, err.rate=T, data=newdata)
# 
# varImpPlot(RF.out.am, sort=T, type=2, main="Variables Contributing Most to Predictive Accuracy of
#  Random Forests",n.var=20)
# 
# 
# 
# ###Creating dotplot for Variable Importance Plot for RF###
# importance(RF.out.am)

# x<-c(2.3246380, 2.3055470, 1.8544127, 1.7447636, 1.6848230, 1.6094923,
#      1.5564487, 1.4832437, 1.4100489, 1.3116247, 1.2875924, 1.1799487,
#      1.1034743, 1.0983414, 1.0689367, 1.0663479, 1.0123892, 0.9961138, 0.9961138, 0.9922545)
# g<-c("GDP Growth", "GDP per Capita", 
#      "Life Expectancy", "Western Europe and US Dummy", "Infant Mortality",
#      "Trade as Percent of GDP", "Mountainous Terrain", "Illiteracy Rate",
#      "Population (logged)", "Linguistic Hetrogeneity", "Anocracy", "Median Regional Polity Score",
#      "Primary Commodity Exports (Squared)", "Democracy", "Military Power", "Population Density",
#      "Political Instability", "Ethnic Fractionalization", "Secondary Education",
#      "Primary Commodity Exports")
# dotchart(rev(x), rev(g), cex=0.75, xlab="Mean Decrease in Gini Score (OOB Estimates)", 
#          main="Variable Importance for Random Forests", 
#          xlim=c(1, 2.5))
# 
# ###ROC Plots for Different Models###




# attach(data.full) #have to attach the data to get probs for some reason
# detach(data.full)
###Gathering info for ROC Plots: Uncorrected Logists###

# FL.1.pred<-predict(model.fl.1, type="prob")
# CH.1.pred<-predict(model.ch.1, type="prob")
# HS.1.pred<-predict(model.hs.1, type="prob")
# RF.1.pred<-predict(model.rf, type="prob")
# BT.1.pred<-1 - model.bt$p_hat_train
# 
# pred.FL <- prediction(FL.1.pred$war, data$warstds)
# perf.FL <- performance(pred.FL,"tpr","fpr")
# pred.CH <- prediction(CH.1.pred$war, data$warstds)
# perf.CH <- performance(pred.CH,"tpr","fpr")
# pred.HS<-prediction(HS.1.pred$war, data$warstds)
# perf.HS<-performance(pred.HS, "tpr", "fpr")
# pred.RF.1<-prediction(RF.1.pred$war, data$warstds)
# perf.RF.1<-performance(pred.RF.1, "tpr", "fpr")
# pred.BT.1<-prediction(BT.1.pred, data$warstds)
# perf.BT.1 <- performance(pred.BT.1, "tpr", "fpr")
# 
# BT.1.AUC <- performance(pred.BT.1, "auc")@y.values[[1]]
# 
# pdf("ml_comparison.pdf", width = 8, height = 8)
# plot(perf.FL, main="Uncorrected Logits and Random Forests")
# plot(perf.CH, add=T, lty=2)
# plot(perf.HS, add=T, lty=3)
# plot(perf.RF.1, add=T, lty=4)
# plot(pref.BT.1, add=T, lty=5)
# legend(0.32, 0.25, c("Fearon and Laitin (2003) 0.77", "Collier and Hoeffler (2004) 0.82", 
#                      "Hegre and Sambanis (2006) 0.80", "Random Forest 0.91", paste0("Bart ", BT.1.AUC)), lty=c(1,2,3,4,5), bty="n", 
#        cex = .75)
# dev.off()
# ###Now ROC Plots for Penalized Logits and RF###
# FL.2.pred<-predict(model.fl.2, data.full$warstds, type="prob")
# CH.2.pred<-predict(model.ch.2, data.full$warstds, type="prob")
# HS.2.pred<-predict(model.hs.2, data.full$warstds, type="prob")
# 
# 
# pred.FL.2 <- prediction(FL.2.pred$war, data$warstds)
# perf.FL.2 <- performance(pred.FL.2,"tpr","fpr")
# pred.CH.2 <- prediction(CH.2.pred$war, data$warstds)
# perf.CH.2 <- performance(pred.CH.2,"tpr","fpr")
# pred.HS.2<-prediction(HS.2.pred$war, data$warstds)
# perf.HS.2<-performance(pred.HS.2, "tpr", "fpr")
# 
# plot(perf.FL.2, main="Penalized Logits and Random Forests")
# plot(perf.CH.2, add=T, lty=2)
# plot(perf.HS.2, add=T, lty=3)
# plot(perf.RF.1, add=T, lty=4)
# legend(0.32, 0.25, c("Fearon and Laitin (2003) 0.77", "Collier and Hoeffler (2004) 0.77", 
#                      "Hegre and Sambanis (2006) 0.80", "Random Forest 0.91" ), lty=c(1,2,3,4), bty="n", 
#        cex = .75)
# 
# ###Separation Plots###
# library(separationplot)
# 
# ##Have to transform DV back to 0,1 values for sep plots
# data.full$warstds<-factor(
#   data.full$warstds,
#   levels=c("peace","war"),
#   labels=c(0, 1))
# 
# Warstds<-as.vector(data.full$warstds) #transforming actual obs into vector
# 
# 
# 
# separationplot(FL.1.pred$war, Warstds, type = "line", line = T, lwd2=1,
#   show.expected=T, heading="Fearon and Laitin (2003)", height=1.5, col0="white", col1="black")
# separationplot(CH.1.pred$war, Warstds, type = "line", line = T, lwd2=1,show.expected=T,
#          heading="Collier and Hoeffler (2004)", height=2.5, col0="white", col1="black")
# separationplot(HS.1.pred$war, Warstds, type = "line", line = T, lwd2=1, show.expected=T,
#          heading="Hegre and Sambanis (2006)", height=2.5, col0="white", col1="black")
# separationplot(RF.1.pred$war, Warstds, type = "line", line = T, lwd2=1, show.expected=T,
#         heading="Random Forests", height=2.5, col0="white", col1="black")
# 
# 
# ###Partial Dependence Plots to look at "causal processes"###
# 
# par(mfrow=c(3,3))
# partialPlot(RF.out.am, data2, gdpgrowth, which.class="1", xlab="GDP Growth Rate", main="",
#             ylab=expression(paste(Delta, "Fraction of Votes p(Y=1)")))
# partialPlot(RF.out.am, data2, ln_gdpen, ylim=c(-0.15, 0.15), which.class="1", xlab="GDP per Capita (log)", 
#             main="", ylab=expression(paste(Delta, "Fraction of Votes p(Y=1)")))
# partialPlot(RF.out.am, data2, life, ylim=c(-0.15, 0.15), which.class="1", xlab="Life Expectancy", 
#             main="", ylab=expression(paste(Delta, "Fraction of Votes p(Y=1)")))
# partialPlot(RF.out.am, data2, infant, ylim=c(-0.15, 0.15), which.class="1", xlab="Infant Mortality Rate",
#             main="", ylab=expression(paste(Delta, "Fraction of Votes p(Y=1)")))
# partialPlot(RF.out.am, data2, lmtnest, ylim=c(-0.15, 0.15), which.class="1", xlab="Mountainous Terrain (log)"
#             , main="",  ylab=expression(paste(Delta, "Fraction of Votes p(Y=1)")))
# partialPlot(RF.out.am, data2, pol4sq, ylim=c(-0.15, 0.15), which.class="1", xlab="Polity IV Sq", 
#             main="", ylab=expression(paste(Delta, "Fraction of Votes p(Y=1)")))
# partialPlot(RF.out.am, data2, lpopns, ylim=c(-0.15, 0.15), which.class="1", xlab="Population", main="",
#             ylab=expression(paste(Delta, "Fraction of Votes p(Y=1)")))
# partialPlot(RF.out.am, data2, trade, ylim=c(-0.15, 0.15), which.class="1", xlab="Trade", main="",
#             xlim=c(0,200), ylab=expression(paste(Delta, "Fraction of Votes p(Y=1)")))
# partialPlot(RF.out.am, data2, geo1, ylim=c(-0.15, 0.15), which.class="1", xlab="W. Europe and U.S.", 
#             main="",ylab=expression(paste(Delta, "Fraction of Votes p(Y=1)")))
# 
# 
####Analysis for Out of Sample Africa Data###
# data3<-read.csv(file="AfricaImp.csv") # Reading in the Africa Data from 2001-2014
# 
# model.fl.africa<-glm(as.factor(warstds)~warhist+ln_gdpen+lpopns+lmtnest+ncontig+oil+nwstate
#                   +inst3+pol4+ef+relfrac, family="binomial", data=data.full)
# model.ch.africa<-glm(as.factor(warstds)~sxpnew+sxpsq+ln_gdpen+gdpgrowth+warhist+lmtnest+ef+popdense
#                   +lpopns+coldwar+seceduc+ptime, family="binomial", data=data.full)
# model.hs.africa<-glm(warstds~lpopns+ln_gdpen+inst3+parreg+geo34+proxregc+gdpgrowth+anoc+
#               partfree+nat_war+lmtnest+decade1+pol4sq+nwstate+regd4_alt+etdo4590+milper+
#                geo1+tnatwar+presi,, family="binomial", data=data.full)
# 
# 
# 
# yhat.rf<-predict(RF.out, type="prob") #taken from RF on whole data
# 
# ###We used original CW data for training data here for all models/algorithms###
# Yhat.rf<-as.data.frame(yhat.rf[,2])
# yhat.fl.africa<-predict(model.fl.africa, type="response")
# Yhat.fl.africa<-as.data.frame(yhat.fl.africa)
# yhat.ch.africa<-predict(model.ch.africa, type="response")
# Yhat.ch.africa<-as.data.frame(yhat.ch.africa)
# yhat.hs.africa<-predict(model.hs.africa, type="response")
# Yhat.hs.africa<-as.data.frame(yhat.hs.africa)
# 
# ###Selecting random samples to make pred and actual lengths equal###
# set.seed(100)
# predictors.rf<-Yhat.rf[sample(nrow(Yhat.rf), 737),]
# predictors.fl<-Yhat.fl.africa[sample(nrow(Yhat.fl.africa), 737),]
# predictors.ch<-Yhat.ch.africa[sample(nrow(Yhat.ch.africa), 737),]
# predictors.hs<-Yhat.hs.africa[sample(nrow(Yhat.hs.africa), 737),]
# 
# 
# install.packages("SDMTools")
# library(SDMTools)
# 
# 
# confusion.matrix(data3$warstds, predictors.rf, threshold=.5)
# confusion.matrix(data3$warstds, predictors.fl, threshold=.5)
# confusion.matrix(data3$warstds, predictors.ch, threshold=.5)
# confusion.matrix(data3$warstds, predictors.hs, threshold=.5)
# 
# ###ROC and AUC scores for out of sample data###
# library(ROCR)
# pred.fl.africa <- prediction(predictors.fl, data3$warstds)
# perf.fl.africa<- performance(pred.fl.africa,"tpr","fpr")
# pred.ch.africa<-prediction(predictors.ch, data3$warstds)
# perf.ch.africa<-performance(pred.ch.africa, "tpr", "fpr")
# pred.hs.africa<-prediction(predictors.hs, data3$warstds)
# perf.hs.africa<-performance(pred.hs.africa, "tpr", "fpr")
# pred.rf.africa<-prediction(predictors.rf, data3$warstds)
# perf.rf.africa<-performance(pred.rf.africa, "tpr", "fpr")
# auc.fl.africa<-performance(pred.fl.africa, "auc")
# auc.fl.africa
# auc.ch.africa<-performance(pred.ch.africa, "auc")
# auc.ch.africa
# auc.hs.africa<-performance(pred.hs.africa, "auc")
# auc.hs.africa
# auc.rf.africa<-performance(pred.rf.africa, "auc")
# auc.rf.africa
# 
# ###Plot not included in paper###
# plot(perf.fl.africa, main="Out of Sample Predictive Accuracy: Africa Data 2001-2014")
# plot(perf.ch.africa, lty=2, add=T)
# plot(perf.hs.africa, lty=3, add=T)
# plot(perf.rf.africa, lty=4, add=T)
# legend(0.55, 0.20, c("Fearon and Laitin (2003) 0.43", "Collier and Hoeffler (2004) 0.55",
#                       "Hegre and Sambanis (2006) 0.40", "Random Forest 0.60" ), lty=c(1,2,3,4), bty="n",
#        cex = .75)
# 
# ###csv file for Table 1###
# d<-data.frame(data3$warstds, predictors.fl, predictors.ch, predictors.hs, predictors.rf)
# write.csv(d, file="CompareCW_dat.csv")