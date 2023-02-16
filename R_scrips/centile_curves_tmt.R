
library("gamlss")
pop_norms <- read.csv("~/r_files/pop_norms.csv")

boys = subset(pop_norms, pop_norms$Gender == 1) 
girls = subset(pop_norms, pop_norms$Gender == 2)

x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")

data_perc_boys <- data.frame(boys[,x_axis],boys[,y_axis])
data_perc_girls<- data.frame(girls[,x_axis],girls[,y_axis])

colnames(data_perc_boys)[1] = "x"
colnames(data_perc_boys)[2] = "y"
colnames(data_perc_girls)[1] = "x"
colnames(data_perc_girls)[2] = "y"

data_perc_boys[data_perc_boys==0] <- NA
data_perc_boys1<-na.omit(data_perc_boys)
data_perc_girls[data_perc_girls==0] <- NA
data_perc_girls1<-na.omit(data_perc_girls)

xsteps = (35 - 4)/31
xlim1 = 4 - xsteps
xlim2 = 35 + xsteps

ysteps = (25 - 5)/20
ylim1 = 5 - ysteps
ylim2 = 25 + ysteps

#******************************************************************#
#**************************BOYS****************************#
#******************************************************************#
k1<-log(length(data_perc_boys1$x))
m_gaic_bcto_boys<-gamlss(y~pb((x),method="GAIC",k=k1), sigma.fo=~pb((x),method="GAIC",k=k1), nu.fo=~pb((x),method="GAIC",k=k1),tau.fo=~pb((x),method="GAIC",k=k1), family=BCTo,data=data_perc_boys1, n.cyc=100)
m_gaic_bcpe_boys<-gamlss(y~pb((x),method="GAIC",k=k1), sigma.fo=~pb((x),method="GAIC",k=k1), nu.fo=~pb((x),method="GAIC",k=k1),tau.fo=~pb((x),method="GAIC",k=k1), family=BCPEo,data=data_perc_boys1, n.cyc=100)
m_gaic_bccg_boys<-gamlss(y~pb((x),method="GAIC",k=k1), sigma.fo=~pb((x),method="GAIC",k=k1), nu.fo=~pb((x),method="GAIC",k=k1),tau.fo=~pb((x),method="GAIC",k=k1), family=BCCGo,data=data_perc_boys1, n.cyc=200)

GAIC(m_gaic_bcto_boys, m_gaic_bcpe_boys,m_gaic_bccg_boys, k=log(length(data_perc_boys1$x)))

# == BIC
#df      AIC
#m_gaic_bcto_boys 18.48167 54123.15
#m_gaic_bcpe_boys 38.13110 54417.91
#m_gaic_bccg_boys 51.62367 54771.51

GAIC(m_gaic_bcto_boys, m_gaic_bcpe_boys,m_gaic_bccg_boys, k=log(4))
#                       df      AIC
#m_gaic_bcto_boys 18.48167 53974.64
#m_gaic_bcpe_boys 38.13110 54111.50
#m_gaic_bccg_boys 51.62367 54356.68


newpar <-par(mfrow=c(2,2), mar=par("mar")+c(0,1,0,1),col.axis="black",col='black',col.main='black',col.lab='black',pch="+",cex=0.75,cex.lab=2,cex.axis=1.5,cex.main=2)
plot(m_gaic_bcto_boys,par=newpar)

summary(m_gaic_bcto_boys)
#******************************************************************#
#	      Summary of the Quantile Residuals
#                           mean   =  0.00291804 
#                       variance   =  1.000399 
#               coef. of skewness  =  -0.01990694 
#               coef. of kurtosis  =  3.117666 
#Filliben correlation coefficient  =  0.9986659 
#******************************************************************

label_perc_boys <- centiles.pred(m_gaic_bcto_boys, xvalues=max(data_perc_boys1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_boys <- seq(round(min(data_perc_boys1$x),1), round(max(data_perc_boys1$x),1), length.out = 100)
centile_values_boys <- centiles.pred(m_gaic_bcto_boys, xname="x", xvalues=x_values_boys, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_boys <- centiles.pred(m_gaic_bcto_boys, xname="x", xvalues=data_perc_boys1$x, yval=data_perc_boys1$y, type="z-scores")

lms_values_boys <- predictAll(m_gaic_bcto_boys, newdata=data.frame(x=x_values_boys))
chart <- data.frame(lms_values_boys, centile_values_boys)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys.csv' ),"/percentiles_chart_boys.csv", sep=""))
res_chart <- data.frame(resid_m_boys)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys.csv' ),"/res_chart_boys.csv", sep=""))

newpar <-par(mfrow=c(1,3), mar = c(5, 5, 5, 5),col.axis="black",col='black',col.main='black',col.lab='black',pch="+",cex=0.7,cex.lab=1.2,cex.axis=1,cex.main=1.2)
centiles.fan(m_gaic_bcto_boys, xvar=data_perc_boys1$x, cent = c(3, 10, 25, 50, 75, 90, 97), 
         main.gsub = "@", 
         xleg = min(data_perc_boys1$x), yleg = 22, xlim = c(xlim1,xlim2), 
         ylim = c(ylim1,ylim2), points = FALSE,  median = TRUE, pch =  15, 
         cex = 2, col =  gray(0.7),xlab = "Age", ylab = "TMT",main="TMT Growth Chart BCTo",
         colors = c("terrain"), cex.lab=2)
grid(nx = NULL, ny = NULL,lty = 3, col = "gray", lwd = 1)

centiles.fan(m_gaic_bcpe_boys, xvar=data_perc_boys1$x, cent = c(3, 10, 25, 50, 75, 90, 97), 
         main.gsub = "@", 
         xleg = min(data_perc_boys1$x), yleg = 22, xlim = c(xlim1,xlim2), 
         ylim = c(ylim1,ylim2), points = FALSE,  median = TRUE, pch =  15, 
         cex = 2, col =  gray(0.7),xlab = "Age", ylab = "TMT",main="TMT Growth Chart BCPe",
         colors = c("terrain"), cex.lab=2)
grid(nx = NULL, ny = NULL,lty = 3, col = "gray", lwd = 1)

centiles.fan(m_gaic_bccg_boys, xvar=data_perc_boys1$x, cent = c(3, 10, 25, 50, 75, 90, 97), 
         main.gsub = "@",
         xleg = min(data_perc_boys1$x), yleg = 22, xlim = c(xlim1,xlim2), 
         ylim = c(ylim1,ylim2), points = FALSE,  median = TRUE, pch =  15, 
         cex = 2, col =  gray(0.7),xlab = "Age", ylab = "TMT",main="TMT Growth Chart BCCG",
         colors = c("terrain"), cex.lab=2)
grid(nx = NULL, ny = NULL,lty = 3, col = "gray", lwd = 1)

#******************************************************************#
#**************************GIRLS****************************#
#******************************************************************#

k1<-log(length(data_perc_girls1$x))

m_gaic_bcto_girls<-gamlss(y~pb((x),method="GAIC",k=k1), sigma.fo=~pb((x),method="GAIC",k=k1), nu.fo=~pb((x),method="GAIC",k=k1),tau.fo=~pb((x),method="GAIC",k=k1), family=BCTo,data=data_perc_girls1, n.cyc=100)
m_gaic_bcpe_girls<-gamlss(y~pb((x),method="GAIC",k=k1), sigma.fo=~pb((x),method="GAIC",k=k1), nu.fo=~pb((x),method="GAIC",k=k1),tau.fo=~pb((x),method="GAIC",k=k1), family=BCPEo,data=data_perc_girls1, n.cyc=100)
m_gaic_bccg_girls<-gamlss(y~pb((x),method="GAIC",k=k1), sigma.fo=~pb((x),method="GAIC",k=k1), nu.fo=~pb((x),method="GAIC",k=k1),tau.fo=~pb((x),method="GAIC",k=k1), family=BCCGo,data=data_perc_girls1, n.cyc=200)

GAIC(m_gaic_bcto_girls, m_gaic_bcpe_girls,m_gaic_bccg_girls, k=log(length(data_perc_girls1$x)))
#df      AIC
#m_gaic_bcto_girls 20.39234 49032.72
#m_gaic_bcpe_girls 20.94475 49235.36
#m_gaic_bccg_girls 18.91681 49518.13

GAIC(m_gaic_bcto_girls, m_gaic_bcpe_girls,m_gaic_bccg_girls, k=log(4))
#df      AIC
#m_gaic_bcto_girls 20.39234 48870.61
#m_gaic_bcpe_girls 20.94475 49068.86
#m_gaic_bccg_girls 18.91681 49367.75

newpar <-par(mfrow=c(2,2), mar=par("mar")+c(0,1,0,1),col.axis="black",col='black',col.main='black',col.lab='black',pch="+",cex=0.75,cex.lab=2,cex.axis=1.5,cex.main=2)
plot(m_gaic_bcto_girls,par=newpar)

#******************************************************************
#	      Summary of the Quantile Residuals
#                           mean   =  0.004285232 
#                       variance   =  1.000237 
#               coef. of skewness  =  -0.02569234 
#               coef. of kurtosis  =  3.221659 
#Filliben correlation coefficient  =  0.9979028
#******************************************************************
summary(m_gaic_bcto_girls)

label_perc_girls<- centiles.pred(m_gaic_bcto_girls, xvalues=max(data_perc_girls1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_girls <- seq(round(min(data_perc_girls1$x),1), round(max(data_perc_girls1$x),1), length.out = 100)
centile_values_girls <- centiles.pred(m_gaic_bcto_girls, xname="x", xvalues=x_values_girls, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_girls<- centiles.pred(m_gaic_bcto_girls, xname="x", xvalues=data_perc_girls1$x, yval=data_perc_girls1$y, type="z-scores")

lms_values_boys <- predictAll(m_gaic_bcto_girls, newdata=data.frame(x=x_values_girls))
chart <- data.frame(lms_values_girls, centile_values_girls)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls.csv' ),"/percentiles_chart_girls.csv", sep=""))
res_chart <- data.frame(resid_m_girls)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls.csv' ),"/res_chart_girls.csv", sep=""))

newpar <-par(mfrow=c(1,3), mar = c(5, 5, 5, 5),col.axis="black",col='black',col.main='black',col.lab='black',pch="+",cex=0.7,cex.lab=1.2,cex.axis=1,cex.main=1.2)

centiles.fan(m_gaic_bcto_girls, xvar=data_perc_girls1$x, cent = c(3, 10, 25, 50, 75, 90, 97), 
         main.gsub = "@", 
         xleg = min(data_perc_girls1$x), yleg = 22, xlim = c(xlim1,xlim2), 
         ylim = c(ylim1,ylim2), points = FALSE,  median = TRUE, pch =  15, 
         cex = 2, col =  gray(0.7),xlab = "Age", ylab = "TMT",main="TMT Growth Chart BCTo",
         colors = c("terrain"), cex.lab=2)
grid(nx = NULL, ny = NULL,lty = 3, col = "gray", lwd = 1)

centiles.fan(m_gaic_bcpe_girls, xvar=data_perc_girls1$x, cent = c(3, 10, 25, 50, 75, 90, 97), 
         main.gsub = "@", 
         xleg = min(data_perc_girls1$x), yleg = 22, xlim = c(xlim1,xlim2), 
         ylim = c(ylim1,ylim2), points = FALSE,  median = TRUE, pch =  15, 
         cex = 2, col =  gray(0.7),xlab = "Age", ylab = "TMT",main="TMT Growth Chart BCPe",
         colors = c("terrain"), cex.lab=2)
grid(nx = NULL, ny = NULL,lty = 3, col = "gray", lwd = 1)

centiles.fan(m_gaic_bccg_girls, xvar=data_perc_girls1$x, cent = c(3, 10, 25, 50, 75, 90, 97), 
         main.gsub = "@",
         xleg = min(data_perc_girls1$x), yleg = 22, xlim = c(xlim1,xlim2), 
         ylim = c(ylim1,ylim2), points = FALSE,  median = TRUE, pch =  15, 
         cex = 2, col =  gray(0.7),xlab = "Age", ylab = "TMT",main="TMT Growth Chart BCCG",
         colors = c("terrain"), cex.lab=2)
grid(nx = NULL, ny = NULL,lty = 3, col = "gray", lwd = 1)

newpar <-par(mfrow=c(1,2), mar=par("mar")+c(0,1,0,1),col.axis="black",col='black',col.main='black',col.lab='black',pch="+",cex=0.45,cex.lab=2,cex.axis=1,cex.main=1.2)
wp(m_gaic_bcto_girls,ylim.all = 3, cex.lab=3)
wp(m_gaic_bcto_boys,ylim.all = 3, cex.lab=3)


