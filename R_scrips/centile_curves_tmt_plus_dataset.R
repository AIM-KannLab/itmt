
library("gamlss")
pop_norms <- read.csv("~/r_files/pop_norms.csv")

boys = subset(pop_norms, pop_norms$Gender == 1) 
boys[boys==0] <- NA
boys1<-na.omit(boys)

xsteps = (35 - 4)/31
xlim1 = 4 - xsteps
xlim2 = 35 + xsteps

ysteps = (25 - 5)/20
ylim1 = 5 - ysteps
ylim2 = 25 + ysteps

#******************************************************************#
#**************************BOYS****************************#
#******************************************************************#
k1<-log(length(boys1$x))

m_boys<-gamlss(formula = TMT.PRED.AVG.filtered~pb((Age),method="GAIC",k=k1) + Dataset -1, 
sigma.fo=~pb((Age),method="GAIC",k=k1) + Dataset - 1,
 nu.fo=~pb((Age),method="GAIC",k=k1) + Dataset - 1,
 tau.fo=~pb((Age),method="GAIC",k=k1) + Dataset -1 , family=BCTo,data=boys1, n.cyc=1000)

summary(m_boys)


label_perc_boys <- centiles.pred(m_boys, xvalues=max(boys1$Age), xname= "Age", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_boys <- seq(round(min(boys1$Age),1), round(max(boys1$Age),1), length.out = 100)
centile_values_boys <- centiles.pred(m_boys, xname="Age", xvalues=x_values_boys, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_boys <- centiles.pred(m_boys, xname="Age", xvalues=boys1$Age, yval=boys1$TMT.PRED.AVG.filtered, type="z-scores")

lms_values_boys <- predictAll(m_boys, newdata=data.frame(x=x_values_boys))
chart <- data.frame(lms_values_boys, centile_values_boys)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys.csv' ),"/percentiles_chart_boys.csv", sep=""))
res_chart <- data.frame(resid_m_boys)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys.csv' ),"/res_chart_boys.csv", sep=""))

centiles.fan(m_boys, xvar=boys1$Age, cent = c(3, 10, 25, 50, 75, 90, 97), 
         main.gsub = "@", 
         xleg = min(boys1$Age), yleg = 22, xlim = c(xlim1,xlim2), 
         ylim = c(ylim1,ylim2), points = FALSE,  median = TRUE, pch =  15, 
         cex = 2, col =  gray(0.7),xlab = "Age", ylab = "TMT",main="TMT Growth Chart BCTo",
         colors = c("terrain"), cex.lab=2)
grid(nx = NULL, ny = NULL,lty = 3, col = "gray", lwd = 1)


