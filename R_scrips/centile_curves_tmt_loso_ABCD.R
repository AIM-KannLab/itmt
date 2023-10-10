
library("gamlss")
pop_norms <- read.csv("~/r_files/pop_norms.csv")

boys_abcd = subset(pop_norms, pop_norms$Dataset == 'ABCD' & pop_norms$Gender == 1) 
boys_not_abcd = subset(pop_norms, pop_norms$Dataset != 'ABCD' & pop_norms$Gender == 1)


x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")

data_perc_boys_abcd <- data.frame(boys_abcd[,x_axis],boys_abcd[,y_axis])
data_perc_boys_no_abcd <- data.frame(boys_not_abcd[,x_axis],boys_not_abcd[,y_axis])

colnames(data_perc_boys_abcd)[1] = "x"
colnames(data_perc_boys_abcd)[2] = "y"
colnames(data_perc_boys_no_abcd)[1] = "x"
colnames(data_perc_boys_no_abcd)[2] = "y"

data_perc_boys_abcd[data_perc_boys_abcd==0] <- NA
data_perc_boys_abcd1<-na.omit(data_perc_boys_abcd)
data_perc_boys_no_abcd[data_perc_boys_no_abcd==0] <- NA
data_perc_boys_no_abcd1<-na.omit(data_perc_boys_no_abcd)

xsteps = (35 - 4)/31
xlim1 = 4 - xsteps
xlim2 = 35 + xsteps

ysteps = (25 - 5)/20
ylim1 = 5 - ysteps
ylim2 = 25 + ysteps
# ABCD-specific
k1<-log(length(data_perc_boys_abcd1$x))
m_data_perc_boys_abcd<-gamlss(y~pb((x),method="GAIC",k=k1), sigma.fo=~pb((x),method="GAIC",k=k1), nu.fo=~pb((x),method="GAIC",k=k1),tau.fo=~pb((x),method="GAIC",k=k1), family=BCTo,data=data_perc_boys_abcd1, n.cyc=100)

label_perc_boys <- centiles.pred(m_data_perc_boys_abcd, xvalues=max(data_perc_boys_abcd1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_boys <- seq(round(min(data_perc_boys_abcd1$x),1), round(max(data_perc_boys_abcd1$x),1), length.out = 100)
centile_values_boys <- centiles.pred(m_data_perc_boys_abcd, xname="x", xvalues=x_values_boys, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_boys <- centiles.pred(m_data_perc_boys_abcd, xname="x", xvalues=data_perc_boys_abcd1$x, yval=data_perc_boys_abcd1$y, type="z-scores")

lms_values_boys <- predictAll(m_data_perc_boys_abcd, newdata=data.frame(x=x_values_boys))
chart <- data.frame(lms_values_boys, centile_values_boys)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_abcd.csv' ),"/percentiles_chart_boys_abcd.csv", sep=""))
res_chart <- data.frame(resid_m_boys)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_abcd.csv' ),"/res_chart_boys_abcd.csv", sep=""))

#non-ABCD 
k1<-log(length(data_perc_boys_no_abcd1$x))
m_data_perc_boys_noabcd<-gamlss(y~pb((x),method="GAIC",k=k1), sigma.fo=~pb((x),method="GAIC",k=k1), nu.fo=~pb((x),method="GAIC",k=k1),tau.fo=~pb((x),method="GAIC",k=k1), family=BCTo,data=data_perc_boys_abcd1, n.cyc=100)

label_perc_boys <- centiles.pred(m_data_perc_boys_noabcd, xvalues=max(data_perc_boys_no_abcd1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_boys <- seq(round(min(data_perc_boys_no_abcd1$x),1), round(max(data_perc_boys_no_abcd1$x),1), length.out = 100)
centile_values_boys <- centiles.pred(m_data_perc_boys_noabcd, xname="x", xvalues=x_values_boys, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_boys <- centiles.pred(m_data_perc_boys_noabcd, xname="x", xvalues=data_perc_boys_no_abcd1$x, yval=data_perc_boys_no_abcd1$y, type="z-scores")

lms_values_boys <- predictAll(m_data_perc_boys_noabcd, newdata=data.frame(x=x_values_boys))
chart <- data.frame(lms_values_boys, centile_values_boys)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_noabcd.csv' ),"/percentiles_chart_boys_noabcd.csv", sep=""))
res_chart <- data.frame(resid_m_boys)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_noabcd.csv' ),"/res_chart_boys_noabcd.csv", sep=""))
