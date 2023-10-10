
library("gamlss")
pop_norms <- read.csv("~/r_files/pop_norms.csv")


xsteps = (35 - 4)/31
xlim1 = 4 - xsteps
xlim2 = 35 + xsteps

ysteps = (25 - 5)/20
ylim1 = 5 - ysteps
ylim2 = 25 + ysteps

# ABCD-specific
boys_abcd = subset(pop_norms, pop_norms$Dataset != 'ABCD' & pop_norms$Gender == 1) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_boys_abcd <- data.frame(boys_abcd[,x_axis],boys_abcd[,y_axis])
colnames(data_perc_boys_abcd)[1] = "x"
colnames(data_perc_boys_abcd)[2] = "y"
data_perc_boys_abcd[data_perc_boys_abcd==0] <- NA
data_perc_boys_abcd1<-na.omit(data_perc_boys_abcd)
k1<-log(length(data_perc_boys_abcd1$x))
m_data_perc_boys_abcd<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_boys_abcd1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_abcd1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_abcd1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_abcd1$x))), 
family=BCTo,data=data_perc_boys_abcd1, n.cyc=100)
label_perc_boys <- centiles.pred(m_data_perc_boys_abcd, xvalues=max(data_perc_boys_abcd1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_boys <- seq(round(min(data_perc_boys_abcd1$x),1), round(max(data_perc_boys_abcd1$x),1), length.out = 100)
centile_values_boys <- centiles.pred(m_data_perc_boys_abcd, xname="x", xvalues=x_values_boys, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_boys <- centiles.pred(m_data_perc_boys_abcd, xname="x", xvalues=data_perc_boys_abcd1$x, yval=data_perc_boys_abcd1$y, type="z-scores")
lms_values_boys <- predictAll(m_data_perc_boys_abcd, newdata=data.frame(x=x_values_boys))
chart <- data.frame(lms_values_boys, centile_values_boys)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_abcd.csv' ),"/percentiles_chart_boys_abcd.csv", sep=""))
res_chart <- data.frame(resid_m_boys)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_abcd.csv' ),"/res_chart_boys_abcd.csv", sep=""))

# ABIDE-specific
boys_ABIDE = subset(pop_norms, pop_norms$Dataset != 'ABIDE' & pop_norms$Gender == 1) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_boys_ABIDE <- data.frame(boys_ABIDE[,x_axis],boys_ABIDE[,y_axis])
colnames(data_perc_boys_ABIDE)[1] = "x"
colnames(data_perc_boys_ABIDE)[2] = "y"
data_perc_boys_ABIDE[data_perc_boys_ABIDE==0] <- NA
data_perc_boys_ABIDE1<-na.omit(data_perc_boys_ABIDE)
k1<-log(length(data_perc_boys_ABIDE1$x))
m_data_perc_boys_ABIDE<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_boys_ABIDE1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_ABIDE1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_ABIDE1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_ABIDE1$x))), 
family=BCTo,data=data_perc_boys_ABIDE1, n.cyc=100)
label_perc_boys <- centiles.pred(m_data_perc_boys_ABIDE, xvalues=max(data_perc_boys_ABIDE1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_boys <- seq(round(min(data_perc_boys_ABIDE1$x),1), round(max(data_perc_boys_ABIDE1$x),1), length.out = 100)
centile_values_boys <- centiles.pred(m_data_perc_boys_ABIDE, xname="x", xvalues=x_values_boys, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_boys <- centiles.pred(m_data_perc_boys_ABIDE, xname="x", xvalues=data_perc_boys_ABIDE1$x, yval=data_perc_boys_ABIDE1$y, type="z-scores")
lms_values_boys <- predictAll(m_data_perc_boys_ABIDE, newdata=data.frame(x=x_values_boys))
chart <- data.frame(lms_values_boys, centile_values_boys)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_ABIDE.csv' ),"/percentiles_chart_boys_ABIDE.csv", sep=""))
res_chart <- data.frame(resid_m_boys)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_ABIDE.csv' ),"/res_chart_boys_ABIDE.csv", sep=""))

## AOMIC 
boys_AOMIC = subset(pop_norms, pop_norms$Dataset != 'AOMIC' & pop_norms$Gender == 1) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_boys_AOMIC <- data.frame(boys_AOMIC[,x_axis],boys_AOMIC[,y_axis])
colnames(data_perc_boys_AOMIC)[1] = "x"
colnames(data_perc_boys_AOMIC)[2] = "y"
data_perc_boys_AOMIC[data_perc_boys_AOMIC==0] <- NA
data_perc_boys_AOMIC1<-na.omit(data_perc_boys_AOMIC)
k1<-log(length(data_perc_boys_AOMIC1$x))
m_data_perc_boys_AOMIC<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_boys_AOMIC1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_AOMIC1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_AOMIC1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_AOMIC1$x))), 
family=BCTo,data=data_perc_boys_AOMIC1, n.cyc=100)
label_perc_boys <- centiles.pred(m_data_perc_boys_AOMIC, xvalues=max(data_perc_boys_AOMIC1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_boys <- seq(round(min(data_perc_boys_AOMIC1$x),1), round(max(data_perc_boys_AOMIC1$x),1), length.out = 100)
centile_values_boys <- centiles.pred(m_data_perc_boys_AOMIC, xname="x", xvalues=x_values_boys, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_boys <- centiles.pred(m_data_perc_boys_AOMIC, xname="x", xvalues=data_perc_boys_AOMIC1$x, yval=data_perc_boys_AOMIC1$y, type="z-scores")
lms_values_boys <- predictAll(m_data_perc_boys_AOMIC, newdata=data.frame(x=x_values_boys))
chart <- data.frame(lms_values_boys, centile_values_boys)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_AOMIC.csv' ),"/percentiles_chart_boys_AOMIC.csv", sep=""))
res_chart <- data.frame(resid_m_boys)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_AOMIC.csv' ),"/res_chart_boys_AOMIC.csv", sep=""))

## BABY
boys_BABY = subset(pop_norms, pop_norms$Dataset != 'BABY' & pop_norms$Gender == 1) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_boys_BABY <- data.frame(boys_BABY[,x_axis],boys_BABY[,y_axis])
colnames(data_perc_boys_BABY)[1] = "x"
colnames(data_perc_boys_BABY)[2] = "y"
data_perc_boys_BABY[data_perc_boys_BABY==0] <- NA
data_perc_boys_BABY1<-na.omit(data_perc_boys_BABY)
k1<-log(length(data_perc_boys_BABY1$x))
m_data_perc_boys_BABY<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_boys_BABY1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_BABY1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_BABY1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_BABY1$x))), 
family=BCTo,data=data_perc_boys_BABY1, n.cyc=100)
label_perc_boys <- centiles.pred(m_data_perc_boys_BABY, xvalues=max(data_perc_boys_BABY1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_boys <- seq(round(min(data_perc_boys_BABY1$x),1), round(max(data_perc_boys_BABY1$x),1), length.out = 100)
centile_values_boys <- centiles.pred(m_data_perc_boys_BABY, xname="x", xvalues=x_values_boys, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_boys <- centiles.pred(m_data_perc_boys_BABY, xname="x", xvalues=data_perc_boys_BABY1$x, yval=data_perc_boys_BABY1$y, type="z-scores")
lms_values_boys <- predictAll(m_data_perc_boys_BABY, newdata=data.frame(x=x_values_boys))
chart <- data.frame(lms_values_boys, centile_values_boys)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_BABY.csv' ),"/percentiles_chart_boys_BABY.csv", sep=""))
res_chart <- data.frame(resid_m_boys)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_BABY.csv' ),"/res_chart_boys_BABY.csv", sep=""))

##Calgary

boys_Calgary = subset(pop_norms, pop_norms$Dataset != 'Calgary' & pop_norms$Gender == 1) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_boys_Calgary <- data.frame(boys_Calgary[,x_axis],boys_Calgary[,y_axis])
colnames(data_perc_boys_Calgary)[1] = "x"
colnames(data_perc_boys_Calgary)[2] = "y"
data_perc_boys_Calgary[data_perc_boys_Calgary==0] <- NA
data_perc_boys_Calgary1<-na.omit(data_perc_boys_Calgary)
k1<-log(length(data_perc_boys_Calgary1$x))
m_data_perc_boys_Calgary<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_boys_Calgary1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_Calgary1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_Calgary1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_Calgary1$x))), 
family=BCTo,data=data_perc_boys_Calgary1, n.cyc=100)
label_perc_boys <- centiles.pred(m_data_perc_boys_Calgary, xvalues=max(data_perc_boys_Calgary1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_boys <- seq(round(min(data_perc_boys_Calgary1$x),1), round(max(data_perc_boys_Calgary1$x),1), length.out = 100)
centile_values_boys <- centiles.pred(m_data_perc_boys_Calgary, xname="x", xvalues=x_values_boys, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_boys <- centiles.pred(m_data_perc_boys_Calgary, xname="x", xvalues=data_perc_boys_Calgary1$x, yval=data_perc_boys_Calgary1$y, type="z-scores")
lms_values_boys <- predictAll(m_data_perc_boys_Calgary, newdata=data.frame(x=x_values_boys))
chart <- data.frame(lms_values_boys, centile_values_boys)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_Calgary.csv' ),"/percentiles_chart_boys_Calgary.csv", sep=""))
res_chart <- data.frame(resid_m_boys)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_Calgary.csv' ),"/res_chart_boys_Calgary.csv", sep=""))

##HAN
boys_HAN = subset(pop_norms, pop_norms$Dataset != 'HAN' & pop_norms$Gender == 1) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_boys_HAN <- data.frame(boys_HAN[,x_axis],boys_HAN[,y_axis])
colnames(data_perc_boys_HAN)[1] = "x"
colnames(data_perc_boys_HAN)[2] = "y"
data_perc_boys_HAN[data_perc_boys_HAN==0] <- NA
data_perc_boys_HAN1<-na.omit(data_perc_boys_HAN)
k1<-log(length(data_perc_boys_HAN1$x))
m_data_perc_boys_HAN<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_boys_HAN1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_HAN1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_HAN1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_HAN1$x))), 
family=BCTo,data=data_perc_boys_HAN1, n.cyc=100)
label_perc_boys <- centiles.pred(m_data_perc_boys_HAN, xvalues=max(data_perc_boys_HAN1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_boys <- seq(round(min(data_perc_boys_HAN1$x),1), round(max(data_perc_boys_HAN1$x),1), length.out = 100)
centile_values_boys <- centiles.pred(m_data_perc_boys_HAN, xname="x", xvalues=x_values_boys, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_boys <- centiles.pred(m_data_perc_boys_HAN, xname="x", xvalues=data_perc_boys_HAN1$x, yval=data_perc_boys_HAN1$y, type="z-scores")
lms_values_boys <- predictAll(m_data_perc_boys_HAN, newdata=data.frame(x=x_values_boys))
chart <- data.frame(lms_values_boys, centile_values_boys)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_HAN.csv' ),"/percentiles_chart_boys_HAN.csv", sep=""))
res_chart <- data.frame(resid_m_boys)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_HAN.csv' ),"/res_chart_boys_HAN.csv", sep=""))

##HIMN

boys_HIMH = subset(pop_norms, pop_norms$Dataset != 'HIMH' & pop_norms$Gender == 1) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_boys_HIMH <- data.frame(boys_HIMH[,x_axis],boys_HIMH[,y_axis])
colnames(data_perc_boys_HIMH)[1] = "x"
colnames(data_perc_boys_HIMH)[2] = "y"
data_perc_boys_HIMH[data_perc_boys_HIMH==0] <- NA
data_perc_boys_HIMH1<-na.omit(data_perc_boys_HIMH)
k1<-log(length(data_perc_boys_HIMH1$x))
m_data_perc_boys_HIMH<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_boys_HIMH1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_HIMH1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_HIMH1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_HIMH1$x))), 
family=BCTo,data=data_perc_boys_HIMH1, n.cyc=100)
label_perc_boys <- centiles.pred(m_data_perc_boys_HIMH, xvalues=max(data_perc_boys_HIMH1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_boys <- seq(round(min(data_perc_boys_HIMH1$x),1), round(max(data_perc_boys_HIMH1$x),1), length.out = 100)
centile_values_boys <- centiles.pred(m_data_perc_boys_HIMH, xname="x", xvalues=x_values_boys, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_boys <- centiles.pred(m_data_perc_boys_HIMH, xname="x", xvalues=data_perc_boys_HIMH1$x, yval=data_perc_boys_HIMH1$y, type="z-scores")
lms_values_boys <- predictAll(m_data_perc_boys_HIMH, newdata=data.frame(x=x_values_boys))
chart <- data.frame(lms_values_boys, centile_values_boys)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_HIMH.csv' ),"/percentiles_chart_boys_HIMH.csv", sep=""))
res_chart <- data.frame(resid_m_boys)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_HIMH.csv' ),"/res_chart_boys_HIMH.csv", sep=""))

##ICBM
boys_ICBM = subset(pop_norms, pop_norms$Dataset != 'ICBM' & pop_norms$Gender == 1) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_boys_ICBM <- data.frame(boys_ICBM[,x_axis],boys_ICBM[,y_axis])
colnames(data_perc_boys_ICBM)[1] = "x"
colnames(data_perc_boys_ICBM)[2] = "y"
data_perc_boys_ICBM[data_perc_boys_ICBM==0] <- NA
data_perc_boys_ICBM1<-na.omit(data_perc_boys_ICBM)
k1<-log(length(data_perc_boys_ICBM1$x))
m_data_perc_boys_ICBM<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_boys_ICBM1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_ICBM1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_ICBM1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_ICBM1$x))), 
family=BCTo,data=data_perc_boys_ICBM1, n.cyc=100)
label_perc_boys <- centiles.pred(m_data_perc_boys_ICBM, xvalues=max(data_perc_boys_ICBM1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_boys <- seq(round(min(data_perc_boys_ICBM1$x),1), round(max(data_perc_boys_ICBM1$x),1), length.out = 100)
centile_values_boys <- centiles.pred(m_data_perc_boys_ICBM, xname="x", xvalues=x_values_boys, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_boys <- centiles.pred(m_data_perc_boys_ICBM, xname="x", xvalues=data_perc_boys_ICBM1$x, yval=data_perc_boys_ICBM1$y, type="z-scores")
lms_values_boys <- predictAll(m_data_perc_boys_ICBM, newdata=data.frame(x=x_values_boys))
chart <- data.frame(lms_values_boys, centile_values_boys)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_ICBM.csv' ),"/percentiles_chart_boys_ICBM.csv", sep=""))
res_chart <- data.frame(resid_m_boys)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_ICBM.csv' ),"/res_chart_boys_ICBM.csv", sep=""))


##IXI

boys_IXI = subset(pop_norms, pop_norms$Dataset != 'IXI' & pop_norms$Gender == 1) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_boys_IXI <- data.frame(boys_IXI[,x_axis],boys_IXI[,y_axis])
colnames(data_perc_boys_IXI)[1] = "x"
colnames(data_perc_boys_IXI)[2] = "y"
data_perc_boys_IXI[data_perc_boys_IXI==0] <- NA
data_perc_boys_IXI1<-na.omit(data_perc_boys_IXI)
k1<-log(length(data_perc_boys_IXI1$x))
m_data_perc_boys_IXI<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_boys_IXI1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_IXI1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_IXI1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_IXI1$x))), 
family=BCTo,data=data_perc_boys_IXI1, n.cyc=100)
label_perc_boys <- centiles.pred(m_data_perc_boys_IXI, xvalues=max(data_perc_boys_IXI1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_boys <- seq(round(min(data_perc_boys_IXI1$x),1), round(max(data_perc_boys_IXI1$x),1), length.out = 100)
centile_values_boys <- centiles.pred(m_data_perc_boys_IXI, xname="x", xvalues=x_values_boys, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_boys <- centiles.pred(m_data_perc_boys_IXI, xname="x", xvalues=data_perc_boys_IXI1$x, yval=data_perc_boys_IXI1$y, type="z-scores")
lms_values_boys <- predictAll(m_data_perc_boys_IXI, newdata=data.frame(x=x_values_boys))
chart <- data.frame(lms_values_boys, centile_values_boys)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_IXI.csv' ),"/percentiles_chart_boys_IXI.csv", sep=""))
res_chart <- data.frame(resid_m_boys)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_IXI.csv' ),"/res_chart_boys_IXI.csv", sep=""))

##NYU

boys_NYU = subset(pop_norms, pop_norms$Dataset != 'NYU' & pop_norms$Gender == 1) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_boys_NYU <- data.frame(boys_NYU[,x_axis],boys_NYU[,y_axis])
colnames(data_perc_boys_NYU)[1] = "x"
colnames(data_perc_boys_NYU)[2] = "y"
data_perc_boys_NYU[data_perc_boys_NYU==0] <- NA
data_perc_boys_NYU1<-na.omit(data_perc_boys_NYU)
k1<-log(length(data_perc_boys_NYU1$x))
m_data_perc_boys_NYU<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_boys_NYU1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_NYU1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_NYU1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_NYU1$x))), 
family=BCTo,data=data_perc_boys_NYU1, n.cyc=100)
label_perc_boys <- centiles.pred(m_data_perc_boys_NYU, xvalues=max(data_perc_boys_NYU1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_boys <- seq(round(min(data_perc_boys_NYU1$x),1), round(max(data_perc_boys_NYU1$x),1), length.out = 100)
centile_values_boys <- centiles.pred(m_data_perc_boys_NYU, xname="x", xvalues=x_values_boys, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_boys <- centiles.pred(m_data_perc_boys_NYU, xname="x", xvalues=data_perc_boys_NYU1$x, yval=data_perc_boys_NYU1$y, type="z-scores")
lms_values_boys <- predictAll(m_data_perc_boys_NYU, newdata=data.frame(x=x_values_boys))
chart <- data.frame(lms_values_boys, centile_values_boys)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_NYU.csv' ),"/percentiles_chart_boys_NYU.csv", sep=""))
res_chart <- data.frame(resid_m_boys)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_NYU.csv' ),"/res_chart_boys_NYU.csv", sep=""))

##PING
boys_PING = subset(pop_norms, pop_norms$Dataset != 'PING' & pop_norms$Gender == 1) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_boys_PING <- data.frame(boys_PING[,x_axis],boys_PING[,y_axis])
colnames(data_perc_boys_PING)[1] = "x"
colnames(data_perc_boys_PING)[2] = "y"
data_perc_boys_PING[data_perc_boys_PING==0] <- NA
data_perc_boys_PING1<-na.omit(data_perc_boys_PING)
k1<-log(length(data_perc_boys_PING1$x))
m_data_perc_boys_PING<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_boys_PING1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_PING1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_PING1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_PING1$x))), 
family=BCTo,data=data_perc_boys_PING1, n.cyc=100)
label_perc_boys <- centiles.pred(m_data_perc_boys_PING, xvalues=max(data_perc_boys_PING1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_boys <- seq(round(min(data_perc_boys_PING1$x),1), round(max(data_perc_boys_PING1$x),1), length.out = 100)
centile_values_boys <- centiles.pred(m_data_perc_boys_PING, xname="x", xvalues=x_values_boys, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_boys <- centiles.pred(m_data_perc_boys_PING, xname="x", xvalues=data_perc_boys_PING1$x, yval=data_perc_boys_PING1$y, type="z-scores")
lms_values_boys <- predictAll(m_data_perc_boys_PING, newdata=data.frame(x=x_values_boys))
chart <- data.frame(lms_values_boys, centile_values_boys)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_PING.csv' ),"/percentiles_chart_boys_PING.csv", sep=""))
res_chart <- data.frame(resid_m_boys)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_PING.csv' ),"/res_chart_boys_PING.csv", sep=""))


##Pixar

boys_Pixar = subset(pop_norms, pop_norms$Dataset != 'Pixar' & pop_norms$Gender == 1) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_boys_Pixar <- data.frame(boys_Pixar[,x_axis],boys_Pixar[,y_axis])
colnames(data_perc_boys_Pixar)[1] = "x"
colnames(data_perc_boys_Pixar)[2] = "y"
data_perc_boys_Pixar[data_perc_boys_Pixar==0] <- NA
data_perc_boys_Pixar1<-na.omit(data_perc_boys_Pixar)
k1<-log(length(data_perc_boys_Pixar1$x))
m_data_perc_boys_Pixar<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_boys_Pixar1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_Pixar1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_Pixar1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_Pixar1$x))), 
family=BCTo,data=data_perc_boys_Pixar1, n.cyc=100)
label_perc_boys <- centiles.pred(m_data_perc_boys_Pixar, xvalues=max(data_perc_boys_Pixar1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_boys <- seq(round(min(data_perc_boys_Pixar1$x),1), round(max(data_perc_boys_Pixar1$x),1), length.out = 100)
centile_values_boys <- centiles.pred(m_data_perc_boys_Pixar, xname="x", xvalues=x_values_boys, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_boys <- centiles.pred(m_data_perc_boys_Pixar, xname="x", xvalues=data_perc_boys_Pixar1$x, yval=data_perc_boys_Pixar1$y, type="z-scores")
lms_values_boys <- predictAll(m_data_perc_boys_Pixar, newdata=data.frame(x=x_values_boys))
chart <- data.frame(lms_values_boys, centile_values_boys)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_Pixar.csv' ),"/percentiles_chart_boys_Pixar.csv", sep=""))
res_chart <- data.frame(resid_m_boys)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_Pixar.csv' ),"/res_chart_boys_Pixar.csv", sep=""))


##SALD

boys_SALD = subset(pop_norms, pop_norms$Dataset != 'SALD' & pop_norms$Gender == 1) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_boys_SALD <- data.frame(boys_SALD[,x_axis],boys_SALD[,y_axis])
colnames(data_perc_boys_SALD)[1] = "x"
colnames(data_perc_boys_SALD)[2] = "y"
data_perc_boys_SALD[data_perc_boys_SALD==0] <- NA
data_perc_boys_SALD1<-na.omit(data_perc_boys_SALD)
k1<-log(length(data_perc_boys_SALD1$x))
m_data_perc_boys_SALD<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_boys_SALD1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_SALD1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_SALD1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_SALD1$x))), 
family=BCTo,data=data_perc_boys_SALD1, n.cyc=100)
label_perc_boys <- centiles.pred(m_data_perc_boys_SALD, xvalues=max(data_perc_boys_SALD1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_boys <- seq(round(min(data_perc_boys_SALD1$x),1), round(max(data_perc_boys_SALD1$x),1), length.out = 100)
centile_values_boys <- centiles.pred(m_data_perc_boys_SALD, xname="x", xvalues=x_values_boys, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_boys <- centiles.pred(m_data_perc_boys_SALD, xname="x", xvalues=data_perc_boys_SALD1$x, yval=data_perc_boys_SALD1$y, type="z-scores")
lms_values_boys <- predictAll(m_data_perc_boys_SALD, newdata=data.frame(x=x_values_boys))
chart <- data.frame(lms_values_boys, centile_values_boys)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_SALD.csv' ),"/percentiles_chart_boys_SALD.csv", sep=""))
res_chart <- data.frame(resid_m_boys)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_SALD.csv' ),"/res_chart_boys_SALD.csv", sep=""))


### GIRLS

library("gamlss")
pop_norms <- read.csv("~/r_files/pop_norms.csv")

xsteps = (35 - 4)/31
xlim1 = 4 - xsteps
xlim2 = 35 + xsteps

ysteps = (25 - 5)/20
ylim1 = 5 - ysteps
ylim2 = 25 + ysteps

# ABCD-specific
girls_abcd = subset(pop_norms, pop_norms$Dataset != 'ABCD' & pop_norms$Gender == 2) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_girls_abcd <- data.frame(girls_abcd[,x_axis],girls_abcd[,y_axis])
colnames(data_perc_girls_abcd)[1] = "x"
colnames(data_perc_girls_abcd)[2] = "y"
data_perc_girls_abcd[data_perc_girls_abcd==0] <- NA
data_perc_girls_abcd1<-na.omit(data_perc_girls_abcd)
k1<-log(length(data_perc_girls_abcd1$x))
m_data_perc_girls_abcd<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_girls_abcd1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_abcd1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_abcd1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_abcd1$x))), 
family=BCTo,data=data_perc_girls_abcd1, n.cyc=100)
label_perc_girls <- centiles.pred(m_data_perc_girls_abcd, xvalues=max(data_perc_girls_abcd1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_girls <- seq(round(min(data_perc_girls_abcd1$x),1), round(max(data_perc_girls_abcd1$x),1), length.out = 100)
centile_values_girls <- centiles.pred(m_data_perc_girls_abcd, xname="x", xvalues=x_values_girls, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_girls <- centiles.pred(m_data_perc_girls_abcd, xname="x", xvalues=data_perc_girls_abcd1$x, yval=data_perc_girls_abcd1$y, type="z-scores")
lms_values_girls <- predictAll(m_data_perc_girls_abcd, newdata=data.frame(x=x_values_girls))
chart <- data.frame(lms_values_girls, centile_values_girls)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_abcd.csv' ),"/percentiles_chart_girls_abcd.csv", sep=""))
res_chart <- data.frame(resid_m_girls)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_abcd.csv' ),"/res_chart_girls_abcd.csv", sep=""))

# ABIDE-specific
girls_ABIDE = subset(pop_norms, pop_norms$Dataset != 'ABIDE' & pop_norms$Gender == 2) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_girls_ABIDE <- data.frame(girls_ABIDE[,x_axis],girls_ABIDE[,y_axis])
colnames(data_perc_girls_ABIDE)[1] = "x"
colnames(data_perc_girls_ABIDE)[2] = "y"
data_perc_girls_ABIDE[data_perc_girls_ABIDE==0] <- NA
data_perc_girls_ABIDE1<-na.omit(data_perc_girls_ABIDE)
k1<-log(length(data_perc_girls_ABIDE1$x))
m_data_perc_girls_ABIDE<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_girls_ABIDE1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_ABIDE1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_ABIDE1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_ABIDE1$x))), 
family=BCTo,data=data_perc_girls_ABIDE1, n.cyc=100)
label_perc_girls <- centiles.pred(m_data_perc_girls_ABIDE, xvalues=max(data_perc_girls_ABIDE1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_girls <- seq(round(min(data_perc_girls_ABIDE1$x),1), round(max(data_perc_girls_ABIDE1$x),1), length.out = 100)
centile_values_girls <- centiles.pred(m_data_perc_girls_ABIDE, xname="x", xvalues=x_values_girls, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_girls <- centiles.pred(m_data_perc_girls_ABIDE, xname="x", xvalues=data_perc_girls_ABIDE1$x, yval=data_perc_girls_ABIDE1$y, type="z-scores")
lms_values_girls <- predictAll(m_data_perc_girls_ABIDE, newdata=data.frame(x=x_values_girls))
chart <- data.frame(lms_values_girls, centile_values_girls)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_ABIDE.csv' ),"/percentiles_chart_girls_ABIDE.csv", sep=""))
res_chart <- data.frame(resid_m_girls)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_ABIDE.csv' ),"/res_chart_girls_ABIDE.csv", sep=""))

## AOMIC 
girls_AOMIC = subset(pop_norms, pop_norms$Dataset != 'AOMIC' & pop_norms$Gender == 2) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_girls_AOMIC <- data.frame(girls_AOMIC[,x_axis],girls_AOMIC[,y_axis])
colnames(data_perc_girls_AOMIC)[1] = "x"
colnames(data_perc_girls_AOMIC)[2] = "y"
data_perc_girls_AOMIC[data_perc_girls_AOMIC==0] <- NA
data_perc_girls_AOMIC1<-na.omit(data_perc_girls_AOMIC)
k1<-log(length(data_perc_girls_AOMIC1$x))
m_data_perc_girls_AOMIC<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_girls_AOMIC1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_AOMIC1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_AOMIC1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_AOMIC1$x))), 
family=BCTo,data=data_perc_girls_AOMIC1, n.cyc=100)
label_perc_girls <- centiles.pred(m_data_perc_girls_AOMIC, xvalues=max(data_perc_girls_AOMIC1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_girls <- seq(round(min(data_perc_girls_AOMIC1$x),1), round(max(data_perc_girls_AOMIC1$x),1), length.out = 100)
centile_values_girls <- centiles.pred(m_data_perc_girls_AOMIC, xname="x", xvalues=x_values_girls, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_girls <- centiles.pred(m_data_perc_girls_AOMIC, xname="x", xvalues=data_perc_girls_AOMIC1$x, yval=data_perc_girls_AOMIC1$y, type="z-scores")
lms_values_girls <- predictAll(m_data_perc_girls_AOMIC, newdata=data.frame(x=x_values_girls))
chart <- data.frame(lms_values_girls, centile_values_girls)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_AOMIC.csv' ),"/percentiles_chart_girls_AOMIC.csv", sep=""))
res_chart <- data.frame(resid_m_girls)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_AOMIC.csv' ),"/res_chart_girls_AOMIC.csv", sep=""))

## BABY
girls_BABY = subset(pop_norms, pop_norms$Dataset != 'BABY' & pop_norms$Gender == 2) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_girls_BABY <- data.frame(girls_BABY[,x_axis],girls_BABY[,y_axis])
colnames(data_perc_girls_BABY)[1] = "x"
colnames(data_perc_girls_BABY)[2] = "y"
data_perc_girls_BABY[data_perc_girls_BABY==0] <- NA
data_perc_girls_BABY1<-na.omit(data_perc_girls_BABY)
k1<-log(length(data_perc_girls_BABY1$x))
m_data_perc_girls_BABY<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_girls_BABY1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_BABY1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_BABY1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_BABY1$x))), 
family=BCTo,data=data_perc_girls_BABY1, n.cyc=100)
label_perc_girls <- centiles.pred(m_data_perc_girls_BABY, xvalues=max(data_perc_girls_BABY1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_girls <- seq(round(min(data_perc_girls_BABY1$x),1), round(max(data_perc_girls_BABY1$x),1), length.out = 100)
centile_values_girls <- centiles.pred(m_data_perc_girls_BABY, xname="x", xvalues=x_values_girls, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_girls <- centiles.pred(m_data_perc_girls_BABY, xname="x", xvalues=data_perc_girls_BABY1$x, yval=data_perc_girls_BABY1$y, type="z-scores")
lms_values_girls <- predictAll(m_data_perc_girls_BABY, newdata=data.frame(x=x_values_girls))
chart <- data.frame(lms_values_girls, centile_values_girls)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_BABY.csv' ),"/percentiles_chart_girls_BABY.csv", sep=""))
res_chart <- data.frame(resid_m_girls)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_BABY.csv' ),"/res_chart_girls_BABY.csv", sep=""))

##Calgary

girls_Calgary = subset(pop_norms, pop_norms$Dataset != 'Calgary' & pop_norms$Gender == 2) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_girls_Calgary <- data.frame(girls_Calgary[,x_axis],girls_Calgary[,y_axis])
colnames(data_perc_girls_Calgary)[1] = "x"
colnames(data_perc_girls_Calgary)[2] = "y"
data_perc_girls_Calgary[data_perc_girls_Calgary==0] <- NA
data_perc_girls_Calgary1<-na.omit(data_perc_girls_Calgary)
k1<-log(length(data_perc_girls_Calgary1$x))
m_data_perc_girls_Calgary<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_girls_Calgary1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_Calgary1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_Calgary1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_Calgary1$x))), 
family=BCTo,data=data_perc_girls_Calgary1, n.cyc=100)
label_perc_girls <- centiles.pred(m_data_perc_girls_Calgary, xvalues=max(data_perc_girls_Calgary1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_girls <- seq(round(min(data_perc_girls_Calgary1$x),1), round(max(data_perc_girls_Calgary1$x),1), length.out = 100)
centile_values_girls <- centiles.pred(m_data_perc_girls_Calgary, xname="x", xvalues=x_values_girls, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_girls <- centiles.pred(m_data_perc_girls_Calgary, xname="x", xvalues=data_perc_girls_Calgary1$x, yval=data_perc_girls_Calgary1$y, type="z-scores")
lms_values_girls <- predictAll(m_data_perc_girls_Calgary, newdata=data.frame(x=x_values_girls))
chart <- data.frame(lms_values_girls, centile_values_girls)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_Calgary.csv' ),"/percentiles_chart_girls_Calgary.csv", sep=""))
res_chart <- data.frame(resid_m_girls)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_Calgary.csv' ),"/res_chart_girls_Calgary.csv", sep=""))

##HAN
girls_HAN = subset(pop_norms, pop_norms$Dataset != 'HAN' & pop_norms$Gender == 2) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_girls_HAN <- data.frame(girls_HAN[,x_axis],girls_HAN[,y_axis])
colnames(data_perc_girls_HAN)[1] = "x"
colnames(data_perc_girls_HAN)[2] = "y"
data_perc_girls_HAN[data_perc_girls_HAN==0] <- NA
data_perc_girls_HAN1<-na.omit(data_perc_girls_HAN)
k1<-log(length(data_perc_girls_HAN1$x))
m_data_perc_girls_HAN<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_girls_HAN1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_HAN1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_HAN1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_HAN1$x))), 
family=BCTo,data=data_perc_girls_HAN1, n.cyc=100)
label_perc_girls <- centiles.pred(m_data_perc_girls_HAN, xvalues=max(data_perc_girls_HAN1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_girls <- seq(round(min(data_perc_girls_HAN1$x),1), round(max(data_perc_girls_HAN1$x),1), length.out = 100)
centile_values_girls <- centiles.pred(m_data_perc_girls_HAN, xname="x", xvalues=x_values_girls, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_girls <- centiles.pred(m_data_perc_girls_HAN, xname="x", xvalues=data_perc_girls_HAN1$x, yval=data_perc_girls_HAN1$y, type="z-scores")
lms_values_girls <- predictAll(m_data_perc_girls_HAN, newdata=data.frame(x=x_values_girls))
chart <- data.frame(lms_values_girls, centile_values_girls)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_HAN.csv' ),"/percentiles_chart_girls_HAN.csv", sep=""))
res_chart <- data.frame(resid_m_girls)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_HAN.csv' ),"/res_chart_girls_HAN.csv", sep=""))

##HIMN

girls_HIMH = subset(pop_norms, pop_norms$Dataset != 'HIMH' & pop_norms$Gender == 2) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_girls_HIMH <- data.frame(girls_HIMH[,x_axis],girls_HIMH[,y_axis])
colnames(data_perc_girls_HIMH)[1] = "x"
colnames(data_perc_girls_HIMH)[2] = "y"
data_perc_girls_HIMH[data_perc_girls_HIMH==0] <- NA
data_perc_girls_HIMH1<-na.omit(data_perc_girls_HIMH)
k1<-log(length(data_perc_girls_HIMH1$x))
m_data_perc_girls_HIMH<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_girls_HIMH1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_HIMH1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_HIMH1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_HIMH1$x))), 
family=BCTo,data=data_perc_girls_HIMH1, n.cyc=100)
label_perc_girls <- centiles.pred(m_data_perc_girls_HIMH, xvalues=max(data_perc_girls_HIMH1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_girls <- seq(round(min(data_perc_girls_HIMH1$x),1), round(max(data_perc_girls_HIMH1$x),1), length.out = 100)
centile_values_girls <- centiles.pred(m_data_perc_girls_HIMH, xname="x", xvalues=x_values_girls, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_girls <- centiles.pred(m_data_perc_girls_HIMH, xname="x", xvalues=data_perc_girls_HIMH1$x, yval=data_perc_girls_HIMH1$y, type="z-scores")
lms_values_girls <- predictAll(m_data_perc_girls_HIMH, newdata=data.frame(x=x_values_girls))
chart <- data.frame(lms_values_girls, centile_values_girls)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_HIMH.csv' ),"/percentiles_chart_girls_HIMH.csv", sep=""))
res_chart <- data.frame(resid_m_girls)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_HIMH.csv' ),"/res_chart_girls_HIMH.csv", sep=""))

##ICBM
girls_ICBM = subset(pop_norms, pop_norms$Dataset != 'ICBM' & pop_norms$Gender == 2) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_girls_ICBM <- data.frame(girls_ICBM[,x_axis],girls_ICBM[,y_axis])
colnames(data_perc_girls_ICBM)[1] = "x"
colnames(data_perc_girls_ICBM)[2] = "y"
data_perc_girls_ICBM[data_perc_girls_ICBM==0] <- NA
data_perc_girls_ICBM1<-na.omit(data_perc_girls_ICBM)
k1<-log(length(data_perc_girls_ICBM1$x))
m_data_perc_girls_ICBM<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_girls_ICBM1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_ICBM1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_ICBM1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_ICBM1$x))), 
family=BCTo,data=data_perc_girls_ICBM1, n.cyc=100)
label_perc_girls <- centiles.pred(m_data_perc_girls_ICBM, xvalues=max(data_perc_girls_ICBM1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_girls <- seq(round(min(data_perc_girls_ICBM1$x),1), round(max(data_perc_girls_ICBM1$x),1), length.out = 100)
centile_values_girls <- centiles.pred(m_data_perc_girls_ICBM, xname="x", xvalues=x_values_girls, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_girls <- centiles.pred(m_data_perc_girls_ICBM, xname="x", xvalues=data_perc_girls_ICBM1$x, yval=data_perc_girls_ICBM1$y, type="z-scores")
lms_values_girls <- predictAll(m_data_perc_girls_ICBM, newdata=data.frame(x=x_values_girls))
chart <- data.frame(lms_values_girls, centile_values_girls)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_ICBM.csv' ),"/percentiles_chart_girls_ICBM.csv", sep=""))
res_chart <- data.frame(resid_m_girls)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_ICBM.csv' ),"/res_chart_girls_ICBM.csv", sep=""))


##IXI

girls_IXI = subset(pop_norms, pop_norms$Dataset != 'IXI' & pop_norms$Gender == 2) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_girls_IXI <- data.frame(girls_IXI[,x_axis],girls_IXI[,y_axis])
colnames(data_perc_girls_IXI)[1] = "x"
colnames(data_perc_girls_IXI)[2] = "y"
data_perc_girls_IXI[data_perc_girls_IXI==0] <- NA
data_perc_girls_IXI1<-na.omit(data_perc_girls_IXI)
k1<-log(length(data_perc_girls_IXI1$x))
m_data_perc_girls_IXI<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_girls_IXI1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_IXI1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_IXI1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_IXI1$x))), 
family=BCTo,data=data_perc_girls_IXI1, n.cyc=100)
label_perc_girls <- centiles.pred(m_data_perc_girls_IXI, xvalues=max(data_perc_girls_IXI1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_girls <- seq(round(min(data_perc_girls_IXI1$x),1), round(max(data_perc_girls_IXI1$x),1), length.out = 100)
centile_values_girls <- centiles.pred(m_data_perc_girls_IXI, xname="x", xvalues=x_values_girls, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_girls <- centiles.pred(m_data_perc_girls_IXI, xname="x", xvalues=data_perc_girls_IXI1$x, yval=data_perc_girls_IXI1$y, type="z-scores")
lms_values_girls <- predictAll(m_data_perc_girls_IXI, newdata=data.frame(x=x_values_girls))
chart <- data.frame(lms_values_girls, centile_values_girls)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_IXI.csv' ),"/percentiles_chart_girls_IXI.csv", sep=""))
res_chart <- data.frame(resid_m_girls)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_IXI.csv' ),"/res_chart_girls_IXI.csv", sep=""))

##NYU

girls_NYU = subset(pop_norms, pop_norms$Dataset != 'NYU' & pop_norms$Gender == 2) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_girls_NYU <- data.frame(girls_NYU[,x_axis],girls_NYU[,y_axis])
colnames(data_perc_girls_NYU)[1] = "x"
colnames(data_perc_girls_NYU)[2] = "y"
data_perc_girls_NYU[data_perc_girls_NYU==0] <- NA
data_perc_girls_NYU1<-na.omit(data_perc_girls_NYU)
k1<-log(length(data_perc_girls_NYU1$x))
m_data_perc_girls_NYU<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_girls_NYU1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_NYU1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_NYU1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_NYU1$x))), 
family=BCTo,data=data_perc_girls_NYU1, n.cyc=100)
label_perc_girls <- centiles.pred(m_data_perc_girls_NYU, xvalues=max(data_perc_girls_NYU1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_girls <- seq(round(min(data_perc_girls_NYU1$x),1), round(max(data_perc_girls_NYU1$x),1), length.out = 100)
centile_values_girls <- centiles.pred(m_data_perc_girls_NYU, xname="x", xvalues=x_values_girls, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_girls <- centiles.pred(m_data_perc_girls_NYU, xname="x", xvalues=data_perc_girls_NYU1$x, yval=data_perc_girls_NYU1$y, type="z-scores")
lms_values_girls <- predictAll(m_data_perc_girls_NYU, newdata=data.frame(x=x_values_girls))
chart <- data.frame(lms_values_girls, centile_values_girls)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_NYU.csv' ),"/percentiles_chart_girls_NYU.csv", sep=""))
res_chart <- data.frame(resid_m_girls)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_NYU.csv' ),"/res_chart_girls_NYU.csv", sep=""))

##PING
girls_PING = subset(pop_norms, pop_norms$Dataset != 'PING' & pop_norms$Gender == 2) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_girls_PING <- data.frame(girls_PING[,x_axis],girls_PING[,y_axis])
colnames(data_perc_girls_PING)[1] = "x"
colnames(data_perc_girls_PING)[2] = "y"
data_perc_girls_PING[data_perc_girls_PING==0] <- NA
data_perc_girls_PING1<-na.omit(data_perc_girls_PING)
k1<-log(length(data_perc_girls_PING1$x))
m_data_perc_girls_PING<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_girls_PING1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_PING1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_PING1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_PING1$x))), 
family=BCTo,data=data_perc_girls_PING1, n.cyc=100)
label_perc_girls <- centiles.pred(m_data_perc_girls_PING, xvalues=max(data_perc_girls_PING1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_girls <- seq(round(min(data_perc_girls_PING1$x),1), round(max(data_perc_girls_PING1$x),1), length.out = 100)
centile_values_girls <- centiles.pred(m_data_perc_girls_PING, xname="x", xvalues=x_values_girls, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_girls <- centiles.pred(m_data_perc_girls_PING, xname="x", xvalues=data_perc_girls_PING1$x, yval=data_perc_girls_PING1$y, type="z-scores")
lms_values_girls <- predictAll(m_data_perc_girls_PING, newdata=data.frame(x=x_values_girls))
chart <- data.frame(lms_values_girls, centile_values_girls)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_PING.csv' ),"/percentiles_chart_girls_PING.csv", sep=""))
res_chart <- data.frame(resid_m_girls)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_PING.csv' ),"/res_chart_girls_PING.csv", sep=""))


##Pixar

girls_Pixar = subset(pop_norms, pop_norms$Dataset != 'Pixar' & pop_norms$Gender == 2) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_girls_Pixar <- data.frame(girls_Pixar[,x_axis],girls_Pixar[,y_axis])
colnames(data_perc_girls_Pixar)[1] = "x"
colnames(data_perc_girls_Pixar)[2] = "y"
data_perc_girls_Pixar[data_perc_girls_Pixar==0] <- NA
data_perc_girls_Pixar1<-na.omit(data_perc_girls_Pixar)
k1<-log(length(data_perc_girls_Pixar1$x))
m_data_perc_girls_Pixar<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_girls_Pixar1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_Pixar1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_Pixar1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_Pixar1$x))), 
family=BCTo,data=data_perc_girls_Pixar1, n.cyc=100)
label_perc_girls <- centiles.pred(m_data_perc_girls_Pixar, xvalues=max(data_perc_girls_Pixar1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_girls <- seq(round(min(data_perc_girls_Pixar1$x),1), round(max(data_perc_girls_Pixar1$x),1), length.out = 100)
centile_values_girls <- centiles.pred(m_data_perc_girls_Pixar, xname="x", xvalues=x_values_girls, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_girls <- centiles.pred(m_data_perc_girls_Pixar, xname="x", xvalues=data_perc_girls_Pixar1$x, yval=data_perc_girls_Pixar1$y, type="z-scores")
lms_values_girls <- predictAll(m_data_perc_girls_Pixar, newdata=data.frame(x=x_values_girls))
chart <- data.frame(lms_values_girls, centile_values_girls)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_Pixar.csv' ),"/percentiles_chart_girls_Pixar.csv", sep=""))
res_chart <- data.frame(resid_m_girls)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_Pixar.csv' ),"/res_chart_girls_Pixar.csv", sep=""))


##SALD

girls_SALD = subset(pop_norms, pop_norms$Dataset != 'SALD' & pop_norms$Gender == 2) 
x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")
data_perc_girls_SALD <- data.frame(girls_SALD[,x_axis],girls_SALD[,y_axis])
colnames(data_perc_girls_SALD)[1] = "x"
colnames(data_perc_girls_SALD)[2] = "y"
data_perc_girls_SALD[data_perc_girls_SALD==0] <- NA
data_perc_girls_SALD1<-na.omit(data_perc_girls_SALD)
k1<-log(length(data_perc_girls_SALD1$x))
m_data_perc_girls_SALD<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_girls_SALD1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_SALD1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_SALD1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_SALD1$x))), 
family=BCTo,data=data_perc_girls_SALD1, n.cyc=100)
label_perc_girls <- centiles.pred(m_data_perc_girls_SALD, xvalues=max(data_perc_girls_SALD1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_girls <- seq(round(min(data_perc_girls_SALD1$x),1), round(max(data_perc_girls_SALD1$x),1), length.out = 100)
centile_values_girls <- centiles.pred(m_data_perc_girls_SALD, xname="x", xvalues=x_values_girls, cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_girls <- centiles.pred(m_data_perc_girls_SALD, xname="x", xvalues=data_perc_girls_SALD1$x, yval=data_perc_girls_SALD1$y, type="z-scores")
lms_values_girls <- predictAll(m_data_perc_girls_SALD, newdata=data.frame(x=x_values_girls))
chart <- data.frame(lms_values_girls, centile_values_girls)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_SALD.csv' ),"/percentiles_chart_girls_SALD.csv", sep=""))
res_chart <- data.frame(resid_m_girls)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_SALD.csv' ),"/res_chart_girls_SALD.csv", sep=""))


