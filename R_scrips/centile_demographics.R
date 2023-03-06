library("gamlss")
pop_norms <- read.csv("~/r_files/pop_norms_demos.csv")


########### BOYS @@@@@

boys_white = subset(pop_norms, pop_norms$Race == 'White' & pop_norms$Gender == 1) 
boys_asian = subset(pop_norms, pop_norms$Race == 'Asian' & pop_norms$Gender == 1 ) 
boys_black = subset(pop_norms, pop_norms$Race == 'Black'& pop_norms$Gender == 1) 
boys_latino = subset(pop_norms, pop_norms$Race == 'Latino' & pop_norms$Gender == 1) 
boys_mixed = subset(pop_norms, pop_norms$Race == 'Mixed' & pop_norms$Gender == 1) 
boys_other = subset(pop_norms, pop_norms$Race == 'Other' & pop_norms$Gender == 1) 

x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")

data_perc_boys_white <- data.frame(boys_white[,x_axis],boys_white[,y_axis])
data_perc_boys_asian <- data.frame(boys_asian[,x_axis],boys_asian[,y_axis])
data_perc_boys_black <- data.frame(boys_black[,x_axis],boys_black[,y_axis])
data_perc_boys_latino <- data.frame(boys_latino[,x_axis],boys_latino[,y_axis])
data_perc_boys_mixed <- data.frame(boys_mixed[,x_axis],boys_mixed[,y_axis])
data_perc_boys_other <- data.frame(boys_other[,x_axis],boys_other[,y_axis])

colnames(data_perc_boys_white)[1] = "x"
colnames(data_perc_boys_white)[2] = "y"
colnames(data_perc_boys_asian)[1] = "x"
colnames(data_perc_boys_asian)[2] = "y"
colnames(data_perc_boys_black)[1] = "x"
colnames(data_perc_boys_black)[2] = "y"
colnames(data_perc_boys_latino)[1] = "x"
colnames(data_perc_boys_latino)[2] = "y"
colnames(data_perc_boys_mixed)[1] = "x"
colnames(data_perc_boys_mixed)[2] = "y"
colnames(data_perc_boys_other)[1] = "x"
colnames(data_perc_boys_other)[2] = "y"

xsteps = (35 - 4)/31
xlim1 = 4 - xsteps
xlim2 = 35 + xsteps

ysteps = (25 - 5)/20
ylim1 = 5 - ysteps
ylim2 = 25 + ysteps

## white
data_perc_boys_white[data_perc_boys_white==0] <- NA
data_perc_boys_white1<-na.omit(data_perc_boys_white)

m_gaic_bcto_boys_white<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_boys_white1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_white1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_white1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_white1$x))), 
family=BCTo,data=data_perc_boys_white1, n.cyc=100)

label_perc_boys_white <- centiles.pred(m_gaic_bcto_boys_white,
 xvalues=max(data_perc_boys_white1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_boys_white <- seq(round(min(data_perc_boys_white1$x),1), round(max(data_perc_boys_white1$x),1), length.out = 100)
centile_values_boys_white <- centiles.pred(m_gaic_bcto_boys_white,
 xname="x",
 xvalues=x_values_boys_white, 
 cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_boys_white <- centiles.pred(m_gaic_bcto_boys_white, xname="x",
 xvalues=data_perc_boys_white1$x, yval=data_perc_boys_white1$y, type="z-scores")

lms_values_boys_white <- predictAll(m_gaic_bcto_boys_white,
 newdata=data.frame(x=x_values_boys_white))
chart <- data.frame(lms_values_boys_white, centile_values_boys_white)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_white.csv' ),
"/percentiles_chart_boys_white.csv", sep=""))
res_chart <- data.frame(resid_m_boys_white)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_white.csv' ),
"/res_chart_boys_white.csv", sep=""))

## asian
data_perc_boys_asian[data_perc_boys_asian==0] <- NA
data_perc_boys_asian1<-na.omit(data_perc_boys_asian)

m_gaic_bcto_boys_asian<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_boys_asian1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_asian1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_asian1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_asian1$x))), 
family=BCTo,data=data_perc_boys_asian1, n.cyc=30)

label_perc_boys_asian <- centiles.pred(m_gaic_bcto_boys_asian,
 xvalues=max(data_perc_boys_asian1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_boys_asian <- seq(round(min(data_perc_boys_asian1$x),1),
 round(max(data_perc_boys_asian1$x),1), length.out = 100)
centile_values_boys_asian <- centiles.pred(m_gaic_bcto_boys_asian,
 xname="x",
 xvalues=x_values_boys_asian, 
 cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_boys_asian <- centiles.pred(m_gaic_bcto_boys_asian, xname="x",
 xvalues=data_perc_boys_asian1$x, yval=data_perc_boys_asian1$y, type="z-scores")

lms_values_boys_asian <- predictAll(m_gaic_bcto_boys_asian,
 newdata=data.frame(x=x_values_boys_asian))
chart <- data.frame(lms_values_boys_asian, centile_values_boys_asian)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_asian.csv' ),
"/percentiles_chart_boys_asian.csv", sep=""))
res_chart <- data.frame(resid_m_boys_asian)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_asian.csv' ),
"/res_chart_boys_asian.csv", sep=""))

## black
data_perc_boys_black[data_perc_boys_black==0] <- NA
data_perc_boys_black1<-na.omit(data_perc_boys_black)

m_gaic_bcto_boys_black<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_boys_black1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_black1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_black1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_black1$x))), 
family=BCTo,data=data_perc_boys_black1, n.cyc=4)

label_perc_boys_black <- centiles.pred(m_gaic_bcto_boys_black,
 xvalues=max(data_perc_boys_black1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_boys_black <- seq(round(min(data_perc_boys_black1$x),1),
 round(max(data_perc_boys_black1$x),1), length.out = 100)
centile_values_boys_black <- centiles.pred(m_gaic_bcto_boys_black,
 xname="x",
 xvalues=x_values_boys_black, 
 cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_boys_black <- centiles.pred(m_gaic_bcto_boys_black, xname="x",
 xvalues=data_perc_boys_black1$x, yval=data_perc_boys_black1$y, type="z-scores")

lms_values_boys_black <- predictAll(m_gaic_bcto_boys_black,
 newdata=data.frame(x=x_values_boys_black))
chart <- data.frame(lms_values_boys_black, centile_values_boys_black)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_black.csv' ),
"/percentiles_chart_boys_black.csv", sep=""))
res_chart <- data.frame(resid_m_boys_black)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_black.csv' ),
"/res_chart_boys_black.csv", sep=""))

## latino
data_perc_boys_latino[data_perc_boys_latino==0] <- NA
data_perc_boys_latino1<-na.omit(data_perc_boys_latino)

m_gaic_bcto_boys_latino<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_boys_latino1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_latino1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_latino1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_latino1$x))), 
family=BCTo,data=data_perc_boys_latino1, n.cyc=30)

label_perc_boys_latino <- centiles.pred(m_gaic_bcto_boys_latino,
 xvalues=max(data_perc_boys_latino1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_boys_latino <- seq(round(min(data_perc_boys_latino1$x),1),
 round(max(data_perc_boys_latino1$x),1), length.out = 100)
centile_values_boys_latino <- centiles.pred(m_gaic_bcto_boys_latino,
 xname="x",
 xvalues=x_values_boys_latino, 
 cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_boys_latino <- centiles.pred(m_gaic_bcto_boys_latino, xname="x",
 xvalues=data_perc_boys_latino1$x, yval=data_perc_boys_latino1$y, type="z-scores")

lms_values_boys_latino <- predictAll(m_gaic_bcto_boys_latino,
 newdata=data.frame(x=x_values_boys_latino))
chart <- data.frame(lms_values_boys_latino, centile_values_boys_latino)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_latino.csv' ),
"/percentiles_chart_boys_latino.csv", sep=""))
res_chart <- data.frame(resid_m_boys_latino)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_latino.csv' ),
"/res_chart_boys_latino.csv", sep=""))

## mixed
data_perc_boys_mixed[data_perc_boys_mixed==0] <- NA
data_perc_boys_mixed1<-na.omit(data_perc_boys_mixed)

m_gaic_bcto_boys_mixed<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_boys_mixed1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_mixed1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_mixed1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_mixed1$x))), 
family=BCTo,data=data_perc_boys_mixed1, n.cyc=40)

label_perc_boys_mixed <- centiles.pred(m_gaic_bcto_boys_mixed,
 xvalues=max(data_perc_boys_mixed1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_boys_mixed <- seq(round(min(data_perc_boys_mixed1$x),1),
 round(max(data_perc_boys_mixed1$x),1), length.out = 100)
centile_values_boys_mixed <- centiles.pred(m_gaic_bcto_boys_mixed,
 xname="x",
 xvalues=x_values_boys_mixed, 
 cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_boys_mixed <- centiles.pred(m_gaic_bcto_boys_mixed, xname="x",
 xvalues=data_perc_boys_mixed1$x, yval=data_perc_boys_mixed1$y, type="z-scores")

lms_values_boys_mixed <- predictAll(m_gaic_bcto_boys_mixed,
 newdata=data.frame(x=x_values_boys_mixed))
chart <- data.frame(lms_values_boys_mixed, centile_values_boys_mixed)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_mixed.csv' ),
"/percentiles_chart_boys_mixed.csv", sep=""))
res_chart <- data.frame(resid_m_boys_mixed)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_mixed.csv' ),
"/res_chart_boys_mixed.csv", sep=""))

## other
data_perc_boys_other[data_perc_boys_other==0] <- NA
data_perc_boys_other1<-na.omit(data_perc_boys_other)

m_gaic_bcto_boys_other<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_boys_other1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_other1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_other1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_boys_other1$x))), 
family=BCTo,data=data_perc_boys_other1, n.cyc=40)

label_perc_boys_other <- centiles.pred(m_gaic_bcto_boys_other,
 xvalues=max(data_perc_boys_other1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_boys_other <- seq(round(min(data_perc_boys_other1$x),1),
 round(max(data_perc_boys_other1$x),1), length.out = 100)
centile_values_boys_other <- centiles.pred(m_gaic_bcto_boys_other,
 xname="x",
 xvalues=x_values_boys_other, 
 cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_boys_other <- centiles.pred(m_gaic_bcto_boys_other, xname="x",
 xvalues=data_perc_boys_other1$x, yval=data_perc_boys_other1$y, type="z-scores")

lms_values_boys_other <- predictAll(m_gaic_bcto_boys_other,
 newdata=data.frame(x=x_values_boys_other))
chart <- data.frame(lms_values_boys_other, centile_values_boys_other)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_other.csv' ),
"/percentiles_chart_boys_other.csv", sep=""))
res_chart <- data.frame(resid_m_boys_other)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_boys_other.csv' ),
"/res_chart_boys_other.csv", sep=""))

########### GIRLS @@@@@

girls = subset(pop_norms, pop_norms$Gender == 2) 

girls_white = subset(pop_norms, pop_norms$Race == 'White' & pop_norms$Gender == 2) 
girls_asian = subset(pop_norms, pop_norms$Race == 'Asian' & pop_norms$Gender == 2) 
girls_black = subset(pop_norms, pop_norms$Race == 'Black' & pop_norms$Gender == 2 ) 
girls_latino = subset(pop_norms, pop_norms$Race == 'Latino'& pop_norms$Gender == 2) 
girls_mixed = subset(pop_norms, pop_norms$Race == 'Mixed'& pop_norms$Gender == 2) 
girls_other = subset(pop_norms, pop_norms$Race == 'Other'& pop_norms$Gender == 2) 

x_axis = make.names("Age")
y_axis = make.names("TMT.PRED.AVG.filtered")

data_perc_girls_white <- data.frame(girls_white[,x_axis],girls_white[,y_axis])
data_perc_girls_asian <- data.frame(girls_asian[,x_axis],girls_asian[,y_axis])
data_perc_girls_black <- data.frame(girls_black[,x_axis],girls_black[,y_axis])
data_perc_girls_latino <- data.frame(girls_latino[,x_axis],girls_latino[,y_axis])
data_perc_girls_mixed <- data.frame(girls_mixed[,x_axis],girls_mixed[,y_axis])
data_perc_girls_other <- data.frame(girls_other[,x_axis],girls_other[,y_axis])

colnames(data_perc_girls_white)[1] = "x"
colnames(data_perc_girls_white)[2] = "y"
colnames(data_perc_girls_asian)[1] = "x"
colnames(data_perc_girls_asian)[2] = "y"
colnames(data_perc_girls_black)[1] = "x"
colnames(data_perc_girls_black)[2] = "y"
colnames(data_perc_girls_latino)[1] = "x"
colnames(data_perc_girls_latino)[2] = "y"
colnames(data_perc_girls_mixed)[1] = "x"
colnames(data_perc_girls_mixed)[2] = "y"
colnames(data_perc_girls_other)[1] = "x"
colnames(data_perc_girls_other)[2] = "y"

xsteps = (35 - 4)/31
xlim1 = 4 - xsteps
xlim2 = 35 + xsteps

ysteps = (25 - 5)/20
ylim1 = 5 - ysteps
ylim2 = 25 + ysteps

## white
data_perc_girls_white[data_perc_girls_white==0] <- NA
data_perc_girls_white1<-na.omit(data_perc_girls_white)

m_gaic_bcto_girls_white<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_girls_white1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_white1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_white1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_white1$x))), 
family=BCTo,data=data_perc_girls_white1, n.cyc=100)

label_perc_girls_white <- centiles.pred(m_gaic_bcto_girls_white,
 xvalues=max(data_perc_girls_white1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_girls_white <- seq(round(min(data_perc_girls_white1$x),1), round(max(data_perc_girls_white1$x),1), length.out = 100)
centile_values_girls_white <- centiles.pred(m_gaic_bcto_girls_white,
 xname="x",
 xvalues=x_values_girls_white, 
 cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_girls_white <- centiles.pred(m_gaic_bcto_girls_white, xname="x",
 xvalues=data_perc_girls_white1$x, yval=data_perc_girls_white1$y, type="z-scores")

lms_values_girls_white <- predictAll(m_gaic_bcto_girls_white,
 newdata=data.frame(x=x_values_girls_white))
chart <- data.frame(lms_values_girls_white, centile_values_girls_white)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_white.csv' ),
"/percentiles_chart_girls_white.csv", sep=""))
res_chart <- data.frame(resid_m_girls_white)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_white.csv' ),
"/res_chart_girls_white.csv", sep=""))

## asian
data_perc_girls_asian[data_perc_girls_asian==0] <- NA
data_perc_girls_asian1<-na.omit(data_perc_girls_asian)

m_gaic_bcto_girls_asian<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_girls_asian1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_asian1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_asian1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_asian1$x))), 
family=BCTo,data=data_perc_girls_asian1, n.cyc=100)

label_perc_girls_asian <- centiles.pred(m_gaic_bcto_girls_asian,
 xvalues=max(data_perc_girls_asian1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_girls_asian <- seq(round(min(data_perc_girls_asian1$x),1),
 round(max(data_perc_girls_asian1$x),1), length.out = 100)
centile_values_girls_asian <- centiles.pred(m_gaic_bcto_girls_asian,
 xname="x",
 xvalues=x_values_girls_asian, 
 cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_girls_asian <- centiles.pred(m_gaic_bcto_girls_asian, xname="x",
 xvalues=data_perc_girls_asian1$x, yval=data_perc_girls_asian1$y, type="z-scores")

lms_values_girls_asian <- predictAll(m_gaic_bcto_girls_asian,
 newdata=data.frame(x=x_values_girls_asian))
chart <- data.frame(lms_values_girls_asian, centile_values_girls_asian)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_asian.csv' ),
"/percentiles_chart_girls_asian.csv", sep=""))
res_chart <- data.frame(resid_m_girls_asian)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_asian.csv' ),
"/res_chart_girls_asian.csv", sep=""))

## black
data_perc_girls_black[data_perc_girls_black==0] <- NA
data_perc_girls_black1<-na.omit(data_perc_girls_black)

m_gaic_bcto_girls_black<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_girls_black1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_black1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_black1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_black1$x))), 
family=BCTo,data=data_perc_girls_black1, n.cyc=100)

label_perc_girls_black <- centiles.pred(m_gaic_bcto_girls_black,
 xvalues=max(data_perc_girls_black1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_girls_black <- seq(round(min(data_perc_girls_black1$x),1),
 round(max(data_perc_girls_black1$x),1), length.out = 100)
centile_values_girls_black <- centiles.pred(m_gaic_bcto_girls_black,
 xname="x",
 xvalues=x_values_girls_black, 
 cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_girls_black <- centiles.pred(m_gaic_bcto_girls_black, xname="x",
 xvalues=data_perc_girls_black1$x, yval=data_perc_girls_black1$y, type="z-scores")

lms_values_girls_black <- predictAll(m_gaic_bcto_girls_black,
 newdata=data.frame(x=x_values_girls_black))
chart <- data.frame(lms_values_girls_black, centile_values_girls_black)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_black.csv' ),
"/percentiles_chart_girls_black.csv", sep=""))
res_chart <- data.frame(resid_m_girls_black)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_black.csv' ),
"/res_chart_girls_black.csv", sep=""))

## latino
data_perc_girls_latino[data_perc_girls_latino==0] <- NA
data_perc_girls_latino1<-na.omit(data_perc_girls_latino)

m_gaic_bcto_girls_latino<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_girls_latino1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_latino1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_latino1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_latino1$x))), 
family=BCTo,data=data_perc_girls_latino1, n.cyc=100)

label_perc_girls_latino <- centiles.pred(m_gaic_bcto_girls_latino,
 xvalues=max(data_perc_girls_latino1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_girls_latino <- seq(round(min(data_perc_girls_latino1$x),1),
 round(max(data_perc_girls_latino1$x),1), length.out = 100)
centile_values_girls_latino <- centiles.pred(m_gaic_bcto_girls_latino,
 xname="x",
 xvalues=x_values_girls_latino, 
 cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_girls_latino <- centiles.pred(m_gaic_bcto_girls_latino, xname="x",
 xvalues=data_perc_girls_latino1$x, yval=data_perc_girls_latino1$y, type="z-scores")

lms_values_girls_latino <- predictAll(m_gaic_bcto_girls_latino,
 newdata=data.frame(x=x_values_girls_latino))
chart <- data.frame(lms_values_girls_latino, centile_values_girls_latino)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_latino.csv' ),
"/percentiles_chart_girls_latino.csv", sep=""))
res_chart <- data.frame(resid_m_girls_latino)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_latino.csv' ),
"/res_chart_girls_latino.csv", sep=""))

## mixed
data_perc_girls_mixed[data_perc_girls_mixed==0] <- NA
data_perc_girls_mixed1<-na.omit(data_perc_girls_mixed)

m_gaic_bcto_girls_mixed<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_girls_mixed1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_mixed1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_mixed1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_mixed1$x))), 
family=BCTo,data=data_perc_girls_mixed1, n.cyc=100)

label_perc_girls_mixed <- centiles.pred(m_gaic_bcto_girls_mixed,
 xvalues=max(data_perc_girls_mixed1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_girls_mixed <- seq(round(min(data_perc_girls_mixed1$x),1),
 round(max(data_perc_girls_mixed1$x),1), length.out = 100)
centile_values_girls_mixed <- centiles.pred(m_gaic_bcto_girls_mixed,
 xname="x",
 xvalues=x_values_girls_mixed, 
 cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_girls_mixed <- centiles.pred(m_gaic_bcto_girls_mixed, xname="x",
 xvalues=data_perc_girls_mixed1$x, yval=data_perc_girls_mixed1$y, type="z-scores")

lms_values_girls_mixed <- predictAll(m_gaic_bcto_girls_mixed,
 newdata=data.frame(x=x_values_girls_mixed))
chart <- data.frame(lms_values_girls_mixed, centile_values_girls_mixed)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_mixed.csv' ),
"/percentiles_chart_girls_mixed.csv", sep=""))
res_chart <- data.frame(resid_m_girls_mixed)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_mixed.csv' ),
"/res_chart_girls_mixed.csv", sep=""))

## other
data_perc_girls_other[data_perc_girls_other==0] <- NA
data_perc_girls_other1<-na.omit(data_perc_girls_other)

m_gaic_bcto_girls_other<-gamlss(y~pb((x),method="GAIC",k=log(length(data_perc_girls_other1$x))), 
sigma.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_other1$x))), 
nu.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_other1$x))),
tau.fo=~pb((x),method="GAIC",k=log(length(data_perc_girls_other1$x))), 
family=BCTo,data=data_perc_girls_other1, n.cyc=100)

label_perc_girls_other <- centiles.pred(m_gaic_bcto_girls_other,
 xvalues=max(data_perc_girls_other1$x), xname= "x", cent = c(3, 10, 25, 50, 75, 90, 97))
x_values_girls_other <- seq(round(min(data_perc_girls_other1$x),1),
 round(max(data_perc_girls_other1$x),1), length.out = 100)
centile_values_girls_other <- centiles.pred(m_gaic_bcto_girls_other,
 xname="x",
 xvalues=x_values_girls_other, 
 cent = c(3, 10, 25, 50, 75, 90, 97))
resid_m_girls_other <- centiles.pred(m_gaic_bcto_girls_other, xname="x",
 xvalues=data_perc_girls_other1$x, yval=data_perc_girls_other1$y, type="z-scores")

lms_values_girls_other <- predictAll(m_gaic_bcto_girls_other,
 newdata=data.frame(x=x_values_girls_other))
chart <- data.frame(lms_values_girls_other, centile_values_girls_other)
write.csv(chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_other.csv' ),
"/percentiles_chart_girls_other.csv", sep=""))
res_chart <- data.frame(resid_m_girls_other)
write.csv(res_chart, file = paste(dirname('/Users/anna/r_files/cur_data_girls_other.csv' ),
"/res_chart_girls_other.csv", sep=""))