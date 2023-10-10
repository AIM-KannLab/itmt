library(tidyverse)
library(knitr)
library(effects)
library(psych)
library(candisc)
#1 = Less than $5,000 ; 2 = $5,000 through $11,999 ;
# 3 = $12,000 through $15,999 ; 4 = $16,000 through $24,999 ;
# 5 = $25,000 through $34,999 ; 6 = $35,000 through $49,999 ; 
#7 = $50,000 through $74,999 ; 8 = $75,000 through $99,999 ; 
#9 = $100,000 through $199,999 ; 10 = $200,000 and greater ; 
# 999, Don't know ; 777, Refuse to answer

# select from dplyr
select <- dplyr::select
recode <- dplyr::recode

ethnic <- read.csv("~/r_files/ethnic_ds.csv")
ethnic <- ethnic %>% mutate(steps_c = steps - mean(steps, na.rm = TRUE))
ethnic$income <- factor(ethnic$income)
ethnic$race <- factor(ethnic$race)
ethnic <- ethnic %>% mutate(dtcal_c = dtcal - mean(dtcal, na.rm = TRUE))

scimath1 <- lm(tile ~ steps_c+dtcal_c+income, data = ethnic)
scimath2 <- lm(tile ~ steps_c+dtcal_c+race+income, data = ethnic)
anova(scimath1, scimath2)

'''Residuals:
    Min      1Q  Median      3Q     Max 
-55.133 -21.049  -4.396  18.849  67.287 

Coefficients:
                     Estimate Std. Error t value Pr(>|t|)    
(Intercept)         5.145e+01  4.836e+00  10.637  < 2e-16 ***
steps_c            -6.078e-05  1.781e-05  -3.413 0.000651 ***
dtcal_c             2.689e-03  1.151e-03   2.336 0.019539 *  
raceBlack           9.781e+00  3.448e+00   2.837 0.004591 ** 
raceLatino          3.370e-01  3.369e+00   0.100 0.920332    
raceMixed           8.610e-02  3.486e+00   0.025 0.980297    
raceOther          -1.073e+01  4.190e+00  -2.561 0.010502 *  
raceWhite/European -9.119e+00  3.103e+00  -2.938 0.003325 ** 
income2             1.725e+00  4.736e+00   0.364 0.715629    
income3             4.353e+00  5.332e+00   0.817 0.414281    
income4             4.629e+00  4.555e+00   1.016 0.309630    
income5            -7.437e-01  4.202e+00  -0.177 0.859528    
income6            -8.026e+00  4.095e+00  -1.960 0.050065 .  
income7            -3.528e+00  3.925e+00  -0.899 0.368798    
income8            -7.371e+00  3.941e+00  -1.871 0.061506 .  
income9            -9.847e+00  3.831e+00  -2.571 0.010205 *  
income10           -1.299e+01  4.006e+00  -3.242 0.001200 ** 
income777          -9.978e+00  4.640e+00  -2.151 0.031593 *  
income999          -1.012e+01  4.579e+00  -2.211 0.027148 *  
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 25.7 on 2833 degrees of freedom
Multiple R-squared:  0.1237,	Adjusted R-squared:  0.1182 
F-statistic: 22.23 on 18 and 2833 DF,  p-value: < 2.2e-16

Analysis of Variance Table

Model 1: tile ~ steps_c + dtcal_c + income
Model 2: tile ~ steps_c + dtcal_c + race + income
  Res.Df     RSS Df Sum of Sq      F    Pr(>F)    
1   2838 1968970                                  
2   2833 1870893  5     98077 29.703 < 2.2e-16 ***'''

