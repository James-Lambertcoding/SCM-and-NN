## Clean data for Neural Networks --------------------

## Create single datafarme for GDP (Constants $2015)
## Datasets include all countries not just specific auto

## 00. Packages -------------------------

library(dplyr)
library(reshape2)
library(stringr)

## 1.0 Data --------------------------
wb_name <- "Data_Pre_Process_NN/NY_GDP_MKTP_KD.csv"
wb_df <- read.csv(wb_name,stringsAsFactors = F)

## 2.0 Cleaning ----------------------
wb_df1 <- wb_df

for(j in 5:ncol(wb_df1)){
  
  wb_df1[,j] <- as.double(wb_df1[,j])
  
}

wb_df2 <- wb_df1[complete.cases(wb_df1),] %>% 
  select(-Country.Code) %>% 
  select(-Series.Name) %>% 
  select(-Series.Code)

colnames(wb_df2)[2:ncol(wb_df2)] <- 1960:2021

## melt and then rejoint to transpose the dataframe but maintain data quality
wb_df3 <- reshape2::melt(wb_df2, id = "Country.Name")

country_name <- unique(wb_df3[,"Country.Name"])
country_name_less_aus <- country_name[!(country_name %in% "Australia")]

list_df <- list()

new_df <- wb_df3 %>% 
  filter(Country.Name == "Australia")

colnames(new_df)[3] <- new_df[1,1]

new_df1 <- new_df %>% 
  select(-Country.Name) %>% 
  rename(year = variable)

for(i in 1:length(country_name_less_aus)){
  
  print(paste0(i,"a"))
  
  list_df[[i]] <- wb_df3 %>% 
    filter(Country.Name == country_name_less_aus[i])
  
  print(paste0(i,"b"))
  
  colnames(list_df[[i]])[3] <- list_df[[i]][1,1]
  
  print(paste0(i,"c"))
  
  list_df[[i]] <- list_df[[i]] %>% 
    select(-Country.Name)%>% 
    rename(year = variable)
  
    print(paste0(i,"d"))
  
  new_df1 <- new_df1 %>% 
    left_join(list_df[[i]], by = "year")
  
  
}

## PIT STOP to create PRM datasets

PRM_df_prior <- new_df1[1:56,c(3:ncol(new_df1),2)] %>% 
  mutate(year = 1960:2015)

PRM_df_prior1 <- PRM_df_prior[,c(ncol(PRM_df_prior),1:(ncol(PRM_df_prior)))]

write.csv(x = PRM_df_prior,file = "PRM Data/PRM_v1.csv")

## 3.0 Normalization ----------
## Linear --------------
new_max <- 1
new_min <- 0.1


new_df2 <- new_df1 %>% 
  select(-year)

for(j in 1:ncol(new_df2)){
  
  new_df2[,j] <- new_df2[,j]
  
}

cal_new_df <- reshape2::melt(data =new_df1, id ="year" )

old_max <- max(cal_new_df[,"value"])
old_min <- min(cal_new_df[,"value"])

coef_linear <- data.frame("year" = "All",
                          "max_coef" = old_max,
                          "min_coef" = old_min)

for(i in 1:nrow(new_df2)){
  
  for(j in 1:ncol(new_df2)){
    
    new_df2[i,j] <- ((new_max-new_min)*((new_df2[i,j]-old_min)/(old_max-old_min)))+new_min

  }
}

## rearrange to but australia last
new_df2 <- new_df2[,c(2:ncol(new_df2),1)]

## Normal Distribution Method --------------

old_mean <- mean(cal_new_df[,"value"])
old_sd <- sd(cal_new_df[,"value"])

coef_ND <- data.frame("year" = "all",
                          "mean_coef" = old_mean,
                          "sd_coef" = old_sd)
new_df2a <- new_df1 %>% 
  select(-year)

new_df2b <-new_df2a

for(i in 1:nrow(new_df2a)){
  
  for(j in 1:ncol(new_df2)){
    
    
    new_df2b[i,j] <- (new_df2a[i,j]-old_mean)/old_sd
  }
  
  
}

new_df2b <- new_df2b[,c(2:ncol(new_df2b),1)]

new_df3_1 <- new_df2[1:56,]
new_df3_2 <- new_df2[57:62,]

new_df3b_1 <- new_df2b[1:56,]
new_df3b_2 <- new_df2b[57:62,]

## Write CSV
write.csv(x = new_df3_1,file = "Data_Pre_Process_NN/GDP_86_linear_prior.csv", row.names = F)
write.csv(x = new_df3_2,file = "Data_Pre_Process_NN/GDP_86_linear_post.csv",row.names = F)

write.csv(x = new_df3b_1,file = "Data_Pre_Process_NN/GDP_86_ND_prior.csv",row.names = F)
write.csv(x = new_df3b_2,file = "Data_Pre_Process_NN/GDP_86_ND_post.csv",row.names = F)

write.csv(x = coef_linear, file = "Data_Pre_Process_NN/coef_linear_86.csv")
write.csv(x = coef_ND, file = "Data_Pre_Process_NN/coef_ND_86.csv")

