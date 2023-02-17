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

# get the auto countries
auto_name <- "Data/auto_vp.csv"
auto_df <- read.csv(auto_name, stringsAsFactors = F)
auto_countries <- unique(auto_df[,"Country"])

## 2.0 Cleaning ----------------------
wb_df1 <- wb_df

for(j in 5:ncol(wb_df1)){
  
  wb_df1[,j] <- as.double(wb_df1[,j])
  
}

colnames(wb_df1)[5:ncol(wb_df1)] <- 1960:2021


## for all auto companies have to reduce years to 1997
wb_df2 <- wb_df1[,c(1:4,42:ncol(wb_df1))] 
  

wb_df2 <- wb_df2[complete.cases(wb_df2),] %>% 
  select(-Country.Code) %>% 
  select(-Series.Name) %>% 
  select(-Series.Code) %>% 
  filter(Country.Name %in% auto_countries)



## melt and then rejoint to transpose the dataframe but maintain data quality
wb_df3 <- reshape2::melt(wb_df2, id = "Country.Name")

## remove non-auto countries


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

nrow(new_df2)

new_df3_1 <- new_df2[1:19,]
new_df3_2 <- new_df2[20:25,]

new_df3b_1 <- new_df2b[1:19,]
new_df3b_2 <- new_df2b[20:25,]

## Write CSV
write.csv(x = new_df3_1,file = "NN_v2/Data_Pre_Process_NN/GDP_41_linear_prior.csv", row.names = F)
write.csv(x = new_df3_2,file = "NN_v2/Data_Pre_Process_NN/GDP_41_linear_post.csv",row.names = F)

write.csv(x = new_df3b_1,file = "NN_v2/Data_Pre_Process_NN/GDP_41_ND_prior.csv",row.names = F)
write.csv(x = new_df3b_2,file = "NN_v2/Data_Pre_Process_NN/GDP_41_ND_post.csv",row.names = F)

write.csv(x = coef_linear, file = "NN_v2/Data_Pre_Process_NN/coef_linear_41.csv")
write.csv(x = coef_ND, file = "NN_v2/Data_Pre_Process_NN/coef_ND_41.csv")

