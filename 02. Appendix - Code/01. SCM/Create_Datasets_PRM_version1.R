## create datasets fro PRM

## this will create the 3 datasets required in the format that PRM code requires

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

prm_c_df <- wb_df %>% 
  filter(Country.Name %in% auto_countries)

for(j in 5:ncol(prm_c_df)){
  
  prm_c_df[,j] <- as.double(prm_c_df[,j])
  
}

prm_c_df_1 <- prm_c_df[complete.cases(prm_c_df),]

colnames(prm_c_df_1)[5:ncol(prm_c_df_1)] <- 1960:2021

prm_c_df_2 <- prm_c_df_1[,c(1,5:ncol(prm_c_df_1))]

## melt and restructure
prm_c_df3 <- reshape2::melt(data = prm_c_df_2, id = "Country.Name")

prm_mean <- mean(prm_c_df3[,"value"])
prm_sd <- sd(prm_c_df3[,"value"])
prm_max <-max(prm_c_df3[,"value"])
prm_min <-min(prm_c_df3[,"value"])
new_min <- 0.1
new_max <- 1

country_name <- unique(prm_c_df3[,"Country.Name"])
country_name_less_aus <- country_name[!(country_name %in% "Australia")]

list_prm_df <- list()

prm_new_df <- prm_c_df3 %>% 
  filter(Country.Name == "Australia")

colnames(prm_new_df)[3] <- prm_new_df[1,1]

prm_new_df1 <- prm_new_df %>% 
  select(-Country.Name) %>% 
  rename(year = variable)

for(i in 1:length(country_name_less_aus)){
  
  print(paste0(i,"a"))
  
  list_prm_df[[i]] <- prm_c_df3 %>% 
    filter(Country.Name == country_name_less_aus[i])
  
  print(paste0(i,"b"))
  
  colnames(list_prm_df[[i]])[3] <- list_prm_df[[i]][1,1]
  
  print(paste0(i,"c"))
  
  list_prm_df[[i]] <- list_prm_df[[i]] %>% 
    select(-Country.Name)%>% 
    rename(year = variable)
  
  print(paste0(i,"d"))
  
  prm_new_df1 <- prm_new_df1 %>% 
    left_join(list_prm_df[[i]], by = "year")
  
  
}

prm_new_df2 <- prm_new_df1[,c(1,3:ncol(prm_new_df1),2)]
prm_new_df3 <- prm_new_df2
prm_new_df4 <- prm_new_df2


for(i in 1:nrow(prm_new_df3)){
  for(j in 2:ncol(prm_new_df3)){
    
    prm_new_df3[i,j] <- (prm_new_df3[i,j]-prm_mean)/prm_sd
    prm_new_df4[i,j] <- ((new_max-new_min)*((prm_new_df4[i,j]-prm_min)/(prm_max-prm_min)))+new_min
    
    
  }
  
}



write.csv(x = prm_new_df2, file = "PRM Data/PRM_v3.csv")
write.csv(x = prm_new_df3, file = "PRM Data/PRM_v3_norm1.csv")
write.csv(x = prm_new_df4, file = "PRM Data/PRM_v3_norm2.csv")

