## create predictor and answer data set
## for the NN model version 5
## normalization is based on mean and SD method
## not on linear version

## 0.0 packages -----------

library(dplyr)
library(reshape2)
library(stringr)

## 1.0 data -------------

## auto df
auto_name <- "Data/auto_vp.csv"
auto_df <- read.csv(auto_name, stringsAsFactors = F)

auto_df1 <- reshape2::melt(auto_df, value.name = "auto", id = "Country") %>% 
  mutate(Series.Code = "auto") %>% 
  rename(year = variable) %>% 
  rename(value= auto) %>% 
  select(Country, Series.Code, year,value) %>% 
  filter(year != 2022)

## 1.1 WB Data --------------------------
wb_name <- "Data_Pre_Process_NN/NY_GDP_MKTP_KD.csv"
wb_df <- read.csv(wb_name,stringsAsFactors = F)

## 1.2 Cleaning ----------------------
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

## 2.0 bring auto and gdp together ---------------
new_df2 <- new_df1 %>% 
  mutate("GDP" = 1) %>%  
  mutate("Auto" = 0) 

auto_len <- length(2000:2021)

auto_temp <- new_df2[1:auto_len,]
auto_temp[,] <-0
auto_temp[,"year"] <- 2000:2021
auto_temp[,"Auto"] <- 1
colnames(auto_temp)

auto_temp1 <- auto_temp

for(j in 2:(ncol(auto_temp)-2)){
  
  if(!is.na(match(colnames(auto_temp)[j], table = auto_df[,"Country"]))){
    
    for(i in 1:nrow(auto_temp)){
      auto_temp1[i,j] <- auto_df[match(colnames(auto_temp)[j], table = auto_df[,"Country"]),i+1]
  }
  
}
  
}  

new_df2[,"year"] <- as.double(new_df2[,"year"])+1959


nn5_df <- new_df2 %>% 
  bind_rows(auto_temp1)

## 3.0 Normalisation ----------------

## Calculate Normlaisation metrics
cal_new_df <- reshape2::melt(data =new_df1, id ="year" )

old_mean_gdp <- mean(cal_new_df[,"value"])
old_sd_gdp <- sd(cal_new_df[,"value"])

coef_ND <- data.frame("dataset" = c("GDP","auto"),
                      "mean_coef" = c(old_mean_gdp,0),
                      "sd_coef" = c(old_sd_gdp,0))  

# colnames(nn5_df)
## need countries in nn5_df
count_nn5 <- colnames(nn5_df)[2:(ncol(nn5_df)-2)]

auto_df2 <- auto_df1 %>% 
  filter(Country %in% count_nn5)

old_mean_auto <- mean(auto_df2[,"value"])
old_sd_auto <- sd(auto_df2[,"value"])

coef_ND[2,"mean_coef"] <- old_mean_auto
coef_ND[2,"sd_coef"] <- old_sd_auto


## Normalize the nn5_df

nn5_df1 <- nn5_df

for(i in 1:nrow(nn5_df1)){
  
  if(nn5_df1[i,"GDP"] == 1){
  
    nn5_df1[i,2:(ncol(nn5_df1)-2)] <- (nn5_df1[i,2:(ncol(nn5_df1)-2)]-coef_ND[1,"mean_coef"])/coef_ND[1,"sd_coef"]
    
  } else {
    
    nn5_df1[i,2:(ncol(nn5_df1)-2)] <- (nn5_df1[i,2:(ncol(nn5_df1)-2)]-coef_ND[2,"mean_coef"])/coef_ND[2,"sd_coef"]
  
  }
  
}

nn5_df_prior <- nn5_df1 %>% 
  filter(year < 2016)
nn5_df_post <- nn5_df1 %>% 
  filter(year >= 2016)

nn5_years <- nn5_df1[,"year"]

# colnames(nn5_df_prior)
nn5_df_prior1 <- nn5_df_prior[,c(3:ncol(nn5_df_prior),2)]
nn5_df_post1 <- nn5_df_post[,c(3:ncol(nn5_df_post),2)]


write.csv(x = nn5_df_prior1, file = "NN_v5/Data for NN/wb_auto_nn5_prior.csv", row.names = F)
write.csv(x = nn5_df_post1, file = "NN_v5/Data for NN/wb_auto_nn5_post.csv", row.names = F)
write.csv(x = coef_ND, file = "NN_v5/Data for NN/coef_nn5.csv", row.names = F)
write.csv(x = nn5_years, file = "NN_v5/Data for NN/years_nn5.csv", row.names = F)


