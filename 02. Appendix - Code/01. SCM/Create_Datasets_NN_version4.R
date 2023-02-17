## create predictor and answer data set
## for the NN model version 3
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

auto_df1 <- reshape2::melt(auto_df, value.name = "auto") %>% 
  mutate(Series.Code = "auto") %>% 
  rename(year = variable) %>% 
  rename(value= auto) %>% 
  select(Country, Series.Code, year,value) %>% 
  filter(year != 2022)

## world bank halves
wb1_name <- "Data/full_set_1.csv"
wb1_df <- read.csv(wb1_name,stringsAsFactors = F)
wb2_name <- "Data/full_set_2.csv"
wb2_df <- read.csv(wb2_name,stringsAsFactors = F)

colnames(wb1_df)[5:length(colnames(wb1_df))] <- paste0("Y",1997:2021)
colnames(wb2_df)[5:length(colnames(wb2_df))] <- paste0("Y",1972:1996)


wb1_df1 <- wb1_df %>% 
  select(-Country.Code) %>% 
  rename(Country = Country.Name) %>% 
  select(-Series.Name)

wb2_df1 <- wb2_df %>% 
  select(-Country.Code) %>% 
  rename(Country = Country.Name) %>% 
  select(-Series.Name)

## bring two halves together
wb_df1 <- wb2_df1 %>% 
  bind_rows(wb1_df1)


## make numbers are numbers
for(i in 3:length(colnames(wb_df1))){
  
  wb_df1[,i] <- as.double(wb_df1[,i])
  
}

aus_df <- wb_df1 %>% 
  filter(Country == "Australia")


# melt the full lot
wb_df2 <- reshape2::melt(wb_df1) %>% 
  rename(year = variable) %>% 
  bind_rows(auto_df1)

## make years intergers
wb_df2[,"year"] <- as.integer( stringr::str_remove(wb_df2[,"year"],"Y"))

## remove odd data
wb_df3 <- wb_df2 %>% 
  filter(Country !="") %>% 
  filter(Series.Code != "") %>% 
  filter(Country !="Data from database: World Development Indicators") %>% 
  filter(Country !="Last Updated: 12/22/2022")

## sort the unique codes
series_name <- unique(wb_df3[,"Series.Code"])
country_name <- unique(wb_df3[,"Country"])
years_name <- unique(wb_df3[,"year"])

country_name_less_aus <- country_name[!(country_name %in% "Australia")]

## create the list to store each country
list_df <- list()

## begin the new df as Aus
new_df <- wb_df3 %>% 
  filter(Country == "Australia")


## remove na 
new_df1 <- new_df[!is.na(new_df[,"value"]),]

## change value to country name
colnames(new_df1)[4] <- new_df1[1,1]

new_df1 <- new_df1 %>% 
  select(-Country)

for(i in 1:length(country_name_less_aus)){
  
  print(paste0(i,"a"))
  
  list_df[[i]] <- wb_df3 %>% 
    filter(Country == country_name_less_aus[i])
  
  print(paste0(i,"b"))
  
  colnames(list_df[[i]])[4] <- list_df[[i]][1,1]
  
  print(paste0(i,"c"))
  
  list_df[[i]] <- list_df[[i]] %>% 
    select(-Country)
  
  print(colnames(list_df[[i]]))
  
  print(paste0(i,"d"))
  
  new_df1 <- new_df1 %>% 
    left_join(list_df[[i]], by = c("Series.Code","year"))
  
  print(paste0(i,"e"))
  new_df1 <- new_df1[!is.na(new_df1[,country_name_less_aus[i]]),]
  
  print(paste0(i,"f"))
  
}


## double check nas removed
new_df2 <- new_df1[complete.cases(new_df1),]

## normalise the data
coeff_df <- new_df2 %>% 
  select(Series.Code,year) %>% 
  mutate(mean_coef = 0) %>% 
  mutate(sd_coef =0) 

## new data frame to store normalised data
new_df3 <- new_df2

for(i in 1:nrow(new_df3) ){
  
  temp_mean <- mean(as.double(new_df2[i,3:ncol(new_df2)]))
  temp_sd <- sd(as.double(new_df2[i,3:ncol(new_df2)]))
  coeff_df[i,"mean_coef"] <- temp_mean
  coeff_df[i,"sd_coef"] <- temp_sd
  
  new_df3[i,3:(ncol(new_df3)-1)] <- (new_df3[i,3:(ncol(new_df3)-1)]-temp_mean)/temp_sd
  
}

write.csv(x = coeff_df, file = "Plots/NN/long_norm/coeff_df_long_norm.csv", row.names = F)

new_df4 <- new_df3[complete.cases(new_df3),]

## break to find codes --------------
codes <- unique(new_df4[,"Series.Code"])

names_df <- wb1_df[,c("Series.Code","Series.Name","Country.Name")] %>% 
  filter(Country.Name == "Australia") %>% 
  select(-Country.Name)

codex <- data.frame("Series.Code" = codes) %>% 
  left_join(names_df, by = "Series.Code") %>%  
  mutate(data_count =0)

for(i in 1:nrow(codex)){
  
  codex[i,"data_count"] <- sum( new_df4[,"Series.Code"]  %in% codex[i,"Series.Code"])
  
}

boxplot(codex[,"data_count"])

# codex2 <- codex %>% 
#   filter(data_count > 29)

# write.csv(x = codex, file = "Codex/series_data.csv")
# write.csv(x = codex2, file = "Codex/series_data2.csv")

## data of interest -----------------

codes_list <- c("EN.CO2.MANF.ZS",
                "EN.ATM.CO2E.PC",
                "SL.TLF.CACT.ZS",
                "SL.TLF.ACTI.ZS",
                "NV.MNF.TECH.ZS.UN",
                "SL.AGR.EMPL.ZS",
                "SL.IND.EMPL.ZS",
                "SL.SRV.EMPL.ZS",
                "SL.EMP.TOTL.SP.ZS",
                "SL.UEM.TOTL.ZS",
                "TX.VAL.MRCH.XD.WD",
                "NE.CON.GOVT.ZS",
                "NY.GDP.MKTP.CD",
                "TM.VAL.MRCH.XD.WD",
                "NV.AGR.TOTL.ZS",
                "NV.AGR.TOTL.KD",
                "NE.EXP.GNFS.ZS",
                "NY.GDP.MKTP.KD",
                "NY.GDP.PCAP.KD",
                "NV.IND.TOTL.KD",
                "NV.IND.TOTL.ZS",
                "NV.SRV.TOTL.ZS",
                "NV.SRV.TOTL.KD",
                "NE.TRD.GNFS.ZS",
                "NE.GDI.FTOT.ZS",
                "NE.GDI.FTOT.CD",
                "BX.KLT.DINV.WD.GD.ZS",
                "BX.KLT.DINV.CD.WD",
                "NV.IND.MANF.ZS",
                "NV.IND.MANF.CD",
                "auto"
)




code_matrix <- as.data.frame(matrix(ncol = length(codes_list), nrow = nrow(new_df4),0))
colnames(code_matrix) <- codes_list

new_df5 <- new_df4 %>% 
  bind_cols(code_matrix) %>% 
  filter(Series.Code %in% codes_list)



for(i in 1:nrow(new_df5)){
  
  new_df5[i,(ncol(new_df4)+1):ncol(new_df5)] <- (codes_list %in% new_df5[i,"Series.Code"])*T
  
}

## filter prior data
new_df5a <- new_df5 %>% 
  filter(year < 2016)

## filter post data
new_df5b <- new_df5 %>% 
  filter(year >= 2016)

## reorder the colnames
new_df6a <- new_df5a[,c(4:ncol(new_df5a), 3)]
new_df6b <- new_df5b[,c(4:ncol(new_df5b), 3)]

# colnames(new_df6a)
# write.csv(x = new_df6a, file = "Output Data/wb_select_prior_norm.csv")
# write.csv(x = new_df6b, file = "Output Data/wb_select_post_norm.csv")
