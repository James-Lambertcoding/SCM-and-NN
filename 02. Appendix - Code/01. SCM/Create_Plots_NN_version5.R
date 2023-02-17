## Creating Plots for NN version 2 Outputs

## 0.0 Packages ----------------------

library(ggplot2)
library(ggthemes)
#install.packages("extrafont")
library(extrafont)
library(scales)
library(dplyr)

## 1.0 Data ---------------------------

model_names <- c("Base Model - MSE, ND",
                 "Deep Model - MSE, ND",
                 "Wide Model - MSE, ND",
                 "Base Model - MAE, ND",
                 "Deep Model - MAE, ND",
                 "Wide Model - MAE, ND"
)

## 1.1 History
hist_loc <- "NN_v5/History"
hist_names <- list.files(path = hist_loc, pattern = ".csv")
hist_list <- list()
hist_plot_names <- stringr::str_remove(hist_names,".csv")



for(i in 1:length(hist_names)){
  
  hist_list[[i]] <- read.csv(file = paste0(hist_loc,"/",hist_names[i]), stringsAsFactors = F)
  
}



## 1.2 prior predictions
pred_loc <-"NN_v5/Model_Data"
prior_names <- list.files(path = pred_loc, pattern = "pred_prior_")
prior_list <- list()

for(i in 1:length(prior_names)){
  
  prior_list[[i]] <- read.csv(file = paste0(pred_loc,"/",prior_names[i]), stringsAsFactors = F)
  
}

## 1.3 post predictions
post_names <- list.files(path = pred_loc, pattern = "pred_post_")
post_list <- list()

for(i in 1:length(post_names)){
  
  post_list[[i]] <- read.csv(file = paste0(pred_loc,"/",post_names[i]), stringsAsFactors = F)
  
}

## 1.4 model input data
input_loc <- "NN_v5/Data for NN"
input_name <- list.files(path = input_loc, pattern = ".csv")
input_list <- list()

for(i in 1:length(input_name)){
  
  input_list[[i]] <- read.csv(file = paste0(input_loc,"/",input_name[i]), stringsAsFactors = F)
  
}


## 2.0 Data Manipulation ------------------------
## 2.1 Histories -------------------------------
var_titles <-  c("Loss",
                 "Mean Absolute Error",
                 "Mean Square Error",
                 "Validation Loss",
                 "Validation MAE",
                 "Validation MSE")


hist_list2 <- list()

for(i in 1:length(hist_names)){
  
  hist_list[[i]][,"epoch"] <- as.integer(hist_list[[i]][,"epoch"])+1
  colnames(hist_list[[i]])[1:6] <- var_titles
  hist_list2[[i]] <- reshape2::melt(hist_list[[i]], id = 'epoch',stringsAsFactors = F)
  
  
}


plot_lines <- colnames(hist_list[[1]])[1:(ncol(hist_list[[1]])-1)]

hist_plot_df <- list()
i_len <- length(plot_lines)/2

filter_epoch = 4900

for(k in 1:length(hist_names)){
  hist_plot_df[[k]] <- list()
  
  for(i in 1:i_len){
    
    
    temp_df1 <- hist_list2[[k]] %>% 
      filter(variable == plot_lines[i])
    
    temp_df2 <- hist_list2[[k]] %>% 
      filter(variable == plot_lines[i + i_len])
    
    hist_plot_df[[k]][[i]] <- temp_df1 %>% 
      bind_rows(temp_df2)  %>% 
      filter(epoch >= filter_epoch)
    
    
  }  
}


## 2.2 Predictions -----------------------

input_list[[1]]

## Actual Figures
actual_nd_a <- input_list[[3]] %>% 
  select(Australia, GDP,Auto) 
actual_nd_b <- input_list[[2]] %>% 
  select(Australia, GDP, Auto)
actual_nd <- actual_nd_a %>% 
  bind_rows(actual_nd_b) %>% 
  mutate(year = c(1960:2015,2000:2015,2016:2021,2016:2021))

## prep coef df

actual_nd2 <- actual_nd %>% 
  mutate(mean_coef_gdp = input_list[[1]][1,2]) %>% 
  mutate(sd_coef_gdp = input_list[[1]][1,3]) %>% 
  mutate(mean_coef_auto = input_list[[1]][2,2]) %>% 
  mutate(sd_coef_auto = input_list[[1]][2,3])
  


pred_all_list <- list()
pred_auto_list <- list()
pred_gdp_list <- list()



for(i in 1:length(prior_names)){
  
  pred_all_list[[i]] <- prior_list[[i]] %>% 
    bind_rows(post_list[[i]]) 
  
  colnames(pred_all_list[[i]])[1] <- "nn_aus"
  
  pred_all_list[[i]] <- pred_all_list[[i]] %>% 
      bind_cols(actual_nd2)%>% 
      rename(aus = Australia) %>% 
      mutate(Australia = case_when(
        GDP ==1 ~ (aus*sd_coef_gdp)+mean_coef_gdp,
        T ~   (aus*sd_coef_auto)+mean_coef_auto))%>%
      mutate('NN Australia' = case_when(
        GDP ==1 ~ (nn_aus*sd_coef_gdp)+mean_coef_gdp,
        T ~   (nn_aus*sd_coef_auto)+mean_coef_auto)) %>% 
      select(year, 'NN Australia', Australia, Auto, GDP)
    
  pred_auto_list[[i]] <- pred_all_list[[i]] %>% 
    filter(Auto == 1) %>% 
    select(year, `NN Australia`, Australia)
  pred_gdp_list[[i]] <- pred_all_list[[i]] %>% 
    filter(Auto == 0)%>% 
    select(year, `NN Australia`, Australia)
  pred_auto_list[[i]] <- reshape2::melt(pred_auto_list[[i]], id = "year")
  pred_gdp_list[[i]] <- reshape2::melt(pred_gdp_list[[i]], id = "year")
  
  
  pred_all_list[[i]] <- pred_all_list[[i]] %>% 
    mutate(Error = ((`NN Australia`-Australia)/Australia)*100) %>% 
    mutate(MSE = (Australia - `NN Australia`)^2) %>% 
    mutate(MAE = abs(Australia - `NN Australia`))
  
  }
 
## 3.0 Plot  -------------------------
## 3.1 Histories ---------------------
plots_hist_lists <- list()

for(k in 1:length(hist_names)){
  
  plots_lists[[k]] <- list()
  
  for (i in 1:i_len) {
    
    plots_lists[[k]][[i]] <- ggplot(data = hist_plot_df[[k]][[i]],aes(x=epoch, y = value, group = variable))+
      geom_line(aes(linetype=variable, color=variable))+
      theme(legend.position="top", bg = "white")+
      labs(title = model_names[k])+
      theme_economist_white(gray_bg = FALSE,base_family="ITC Officina Sans") 
    scale_y_continuous(labels = scales::label_number(scale_cut = scales::cut_short_scale()))
    #theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)))
    
    ggplot2::ggsave(filename = paste0("NN_v5/Plots_hist/100 Epoch/hist_plot_full_",hist_plot_names[k],"_",i,".png"),plot = plots_lists[[k]][[i]])  
    
  }
}



## 3.2 Predictions ---------------------

plot_list_pred_gdp <- list()
plot_list_pred_auto <- list()


for(k in 1:length(prior_names)){
  
  ## GDP Plot
  plot_list_pred_gdp[[k]] <- ggplot(data = pred_gdp_list[[k]],aes(x=year, y = value, group = variable))+
    labs(title = paste0(model_names[k]))+
    geom_line(aes(linetype=variable, color=variable))+
    geom_vline(xintercept = 2015, linetype = "dotdash",
               color = "#581845", linewidth =1.5)+
    theme(legend.position="top", bg = "white")+
    theme_economist_white(gray_bg = FALSE,base_family="ITC Officina Sans") +
    scale_y_continuous(labels = scales::label_number(scale_cut = scales::cut_short_scale()))+
    labs(y = "GDP (Constant US$ 2015)")+
    theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)))
  
  ## Save GDP Plot
  ggplot2::ggsave(filename = paste0("NN_v5/Plots_Pred/GDP/25_",k,"_pred_GDP_",model_names[k],".png"),plot = plot_list_pred_gdp[[k]])  
  
  ## Auto Plot
  plot_list_pred_auto[[k]] <- ggplot(data = pred_auto_list[[k]],aes(x=year, y = value, group = variable))+
    labs(title = paste0(model_names[k]))+
    geom_line(aes(linetype=variable, color=variable))+
    geom_vline(xintercept = 2015, linetype = "dotdash",
               color = "#581845", linewidth =1.5)+
    theme(legend.position="top", bg = "white")+
    theme_economist_white(gray_bg = FALSE,base_family="ITC Officina Sans") +
    scale_y_continuous(labels = scales::label_number(scale_cut = scales::cut_short_scale()))+
    labs(y = "GDP (Constant US$ 2015)")+
    theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)))
  
  ggplot2::ggsave(filename = paste0("NN_v5/Plots_Pred/Auto/25_",k,"_pred_Auto_",model_names[k],".png"),plot = plot_list_pred_auto[[k]])  
  
  
  
}

## error Calculator

error_df <- data.frame("model_name" = model_names,
                       "MSE_GDP" = rep(0),
                       "MAE_GDP" = rep(0),
                       "pred_GDP" = rep(0),
                       "MSE_Auto" = rep(0),
                       "MAE_Auto" = rep(0),
                       "pred_Auto" = rep(0))

for(i in 1:nrow(error_df)){
  
  temp__gdp_df <- pred_all_list[[i]] %>% 
    filter(year < 2016) %>% 
    filter(GDP == 1)
  
  temp__auto_df <- pred_all_list[[i]] %>% 
    filter(year < 2016) %>% 
    filter(GDP == 0)

  temp_df_gdp_2 <-pred_all_list[[i]] %>% 
    filter(year == 2019)%>% 
    filter(GDP == 1)
  
  temp_df_auto_2 <-pred_all_list[[i]] %>% 
    filter(year == 2019)%>% 
    filter(GDP == 0)
  
  temp_MSE_gdp <- sum(temp__gdp_df[,"MSE"])/nrow(temp__gdp_df)
  temp_MAE_gdp <- sum(temp__gdp_df[,"MAE"])/nrow(temp__gdp_df)
  temp_MSE_auto <- sum(temp__auto_df[,"MSE"])/nrow(temp__auto_df)
  temp_MAE_auto <- sum(temp__auto_df[,"MAE"])/nrow(temp__auto_df)
  
  
  error_df[i,"MSE_GPD"] <- temp_MSE_gdp
  error_df[i,"MAE_GDP"] <- temp_MAE_gdp
  error_df[i,"pred_GDP"] <- temp_df_gdp_2[,"NN Australia"]
  
  error_df[i,"MSE_Auto"] <- temp_MSE_auto
  error_df[i,"MAE_Auto"] <- temp_MAE_auto
  error_df[i,"pred_Auto"] <- temp_df_auto_2[,"NN Australia"]
  
  
  
}

error_df1 <- error_df %>% 
  arrange(MSE_Auto)
error_df2 <- error_df %>% 
  arrange(MSE_GDP)


write.csv(x = error_df1, file = "NN_v5/Plots_Pred/error_totals_auto.csv")
write.csv(x = error_df2, file = "NN_v5/Plots_Pred/error_totals_gdp.csv")


## care about Deep Model - MSE, ND
## Model 6

pred_all_list[[6]] %>% 
  filter(year ==2019)

write.csv(x = pred_all_list[[6]],"Predictions/NN Model 2 Predictions.csv")
