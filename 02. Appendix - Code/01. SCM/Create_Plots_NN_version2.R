## Creating Plots for NN version 2 Outputs

## 0.0 Packages ----------------------

library(ggplot2)
library(ggthemes)
#install.packages("extrafont")
library(extrafont)
library(scales)
library(dplyr)

## 1.0 Data ---------------------------

model_names <- c("Base Model - MSE, Linear",
                 "Base Model - MSE, ND",
                 "Deep Model - MSE, Linear",
                 "Deep Model - MSE, ND",
                 "Wide Model - MSE, Linear",
                 "Wide Model - MSE, ND",
                 "Base Model - MAE, Linear",
                 "Base Model - MAE, ND",
                 "Deep Model - MAE, Linear",
                 "Deep Model - MAE, ND",
                 "Wide Model - MAE, Linear",
                 "Wide Model - MAE, ND"
)

## 1.1 History
hist_loc <- "NN_v2/History"
hist_names <- list.files(path = hist_loc, pattern = ".csv")
hist_list <- list()
hist_plot_names <- stringr::str_remove(hist_names,".csv")



for(i in 1:length(hist_names)){
  
  hist_list[[i]] <- read.csv(file = paste0(hist_loc,"/",hist_names[i]), stringsAsFactors = F)
  
}



## 1.2 prior predictions
pred_loc <-"NN_v2/Model_Data"
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
input_loc <- "NN_v2/Input Data"
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

filter_epoch = 9900

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

plots_lists <- list()

## 2.2 Predictions -----------------------

## Actual Figures
actual_lin_a <- input_list[[4]] %>% 
  select(Australia)
actual_lin_b <- input_list[[3]] %>% 
  select(Australia)
actual_lin <- actual_lin_a %>% 
  bind_rows(actual_lin_b)

actual_nd_a <- input_list[[6]] %>% 
  select(Australia)
actual_nd_b <- input_list[[5]] %>% 
  select(Australia)
actual_nd <- actual_nd_a %>% 
  bind_rows(actual_nd_b)

## prep coef df

actual_lin2 <- actual_lin %>% 
  mutate(max_coef = input_list[[1]][1,3]) %>% 
  mutate(min_coef = input_list[[1]][1,4]) 

actual_nd2 <- actual_nd %>% 
  mutate(mean_coef = input_list[[2]][1,3]) %>% 
  mutate(sd_coef = input_list[[2]][1,4]) 

old_min <- 0.1
old_max <- 1


pred_all_list <- list()
pred_all_list2 <- list()


for(i in 1:length(prior_names)){
  
  pred_all_list[[i]] <- prior_list[[i]] %>% 
    bind_rows(post_list[[i]]) %>% 
    mutate(year = 1997:2021)
  
  colnames(pred_all_list[[i]])[1] <- "nn_aus"
  
  if(i %% 2 == 1){
    
    pred_all_list[[i]] <- pred_all_list[[i]] %>% 
      bind_cols(actual_lin2) %>%  
      rename(aus = Australia) %>% 
      mutate(Australia = ((max_coef-min_coef)*((aus-old_min)/(old_max-old_min)))+min_coef) %>% 
      mutate('NN Australia' = ((max_coef-min_coef)*((nn_aus-old_min)/(old_max-old_min)))+min_coef) %>% 
      select(year, 'NN Australia', Australia)
    
    
  } else {
    
    pred_all_list[[i]] <- pred_all_list[[i]] %>% 
      bind_cols(actual_nd2)%>% 
      rename(aus = Australia) %>% 
      mutate(Australia = (aus*sd_coef)+mean_coef) %>% 
      mutate('NN Australia' = (nn_aus*sd_coef)+mean_coef) %>% 
      select(year, 'NN Australia', Australia)
    
    
  }
  
  pred_all_list2[[i]] <- reshape2::melt(pred_all_list[[i]], id = "year")
  pred_all_list[[i]] <- pred_all_list[[i]] %>% 
    mutate(Error = ((`NN Australia`-Australia)/Australia)*100) %>% 
    mutate(MSE = (Australia - `NN Australia`)^2) %>% 
    mutate(MAE = abs(Australia - `NN Australia`))
  
}


## 3.0 Plot  -------------------------
## 3.1 Histories ---------------------
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
    
    ggplot2::ggsave(filename = paste0("NN_v2/Plots_hist/100 Epoch/hist_plot_full_",hist_plot_names[k],"_",i,".png"),plot = plots_lists[[k]][[i]])  
    
  }
}



## 3.2 Predictions ---------------------

plot_list_pred <- list()

for(k in 1:length(prior_names)){
  
  plot_list_pred[[k]] <- ggplot(data = pred_all_list2[[k]],aes(x=year, y = value, group = variable))+
    labs(title = paste0(model_names[k]))+
    geom_line(aes(linetype=variable, color=variable))+
    geom_vline(xintercept = 2015, linetype = "dotdash",
               color = "#581845", linewidth =1.5)+
    theme(legend.position="top", bg = "white")+
    theme_economist_white(gray_bg = FALSE,base_family="ITC Officina Sans") +
    scale_y_continuous(labels = scales::label_number(scale_cut = scales::cut_short_scale()))+
    labs(y = "GDP (Constant US$ 2015)")+
    theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)))
  
  ggplot2::ggsave(filename = paste0("NN_v2/Plots_Pred/pred_plot_",model_names[k],".png"),plot = plot_list_pred[[k]])  
  
  
}

## error Calculator

error_df <- data.frame("model_name" = model_names[1:12],
                       "MSE" = rep(0),
                       "MAE" = rep(0),
                       "pred" = rep(0))

for(i in 1:nrow(error_df)){
  
  temp_df <- pred_all_list[[i]] %>% 
    filter(year < 2016)
  
  temp_df2 <-pred_all_list[[i]] %>% 
    filter(year == 2019)
  
  
  temp_MSE <- sum(temp_df[,"MSE"])/nrow(temp_df)
  temp_MAE <- sum(temp_df[,"MAE"])/nrow(temp_df)
  
  error_df[i,"MSE"] <- temp_MSE
  error_df[i,"MAE"] <- temp_MAE
  error_df[i,"pred"] <- temp_df2[,"NN Australia"]
  
  
}

error_df1 <- error_df %>% 
  arrange(MSE)

write.csv(x = error_df1, file = "NN_v2/Plots_Pred/error_totals.csv")

## care about Deep Model - MSE, ND
## Model 6

pred_all_list[[6]] %>% 
  filter(year ==2019)

write.csv(x = pred_all_list[[6]],"Predictions/NN Model 2 Predictions.csv")
