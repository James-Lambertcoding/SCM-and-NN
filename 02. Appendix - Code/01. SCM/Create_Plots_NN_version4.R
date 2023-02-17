## History plots

## 0.0 packages ---------------

library(ggplot2)
library(ggthemes)
library(scales)
library(extrafont)
library(dplyr)
library(reshape2)
## 1.0 data ------------------

## 1.1 predictions --------------
model_name <- "long_deep"
model_title <- "Neural Network Model 4"
pred_name <- "Plots/NN/long_norm/predictions_long_norm.csv"
pred_df <- read.csv(file =pred_name, stringsAsFactors = F)

## bring the orginial datasets
data_prior_name <- "Plots/NN/long_norm/wb_select_prior_norm.csv"
data_post_name <- "Plots/NN/long_norm/wb_select_post_norm.csv"

data_prior_df <- read.csv(file = data_prior_name, stringsAsFactors = F)
data_post_df <- read.csv(file = data_post_name, stringsAsFactors = F)

## the data serires code

data_prior_df1 <- data_prior_df %>% 
  mutate(Series.Code = 0)
data_post_df1 <- data_post_df %>% 
  mutate(Series.Code = 0)


for(i in 1:nrow(data_prior_df1)){
  
  data_prior_df1[i,"Series.Code"] <- colnames(data_prior_df1)[41:71][as.vector(data_prior_df1[i,41:71] == 1)]
  
}

for(i in 1:nrow(data_post_df1)){
            
  data_post_df1[i,"Series.Code"] <- colnames(data_post_df1)[41:71][as.vector(data_post_df1[i,41:71] == 1)]
            
}
          
codes_df <- data_prior_df1 %>% 
  bind_rows(data_post_df1) %>% 
  select(Series.Code,year)

## add the codes to predictions
pred_df1 <- pred_df %>% 
bind_cols(codes_df)


codes_linksa <- new_df5a[,c("Series.Code","year")]
codes_linksb <- new_df5b[,c("Series.Code","year")]

codes_links <- codes_linksa %>% 
  bind_rows(codes_linksb)

nrow(codes_links)
nrow(pred_df)

coeff_df1<- coeff_df[,1:4]

pre_df2 <- pred_df %>% 
  bind_cols(codes_links) %>% 
  left_join(coeff_df1, by =c("Series.Code", "year")) %>% 
  rename(aus = Australia) %>% 
  rename(aus_nn = NN.Australia) %>% 
  mutate(Australia =  (aus*sd_coef)+mean_coef)%>% 
  mutate(NN =(aus_nn*sd_coef)+mean_coef)

pre_df3 <- pre_df2 %>% 
  select(Series.Code, year, Australia, NN) %>% 
  rename('NN Australia' = NN)

pre_df4 <- reshape2::melt(data = pre_df3, id =c("Series.Code","year"))

codes_plot <- unique(pre_df4[,"Series.Code"])
codes_df <- data.frame( "Series.Code" = codes_plot)

plot_pred_list <- list()

for(i in 1:length(codes_plot)){
  
  plot_pred_list[[i]] <- pre_df4 %>% 
    filter(Series.Code == codes_plot[i])
  plot_pred_list[[i]][,"year"] <- as.integer(plot_pred_list[[i]][,"year"] )
}


plots_pred <- list()

# ## create subtiltes
# sub_tiltes <- codes_df %>% 
#   left_join(codex[,c("Series.Code","Series.Name")], by = "Series.Code")
# sub_tiltes[31,2] <- "Automotive"

#write.csv(x = sub_tiltes, file = "Plots/NN/Long2/subtitle.csv")

sub_tiltes2<- read.csv(file = "Plots/NN/Long2/subtitle.csv", stringsAsFactors = F)

for (i in 1:length(codes_plot)) {
  
  plots_pred[[i]] <- ggplot(data = plot_pred_list[[i]],aes(x=year, y = value, group = variable))+
    labs(title = paste0(model_title), subtitle = sub_tiltes2[i,3])+
    geom_line(aes(linetype=variable, color=variable))+
    geom_vline(xintercept = 2015, linetype = "dotdash",
               color = "#581845", linewidth =1.5)+
    theme(legend.position="top", bg = "white")+
    theme_economist_white(gray_bg = FALSE,base_family="ITC Officina Sans") +
    scale_y_continuous(labels = scales::label_number(scale_cut = scales::cut_short_scale()))+
    labs(y = sub_tiltes2[i,4])+
    theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)))

  ggplot2::ggsave(filename = paste0("Plots/NN/long_norm/",model_name,codes_plot[i],".png"),plot = plots_pred[[i]])  
    
}

plots_pred[[1]]
