library(tidyverse)
library(ggcorrplot)
library(readxl)
library(RColorBrewer)

# Define colors from the Spectral palette for use in the gradient
colors <- brewer.pal(n = 256, name = "Spectral")

# Manually set the midpoint color to black
midpoint_index <- length(colors) %/% 2 + 1  # Calculate the index of the midpoint color
colors[midpoint_index] <- "purple"  # Set the midpoint color to black

dat <- read_xlsx('~/R/Copy of AA_renamed.xlsx')

corr <- cor(dat)^2
order <- colnames(corr)

corr_circle <- corr
corr_circle[upper.tri(corr_circle, diag = T)] <- NA
corr_circle <- corr_circle %>% as.data.frame() %>% rownames_to_column('row') %>%
  pivot_longer(cols = !row, names_to = 'col', values_to = 'corr') %>%
  mutate(row = factor(row, levels = order), col = factor(col, levels = rev(order))) 

corr_names <- corr
corr_names[lower.tri(corr_names, diag = T)] <- NA
corr_names <- corr_names %>% as.data.frame() %>% rownames_to_column('row') %>%
  pivot_longer(cols = !row, names_to = 'col', values_to = 'corr') %>%
  mutate(row = factor(row, levels = rev(order)), col = factor(col, levels = rev(order)))

corr_label <- corr
corr_label[upper.tri(corr, diag = F)] <- NA
corr_label[lower.tri(corr_label, diag = F)] <- NA
corr_label <- corr_label %>% as.data.frame() %>% rownames_to_column('row') %>%
  pivot_longer(cols = !row, names_to = 'col', values_to = 'corr') %>%
  mutate(row = factor(row, levels = rev(order)), col = factor(col, levels = rev(order))) %>% drop_na()

p <- ggplot() +
  geom_tile(data = corr_circle, aes(x = row, y = col, fill = corr), color = 'black') +
  theme_minimal() +
  geom_text(data = corr_names, aes(x = row, y = col, label = round(corr,1), color = corr, fontface = "bold"), size = 4) +
  scale_fill_gradientn(colors = colors, na.value = "white",
                       guide = guide_colourbar(ticks = FALSE, title.position = "top", title.hjust = 0.5)) +  # Using gradientn for a custom array of colors
  scale_color_gradientn(colors = colors, na.value = "white",
                        guide = guide_colourbar(ticks = FALSE, title.position = "top", title.hjust = 0.5)) +  # Same here
  #scale_color_gradient2(low = "blue", mid = "black", high = "red", midpoint = 0.5, na.value = "white") + 
  #scale_fill_gradient2(low = "blue", mid = "black", high = "red", midpoint = 0.5, na.value = "white") +
  guides(fill = guide_colorbar(ticks = FALSE)) +
  geom_text(data = corr_label, aes(x = row, y = col, label = row), size = 4, color = 'black') +
  theme(axis.text = element_blank(), panel.grid = element_blank(), legend.position = "bottom") +
  labs(x='', y ='')

ggplot() +
  geom_point(data = corr_circle, shape = 22, aes(x = row, y = col, fill = corr, size = corr), color = 'black') +
  geom_text(data = corr_names, aes(x = row, y = col, label = round(corr,2), color = corr), size = 4) +
  geom_text(data = corr_label, aes(x = row, y = col, label = row), size = 4, color = 'black') +
  scale_color_gradient2(low = "blue", mid = "black", high = "red", midpoint = 0.5) + 
  scale_fill_gradient2(low = "blue", mid = "black", high = "red", midpoint = 0.5) + 
  theme_minimal() +
  theme(axis.text = element_blank(), panel.grid = element_blank(), legend.position = "bottom") +
  labs(x='', y ='') +
  geom_vline(xintercept = 0:31 + 0.5, color = 'black') +
  geom_hline(yintercept = 0:31 + 0.5, color = 'black')

#ggplot() +
#  geom_point(data = corr_circle, shape = 21, aes(x = row, y = col, fill = corr, size = corr), color = 'black') +
#  geom_text(data = corr_names, aes(x = row, y = col, label = round(corr,2), color = corr), size = 2) +
#  geom_text(data = corr_label, aes(x = row, y = col, label = row), size = 2, color = 'black') +
#  scale_color_gradient2() + 
#  scale_fill_gradient2() + 
#  theme_minimal() +
#  theme(axis.text = element_blank(), panel.grid = element_blank()) +
#  labs(x='', y ='') +
#  geom_vline(xintercept = 0:31 + 0.5, color = 'black') +
#  geom_hline(yintercept = 0:31 + 0.5, color = 'black')

