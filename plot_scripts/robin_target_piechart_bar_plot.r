### Author: Hongli Ma <hongli.ma.explore@gmail.com> 2024-01
### Usage: Please cite RNAsmol when you use this script


library(ggplot2)
library(ggrepel)
library(tidyverse)

# Define RGB colors
rgb_colors <- list(
  c(0.34832334141176474, 0.4657111465098039, 0.8883461629411764),
  c(0.6193179451882354, 0.7441207347647059, 0.9989309188196078),
  c(0.8674276350862745, 0.864376599772549, 0.8626024620196079),
  c(0.9684997476666667, 0.673977379772549, 0.5566492560470588),
  c(0.8393649370784314, 0.32185622094117644, 0.26492398098039216)
)

# Convert RGB to hexadecimal format
hex_colors <- lapply(rgb_colors, function(rgb) {
  rgb_hex <- rgb * 255  # Scale to [0, 255]
  rgb_hex <- round(rgb_hex)  # Round to nearest integer
  rgb_hex <- sprintf("#%02X%02X%02X", rgb_hex[1], rgb_hex[2], rgb_hex[3])  # Convert to hexadecimal
  return(rgb_hex)
})

alpha_values <- rep(0.5, length(colors))

# 数据
data1 <- data.frame(
  category = c('rG4','Hairpin','Three-way junction','Triple helix','Pseudoknot'),
  value = c(0.222,0.519,0.074,0.074,0.111)
)

df <- data.frame(value = c(22.2, 51.9, 7.4, 7.4, 11.1),
                 group = c('rG4', 'Hairpin', 'Pseudoknot','Three-way junction', 'Triple helix' ))

# Get the positions
df2 <- df %>% 
  mutate(csum = rev(cumsum(rev(value))), 
         pos = value/2 + lead(csum, 1),
         pos = if_else(is.na(pos), value/2, pos))


p1 <- ggplot(df, aes(x = "", y = value, fill = fct_inorder(group))) +
  geom_bar(width = 1, stat = "identity", color = "black") +
  geom_label_repel(data = df2,
                   aes(y = pos, label = paste0(value, "%")),
                   size = 4.5, nudge_x = 0.7, show.legend = FALSE) +
  scale_fill_manual(values=alpha(unlist(hex_colors), alpha_values))+
  coord_polar("y", start = 0) +
  theme_void() +  # 设置主题
  theme(
    legend.position = c(0,0.2),  
    legend.title = element_blank()
  )


######################################################################################################################


library(ggplot2)
library(tidyr)
library(RColorBrewer)

# Sample data
df <- data.frame(
  category = c('NRAS', 'TERRA', 'EWSR1', 'AKTIP','Zika_NS5','Zika3PrimeUTR'),
  segment1 =c(0.42, 0.49, 0.53, 0.71,0.47,0.52),
  segment2 = c(0.08, 0.05, 0.46, 0.2,0.23,0.3) 
)

# Reshape data to long format
df_long <- df %>%
  pivot_longer(cols = starts_with("segment"), names_to = "segment", values_to = "value")

colors <- c("#5876E266", "#5876E2CC")

# Plotting
ggplot(df_long, aes(x = category, y = value, fill = segment)) +
  geom_bar(stat = "identity",width=0.7,color="black",size=0.5,show.legend = FALSE) +
  scale_fill_manual(values = setNames(colors, unique(df_long$segment))) +
  labs(x = "Targets", y = "Hit Rate", fill = "Segment") +
  theme_minimal()+
  theme(
    panel.grid.major = element_blank(),  # Remove major grid lines
    panel.grid.minor = element_blank(),   # Remove minor grid lines
    axis.text.x = element_blank(),  # Remove x-axis labels
    axis.text.y = element_blank(),  # Remove y-axis labels
    axis.title.x = element_blank(),  # Remove x-axis title
    axis.title.y = element_blank()   # Remove y-axis title
  )+theme(aspect.ratio = 2/1)


######################################################################################################################


library(ggplot2)
library(tidyr)
library(RColorBrewer)

# Sample data
df <- data.frame(
  category = c('PreQ1', 'SAM_II', 'ZTP'),
  segment1 =c(0.43, 0.6, 0.52),
  segment2 = c(0.25, 0.35, 0.39) 
)

# Reshape data to long format
df_long <- df %>%
  pivot_longer(cols = starts_with("segment"), names_to = "segment", values_to = "value")

colors <- c("#DDDCDB66", "grey")

# Plotting
ggplot(df_long, aes(x = category, y = value, fill = segment)) +
  geom_bar(stat = "identity",width=0.7,color="black",size=0.5,show.legend = FALSE) +
  scale_fill_manual(values = setNames(colors, unique(df_long$segment))) +
  labs(x = "Targets", y = "Hit Rate", fill = "Segment") +
  theme_minimal()+
  theme(
    panel.grid.major = element_blank(),  # Remove major grid lines
    panel.grid.minor = element_blank(),   # Remove minor grid lines
    axis.text.x = element_blank(),  # Remove x-axis labels
    axis.text.y = element_blank(),  # Remove y-axis labels
    axis.title.x = element_blank(),  # Remove x-axis title
    axis.title.y = element_blank()   # Remove y-axis title
  )+theme(aspect.ratio = 3.3/1)


######################################################################################################################


library(ggplot2)
library(tidyr)
library(RColorBrewer)

# Sample data
df <- data.frame(
  category = c('TPP', 'Glutamine_RS'),
  segment1 =c(0.55, 0.56),
  segment2 = c(0.11, 0.35) 
)

# Reshape data to long format
df_long <- df %>%
  pivot_longer(cols = starts_with("segment"), names_to = "segment", values_to = "value")

colors <- c("#F6AB8D99", "#F6AB8DFF")

# Plotting
ggplot(df_long, aes(x = category, y = value, fill = segment)) +
  geom_bar(stat = "identity",width=0.7,color="black",size=0.5,show.legend = FALSE) +
  scale_fill_manual(values = setNames(colors, unique(df_long$segment))) +
  labs(x = "Targets", y = "Hit Rate", fill = "Segment") +
  theme_minimal()+
  theme(
    panel.grid.major = element_blank(),  # Remove major grid lines
    panel.grid.minor = element_blank(),   # Remove minor grid lines
    axis.text.x = element_blank(),  # Remove x-axis labels
    axis.text.y = element_blank(),  # Remove y-axis labels
    axis.title.x = element_blank(),  # Remove x-axis title
    axis.title.y = element_blank()   # Remove y-axis title
  )+theme(aspect.ratio = 4/1)
