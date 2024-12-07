### Author: Hongli Ma <hongli.ma.explore@gmail.com> 2023-12
### Usage: Please cite RNAsmol when you use this script


library(ggplot2)
library(ggridges)
library(viridis)
library(dplyr)
library(readr)

options(
  repr.plot.width = 12,  
  repr.plot.height = 6  
)


combined_data <- data.frame()


for (group in c("chbrbb", "coconut","proteinbinder")) {
  for (fold in 0:9) {
    file_path <- paste0(group, "_", fold, ".txt")
    data <- read.table(file_path, header=FALSE, col.names=c("Value"))
    data$Group <- group
    data$Fold <- fold
    combined_data <- bind_rows(combined_data, data)
  }
}


combined_data$Fold <- as.factor(combined_data$Fold)


ggplot(combined_data, aes(x = Value, y = Fold, fill = Group)) +
  geom_density_ridges(scale = 2.3, alpha = 0.9)+
  #scale_fill_viridis(discrete = TRUE, name = "Group", alpha = 0.3,option="E") +
  scale_fill_manual(values=c("#e9e9e9","#ddd7ee","#E8D5BC"), name = "Group") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold")) +
  theme(plot.subtitle = element_text(face = "bold", color = "grey")) +
  theme(plot.caption = element_text(color = "grey"),axis.title.x = element_blank(), axis.title.y = element_blank(),axis.text.x = element_blank(), axis.text.y = element_blank())

ggsave("/path/to/ridgeplot.png", width = 12, height = 6)
