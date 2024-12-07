### Author: Hongli Ma <hongli.ma.explore@gmail.com> 2023-12
### Usage: Please cite RNAsmol when you use this script

library(ggplot2)
library(tidyverse)
library(ggrepel)


data <- read.table(text = "
rna_perturbation mol_perturbation net_perturbation model dataset
0.8758 0.7876 0.5005 MGraphDTA_RNA PDB
0.9795 0.9333 0.6067 MGraphDTA_RNA ROBIN
0.8772 0.8928 0.5030 IIFDTI_RNA PDB
0.9238 0.8928 0.6380 IIFDTI_RNA ROBIN
0.6990 0.9032 0.5495 GraphDTA_RNA PDB
0.8820 0.9649 0.5662 GraphDTA_RNA ROBIN
0.8428 0.9903 0.7785 DrugBAN_RNA PDB
0.9999 0.9696 0.6825 DrugBAN_RNA ROBIN
0.9068 0.9241 0.8142 RNAsmol_noaug PDB
0.9915 0.9700 0.7250 RNAsmol_noaug ROBIN
0.9089 0.9043 0.6139 RNAsmol_rnaaug PDB
0.9959 0.9652 0.6539 RNAsmol_rnaaug ROBIN
0.9145 0.9920 0.8551 RNAsmol_molaug PDB
0.9937 0.9777 0.6735 RNAsmol_molaug ROBIN
0.9582 0.9939 0.6415 RNAsmol_bothaug PDB
0.9999 0.9800 0.6601 RNAsmol_bothaug ROBIN
", header = TRUE)


data_grouped <- data %>%
  group_by(dataset) %>%
  mutate(
    total = rna_perturbation + mol_perturbation + net_perturbation,
    proportion_rna = rna_perturbation / total,
    proportion_mol = mol_perturbation / total,
    proportion_net = net_perturbation / total
  ) %>%
  ungroup()


data_mean <- data_grouped %>%
  group_by(dataset) %>%
  summarise(
    mean_rna = mean(proportion_rna),
    mean_mol = mean(proportion_mol),
    mean_net = mean(proportion_net)
  ) %>%
  ungroup()



hex_colors1<-c("grey","darkgrey","#dbdbdb")
hex_colors2<-c('#A99DC4',"#8f8fbc","#cdcfe1")

hex_color<-c('#8098BB','#A99DC4','#9DC4A9')

alpha_values <- rep(0.8, length(colors))


data1 <- data.frame(
  category = c('Rhor','Rhom','Rhon'),
  value = c(0.3570158,0.3775104,0.2654738)
)
data2 <- data.frame(
  category = c('Rhor','Rhom','Rhon'),
  value = c(0.3713323,0.3765033,0.2521644)
)

df11 <- data.frame(value = c(35.7, 37.8, 26.5),
                 group = c('Rhor','Rhom','Rhon' ))

# Get the positions
df21 <- df11 %>% 
  mutate(csum = rev(cumsum(rev(value))), 
         pos = value/2 + lead(csum, 1),
         pos = if_else(is.na(pos), value/2, pos))

df12 <- data.frame(value = c(37.1, 37.7, 25.2),
                 group = c('Rhor','Rhom','Rhon'))

# Get the positions
df22 <- df12 %>% 
  mutate(csum = rev(cumsum(rev(value))), 
         pos = value/2 + lead(csum, 1),
         pos = if_else(is.na(pos), value/2, pos))

p1 <- ggplot(df11, aes(x = "", y = value, fill = fct_inorder(group))) +
  geom_bar(width = 1, stat = "identity", color = "black") +
  geom_label_repel(data = df21,
                   aes(y = pos, label = paste0(value, "%")),
                   size = 7.2, nudge_x = 0.8, show.legend = FALSE) +
  scale_fill_manual(values=alpha(unlist(hex_color), alpha_values))+
  coord_polar("y", start = 0) +
  theme_void()+   
  theme(
    legend.position = c(0.1,0.2),  
    legend.title = element_blank(),
    legend.text = element_text(size = 15) 
  )


ggsave('/path/to/pdb_roc_roh_pie_chart.png', dpi = 300)

