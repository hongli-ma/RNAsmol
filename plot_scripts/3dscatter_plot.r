### Author: Hongli Ma <hongli.ma.explore@gmail.com> 2024-01
### Usage: Please cite RNAsmol when you use this script


library(canvasXpress)

# Load the iris dataset
data <- read.table('/path/to/perturbation.txt',head=TRUE)

data3 <- subset(data, metric %in% c("ROCAUC", "PRAUC","ACC"))

# Define the annotations
valz1 <- c(rep("MGraphDTA_RNA", 6), rep("IIFDTI_RNA", 6), rep("GraphDTA_RNA", 6),rep("DrugBAN_RNA", 6),rep("RNAsmol",24))
smpz1 <- c("model")
valz2 <- c(rep(list("PDB", "PDB", "PDB", "ROBIN", "ROBIN", "ROBIN"),8))
smpz2 <- c("dataset")
datz1 <- as.data.frame(matrix(valz1, nrow = 48, ncol = 1, byrow = FALSE, dimnames = list(NULL, smpz1)))
datz2 <- as.data.frame(matrix(valz2, nrow = 48, ncol = 1, byrow = FALSE, dimnames = list(NULL, smpz2)))
rownames(datz1) <- rownames(data3)
rownames(datz2) <- rownames(data3)
colorKey1 = list("model"=list("MGraphDTA_RNA"="darksalmon","IIFDTI_RNA"="cadetblue2","GraphDTA_RNA"="darkseagreen","DrugBAN_RNA"="steelblue","RNAsmol"="red"))


# Create the 3D scatter plot
canvasXpress(
  data = data3[, 1:3],
  varAnnot = cbind(datz1,datz2),
  colorBy = "model",
  shapeBy ="dataset",
  graphType = "Scatter3D",
  theme = "CanvasXpress",
  title = "",
  colorScheme='Bootstrap',
  ellipseBy = "model",
  xAxis = list("rna_perturbation"),
  yAxis = list("mol_pertubation"),
  zAxis = list("net_perturbation"),
  colorKey = colorKey1,
  axisTickScaleFontFactor  = 0.5,
  axisTitleScaleFontFactor = 0.5,
  transparency=0.8
)