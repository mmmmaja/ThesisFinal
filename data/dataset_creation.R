library(dplyr)

"
In this file I create a new dataset using OMICS_older_toclasses 
matching to samples by RID

final_dataset:
RID | lcmm_classes_3 | lcmm_classes_3 | Samples...
"

dataset <- OMICS_data
# Add the RID column which is the rownumber
dataset$RID = rownames(dataset)
# Move the RID column to the first position  
dataset = dataset[, c(ncol(dataset), 1:(ncol(dataset)-1))]

# mapping: stable: 0, decliner: 1
dataset$TwoClass <- ifelse (dataset$TwoClass== "Stable", 0, 1)
print(dataset$TwoClass)

dataset$TwoClass <- as.factor(dataset$TwoClass)
print(dataset$TwoClass)

ThreeClassData <- OMICS_older_toclasses[, c("RID", "lcmm_classes3")]
print((ThreeClassData$lcmm_classes3))
# mapping: 1: 0, 1: 2, 3: 2
ThreeClassData$lcmm_classes3 <- ifelse (
  ThreeClassData$lcmm_classes3 == 1, 0, ifelse (
    ThreeClassData$lcmm_classes3 == 2, 1, 2))
print((ThreeClassData$lcmm_classes3))
# Merge
dataset <- left_join(dataset, ThreeClassData, by = "RID")
# Rename the newly added class column correctly
colnames(dataset)[which(colnames(dataset) == "lcmm_classes3")] <- "ThreeClass"
# Move the ThreeClass column to the second position
dataset = dataset[, c(1, ncol(dataset), 2:(ncol(dataset)-1))]

print("Dataset Done!")

# Save the final dataset
path = "C:/Users/mjgoj/Desktop/THESIS/data/final_dataset.RData"
save(dataset, file = path)

print("Dataset saved successfully!")

## --------------------------------------------------------------------------##
"
Splitting the dataset into different omics types
"

# Splitting dataset based on the omics type
source("C:/Users/mjgoj/Desktop/THESIS/R/utils.R")

# Each omics type inherits first3 columns from the dataset
proteomics_final = dataset[, c("RID", "TwoClass", "ThreeClass")]
metabolomics_final = dataset[, c("RID", "TwoClass", "ThreeClass")]
lipidomics_final = dataset[, c("RID", "TwoClass", "ThreeClass")]

for (i in 4 : ncol(dataset)) {
  feature = colnames(dataset)[i]
  data_type = find_omics_type(feature)
  if (is.null(data_type)) {
    print(paste("Feature", feature, "not found in any dataset"))
  }
  else {
    switch(data_type,
           proteomics = {
             proteomics_final <- cbind(proteomics_final, dataset[, i, drop = FALSE])
           },
           metabolomics = {
             metabolomics_final <- cbind(metabolomics_final, dataset[, i, drop = FALSE])
           },
           lipidomics = {
             lipidomics_final <- cbind(lipidomics_final, dataset[, i, drop = FALSE])
           }
    )
  }
}

# Metabolomics, remove categorical variables
categorical_cols = sapply(metabolomics_final, function(x) length(unique(x))) < 2
print("Categorical columns in metabolomics")
print(colnames(metabolomics_final)[categorical_cols])

# Remove the categorical columns
metabolomics_final = metabolomics_final[, !categorical_cols]
print(metabolomics_final$TAG_HIGH_PYRUVATE)

# Bar plot for each omics type
counts = c(length(colnames(proteomics_final)), length(colnames(metabolomics_final)), length(colnames(lipidomics_final)))
names = c("Proteomics", "Metabolomics", "Lipidomics")
df = data.frame(names, counts)

# Plot the bar plot
ggplot(df, aes(x = names, y = counts, fill = names)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  theme(legend.position = "none") +
  labs(title = "Data Distribution", x = "Omics Type", y = "Feature Count") +
  scale_fill_manual(values = colors) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# All good here

# Save in the CSV file to different sheets
library(openxlsx)
path = "C:/Users/mjgoj/Desktop/THESIS/data/final_dataset_split.xls"

workbook = createWorkbook()
addWorksheet(workbook, "Proteomics")
writeData(workbook, "Proteomics", proteomics_final)

addWorksheet(workbook, "Metabolomics")
writeData(workbook, "Metabolomics", metabolomics_final)

addWorksheet(workbook, "Lipidomics")
writeData(workbook, "Lipidomics", lipidomics_final)

saveWorkbook(workbook, path, overwrite = TRUE)

path = "C:/Users/mjgoj/Desktop/THESIS/data/final_dataset.RData"
# Save all the datasets
save(proteomics_final, metabolomics_final, lipidomics_final, dataset, file = path)

# Extract RIDs from training and testing datasets

RID_train = rownames(training)
text_RID_train = paste(RID_train, collapse = ", ")
RID_test = rownames(testing)
text_RID_test = paste(RID_test, collapse = ", ")


# Define the file path
path <- "C:/Users/mjgoj/Desktop/THESIS/data/RIDs.txt"

# Open the file for writing to overwrite existing content
file <- file(path, open = "w")

# Assuming text_RID_train and text_RID_test are vectors of RIDs
# Write initial content (this will erase existing content)
writeLines("training RIDs", con = file)
writeLines(text_RID_train, con = file) # Make sure text_RID_train is a character vector
writeLines("testing RIDs", con = file)
writeLines(text_RID_test, con = file) # Make sure text_RID_test is a character vector

# Close the file connection
close(file)

print("Initial RID's saved successfully!")

# --------------------------------------------------------------------------##
# At this point I need to verify what the fuck is going on

# Add the RID and ThreeClass to the train and test data
RID_train = rownames(training)
RID_test = rownames(testing)

# Handling the train set
# Extract the labels for the training dataset
OMICS_older_toclasses_train = OMICS_older_toclasses %>% filter(RID %in% RID_train)
ThreeClass_train = OMICS_older_toclasses_train$lcmm_classes3
training$ThreeClass = ThreeClass_train
training$RID = RID_train

# And now the test set
OMICS_older_toclasses_test = OMICS_older_toclasses %>% filter(RID %in% RID_test)
ThreeClass_test = OMICS_older_toclasses_test$lcmm_classes3
testing$ThreeClass = ThreeClass_test
testing$RID = RID_test

# SAVE as xls
library(openxlsx)
path = "C:/Users/mjgoj/Desktop/THESIS/data/wtf_train_test.xls"

workbook = createWorkbook()
addWorksheet(workbook, "Test")
writeData(workbook, "Test", testing)

addWorksheet(workbook, "Train")
writeData(workbook, "Train", training)

saveWorkbook(workbook, path, overwrite = TRUE)



######################

table <- table(
  OMICS_older_toclasses$lcmm_classes3,
  OMICS_older_toclasses$lcmm_classes2)

print(table)
print(unique(OMICS_older_toclasses$RID))
