##########################################################################
# this is not original work.  It is based largely on work 
# done by Timothy DAuria and Herbert Julius Garonfolo
# I thank the both of them for their examples and acknowledge their work
##########################################################################

# set this to directory where your knn.csv resides
setwd("~/Desktop/KNN Project")


# Load Packages
library(tm)
library(class)
library(SnowballC)
library(caret)
library(kernlab)

# Read csv with tow column; text and category
df <- read.csv("knn.csv", sep = ";", header = TRUE)

# Create a Corpus
docs <- Corpus(VectorSource(df$Text))

# Clean corpus
docs <- tm_map(docs, tolower)
docs <- tm_map(docs, removeNumbers)
docs <- tm_map(docs, removeWords, stopwords("english"))
docs <- tm_map(docs, removePunctuation)
docs <- tm_map(docs, stripWhitespace)
docs <- tm_map(docs, stemDocument, language = "english")

# Document Term Matrix
dtm <- DocumentTermMatrix(docs)

# Transform dtm to matrix to data frame - df is easier to work with
mat.df <- as.data.frame(data.matrix(dtm), stringsAsFactors = FALSE)

# add known category
mat.df <- cbind(mat.df, df$Category)
# the added col will be at the end 
# so to change name --> 
colnames(mat.df)[ncol(mat.df)] <- "category"

# break setup into train and test
inTrain <- createDataPartition(y=mat.df$category, p = .5, list = FALSE)

# keep track of all the category
cl <- mat.df[,"category"]

# remove category from data to model with
modelData <- mat.df[,!colnames(mat.df) %in% "category"]

# unlike the train functions where you specify the method and only look 
# at the train data set.  The knn function looks at the train, test, and the train category
# not sure what k = 2 does, but my sense it is guess on how many categories to break for
# check out statQuest in youtube for great video about K nearest neighbor
knn.pred <- knn(modelData[inTrain,], modelData[-inTrain,], cl[inTrain], k=2)

# the knn.pred is the actual prediction of the category of the test data
# confusion matrix
conf.mat <- table("Predict" = knn.pred, "Actual" = cl[inTrain])

conf.mat
