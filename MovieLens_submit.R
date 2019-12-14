
# Function to install and load the library if it doesn't exist in the local repository
pkgTest <- function(x)
{
  if (!require(x,character.only = TRUE))
  {
    install.packages(x,dep=TRUE)
    if(!require(x,character.only = TRUE)) stop("Package not found")
  }
}

pkgTest("tidyverse")
pkgTest("lubridate")
pkgTest("stringr")
pkgTest("rvest")
pkgTest("XML")
pkgTest("tidytext")
pkgTest("wordcloud")
pkgTest("caret")
pkgTest("data.table")
pkgTest("kableExtra")
pkgTest("Matrix.utils")
pkgTest("DT")
pkgTest("recosystem")
pkgTest("rmarkdown")
pkgTest("tinytex")

################################################################################
# Aquiring MovieLen dataset
################################################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings          <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                    col.names = c("userId", "movieId", "rating", "timestamp"))

movies           <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies           <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens        <- left_join(ratings, movies, by = "movieId")
movielens$timestamp <- as_datetime(movielens$timestamp)

################################################################################
# Creating training (edx) and testing (temp) dataset
################################################################################

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx  <- movielens[-test_index,]
temp <- movielens[test_index,]
  
################################################################################
# Creating Validation dataset and revised Training dataset (edx)
################################################################################

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Removing validation dataset from training dataset
edx         <- anti_join(edx, validation)

################################################################################
# Model 1 (Only Considering movieId)
################################################################################

# Calculating average rating for entire training dataset
FinalResult01  <- validation
meanRating_edx <- mean(edx$rating)
FinalResult01$meanRating_edx <- mean(edx$rating)

# Calculating movidId bias on the training set and conduct the prediction
meanMovieBias_edx       <- edx %>% group_by(movieId) %>% 
  summarize(biasMovie01 = sum(rating - meanRating_edx)/(n()))
FinalResult01    <- merge(FinalResult01, meanMovieBias_edx, by="movieId", all.x=TRUE)
FinalResult01$PredRating01 <- FinalResult01$meanRating_edx + FinalResult01$biasMovie01 
rmse_01                 <- RMSE(FinalResult01$rating,FinalResult01$PredRating01)  

################################################################################
# Model 2 (Only Considering movieId and userId)
################################################################################

# Calculating average rating for entire training dataset
FinalResult02  <- validation
meanRating_edx <- mean(edx$rating)
FinalResult02$meanRating_edx <- mean(edx$rating)

# Calculating movidId & userId biases on the training set and conduct the prediction
edx2 <- merge(edx, meanMovieBias_edx, by="movieId", all.x=TRUE)
meanUserBias_edx       <- edx2 %>% group_by(userId) %>% 
  summarize(biasUser02 = sum(rating - meanRating_edx - biasMovie01)/(n()))
FinalResult02    <- merge(FinalResult01, meanUserBias_edx, by="userId", all.x=TRUE)
FinalResult02$PredRating02 <- FinalResult02$meanRating_edx + FinalResult02$biasMovie01 + FinalResult02$biasUser02 
rmse_02                 <- RMSE(FinalResult02$rating,FinalResult02$PredRating02)  

################################################################################
# Model 3 (Only Considering movieId and userId but with regularization)
################################################################################

FinalResult03  <- validation

lambdas  <- seq(0, 10, 1)
for (ilambdas in lambdas){
  meanMovieBiasRegularized_edx       <- edx %>% group_by(movieId) %>% 
    summarize(biasMovieRegularized03 = sum(rating - meanRating_edx)/(n()+ilambdas)) 
  FinalResult03a <- merge(FinalResult03, meanMovieBiasRegularized_edx, by="movieId", all.x=TRUE)

  edx2 <- merge(edx, meanMovieBiasRegularized_edx, by="movieId", all.x=TRUE)
  meanUserBiasRegularized_edx       <- edx2 %>% group_by(userId) %>% 
    summarize(biasUserRegularized03 = sum(rating - meanRating_edx - biasMovieRegularized03)/(n()+ilambdas))
  FinalResult03b <- merge(FinalResult03a, meanUserBiasRegularized_edx, by="userId", all.x=TRUE)

  FinalResult03b$PredRating03 <- rep(meanRating_edx, nrow(FinalResult03a)) + FinalResult03b$biasMovieRegularized03 + FinalResult03b$biasUserRegularized03 
  rmse                 <- RMSE(FinalResult03b$rating, FinalResult03b$PredRating03)

  print(paste0("Trying lambda = ", ilambdas, " and RMSE = ", rmse))
  
  if (ilambdas == 0){
    lambdasChoosen = ilambdas
    rmseChoosen = rmse
  } else {
    if (rmseChoosen > rmse){
      lambdasChoosen = ilambdas
      rmseChoosen = rmse
    }
  }
}
print(paste("Lambda that produces the lowest RMSE : ", lambdasChoosen, sep=""))
print(paste("RMSE : ", rmseChoosen, sep=""))

# Rerun the analysis using the lowert RMSE lambda
meanMovieBiasRegularized_edx       <- edx %>% group_by(movieId) %>% 
  summarize(biasMovieRegularized03 = sum(rating - meanRating_edx)/(n()+lambdasChoosen)) 
FinalResult03a <- merge(FinalResult03, meanMovieBiasRegularized_edx, by="movieId", all.x=TRUE)

edx2 <- merge(edx, meanMovieBiasRegularized_edx, by="movieId", all.x=TRUE)
meanUserBiasRegularized_edx       <- edx2 %>% group_by(userId) %>% 
  summarize(biasUserRegularized03 = sum(rating - meanRating_edx - biasMovieRegularized03)/(n()+lambdasChoosen))
FinalResult03b <- merge(FinalResult03a, meanUserBiasRegularized_edx, by="userId", all.x=TRUE)

FinalResult03b$PredRating03 <- meanRating_edx + FinalResult03b$biasMovieRegularized03 + FinalResult03b$biasUserRegularized03 
rmse_03 = rmseChoosen
  
################################################################################
# Displaying all three models and the choosen model
################################################################################

results <- data.frame(methods=c("Movie Model","Movie & User Model", "Regularized Movie & User Model"),rmse = c(rmse_01, rmse_02, rmse_03))
ChoosenModel <- results[results$rmse == min(results$rmse),]
print(paste("Model choose is : ", ChoosenModel$methods, " and RMSE = ", ChoosenModel$rmse, sep=""))









