library(jpeg)
setwd("~/kaggle-galaxy/")
trn_dir <- "Data/images_train/"
tst_dir <- "Data/images_test/"

# Read in data
readJPEGasR <- function(dir) {
    f_in <- paste0(dir, list.files(dir))

    x <- array(dim = c(424, 424, length(f_in)))
    for (i in 1:length(f_in)) {
        x <- readJPEG(f_in[i], native = FALSE)[,,1]
    }

    return(x)
}
