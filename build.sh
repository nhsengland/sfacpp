#!/bin/bash
while getopts i: flag
do
    case "${flag}" in
        i) install="${OPTARG}";;
    esac
done
shouldInstall=false;
if [ "$install" = "true" ] || [ "$install" = "yes" ] || [ "$install" = "1" ] || [ "$install" = "t" ]; then
    shouldInstall=true;
fi
install=$(echo "$install" | awk '{print tolower($0)}');
echo "install: $install";
# handle the submodules
git submodule update
# install devtools, Rcpp, roxygen2 if necessary
Rscript -e "if(!require('devtools')) install.packages('devtools', repos='https://cran.r-project.org')";
Rscript -e "if(!require('Rcpp')) install.packages('Rcpp', repos='https://cran.r-project.org')";
Rscript -e "if(!require('roxygen2')) install.packages('roxygen2', repos='https://cran.r-project.org')";
# compile Rcpp attributes (from R)
Rscript -e "Rcpp::compileAttributes()";
# roxygen2
Rscript -e "roxygen2::roxygenize()";
# build the package or optionally install the package
if [ "$shouldInstall" = true ]; then
    Rscript -e "devtools::install()";
else
    Rscript -e "devtools::build()";
fi
# Rscript -e "devtools::build()";
