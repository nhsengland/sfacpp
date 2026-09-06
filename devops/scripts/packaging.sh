#!/bin/bash
# -e: environment
# -k: key vault root
# -s: service principal root (to which env is append)
# -t: token name in the keyvault
# -d: databricks workspace url
# -p: path to upload to on databricks filesystem
# -i: build requested for ID
# -f: whether to upload to filesystem
# -a: whether to upload to artifacts feed
# -c: config pfile for pypirc from twine
# -v: build number

while getopts m:i:r:c:u: flag
do
    case "${flag}" in
        m) is_main=${OPTARG};;
        i) bld_id="${OPTARG}";;
        r) rls_nm="${OPTARG}";;
        c) twine_conf="${OPTARG}";;
        u) run_unit_test=${OPTARG};;
    esac
done
# don't need compiler if only building the sdist for dev versions
if [[ $is_main == 1 ]]; then
    # install CMake, Clang C++ compiler
    sudo apt-get install -y cmake 
    # fortran (for compiling OpenBlas)
    sudo apt-get install -y gfortran
    sudo apt-get install -y build-essential
    # boost
    # sudo apt-get install -y libboost-all-dev git
    # llvm compiler
    sudo apt-get install -y lsb-release wget software-properties-common gnupg 
    # libopenblas-dev
    wget https://apt.llvm.org/llvm.sh
    chmod +x llvm.sh
    sudo ./llvm.sh 16
    sudo apt-get install -y libclang-16-dev clang-tools-16 libomp-16-dev llvm-16-dev lld-16
fi
# gcc compiler
# sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
# sudo apt-get install -y g++-13
# set environment variables for CMAKE_ARGS, used in the build process
# export CMAKE_ARGS="-DCMAKE_CXX_COMPILER=/usr/bin/g++-13"
# if set to run unit tests, do so
if [[ $run_unit_test == 1 ]]; then
    (cd cpp/tests && cmake -S . -B build && cmake --build build && cd build && ctest --output-on-failure)
fi
# install build and setuptools
sudo apt-get --yes install python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
# install dependencies into virtual environment
python3 -m pip install --upgrade pip build setuptools twine wheel pandas numpy patsy
# upload to artifact feed
if [[ "$twine_conf" == "" ]]; then
    echo "must specific -c flag"
    exit 1;
fi
if [[ "$bld_id" == "" ]]; then
    echo "must specific build id in -i flag"
    exit 1;
fi
echo "uploading to artifacts feed..."
if [[ $is_main == 1 ]]; then
    python3 setup.py egg_info sdist
    python3 setup.py egg_info bdist_wheel
else
    shrt_id=$(echo $bld_id | tr -d '.');
    # dev_pkg_ver=$(ls dist/*.whl | awk -F '-' '{printf $(NF-3)}');
    dev_bld_id="dev${shrt_id}";
    echo "Development Build ID is $dev_bld_id";
    # python3 setup.py egg_info --tag-build="${dev_bld_id}" bdist_wheel
    python3 setup.py egg_info --tag-build="${dev_bld_id}" sdist
fi
# check whether folder exists
if [ ! -d "dist" ]; then
    echo "could not find dist directory";
    exit 1;
fi;
pkg_ver=$(ls dist/*.whl | awk -F '-' '{printf $(NF-3)}');
pkg_install="sfacpp==$pkg_ver"
python3 -m twine upload -r "strategic-analysis-feed" --config-file $twine_conf dist/*
