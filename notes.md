

## Local Build

Use compiler flag `-DLOCAL_TEST_BUILD`.

### Install armadillo

- Download appropriate version of armadillo from (here)[https://]
- Decompress file, and cd into directory using `Terminal`.
- Run `cmake -DALLOW_OPENBLAS_MACOS=ON .`
- Run `sudo make install`
- Ensure use compiler flag `-larmadillo`

### Install spdlog

Run the following commands in Terminal

- `git clone https://github.com/gabime/spdlog.git`
- `cd spdlog && mkdir build && cd build`
- `cmake .. && cmake --build .`
- `sudo make install`
  
### Install boost c++ libraries

Assuming that Homebrew is installed (macOS)

- `brew update`
- `brew upgrade`
- `brew install boost`

### Install Apache Arrow

```sh
git clone https://github.com/apache/arrow.git
cd arrow
brew update && brew bundle --file=cpp/Brewfile
cd arrow/cpp
mkdir release
cd release
cmake .. -DARROW_CSV=ON -DARROW_PARQUET=ON -DARROW_COMPUTE=ON -DARROW_WITH_ZSTD=ON -DARROW_WITH_ZLIB=ON -DARROW_WITH_SNAPPY=ON -DARROW_WITH_LZ4=ON -DARROW_WITH_BZ2=ON -DARROW_DATASET=ON
make
sudo make install
```


```
# 1. Go to your venv bin directory
cd sfacpp/.venv/bin/

# 2. Find where the symlink points (likely /opt/homebrew/...)
ls -l python
# Example output: python -> /opt/homebrew/bin/python3.11

# 3. Copy the REAL binary (the target of the link) to a new file named 'python_debug'
# REPLACE THE PATH BELOW with the actual path you found in step 2
cp /opt/homebrew/Cellar/python@3.11/3.11.14_1/Frameworks/Python.framework/Versions/3.11/bin/python3.11 ./python_debug

# 4. Remove any lingering signature (optional, but ensures it's treated as local)
codesign --remove-signature ./python_debug
```