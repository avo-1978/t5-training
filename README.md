# WSL (Windows Subsystem for Linux)

## Convert WSL 1 to WSL 2 if needed (for NVIDIA CUDA support)

* In PowerShell:
  * `wsl --set-default-version 2`
  * To list the installed distributions on your system:
    * `wsl -l -v`
  * If Debian is your distro:
    * `wsl --set-version Debian 2`

# Setup GitHub T5-Training Repository and Project

## Cerificates

* If certificates are needed be sure you have them in:
  * `/usr/local/share/ca-certificates`
* `sudo apt install --reinstall ca-certificates`
* `sudo update-ca-certificates`

## Compile and Install Python 3.8.9 from Source

* `sudo apt update`
* `sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev curl`
* `mkdir ~/python_3-8`
* `cd ~/python_3-8`
* `curl -O https://www.python.org/ftp/python/3.8.9/Python-3.8.9.tgz`
* `tar -xzf Python-3.8.9.tgz`
* `cd Python-3.8.9`
* `./configure --enable-optimizations --with-lto --enable-shared`
* `make -j$(nproc)`
* `sudo make altinstall`
* `sudo nano /etc/ld.so.conf.d/python3.8.conf`
  * Write the following in the file and save:
    * `/usr/local/lib`
* `sudo ldconfig`
* `python3.8 --version`

## Clone the Project with Git

* `sudo apt install git`
* `https://github.com/avo-1978/t5-training.git`

## Make a Virtual Environment with Python 3.8 Version

* Enter the directory where you have cloned the project
* `cd pure-python`
* `python3.8 -m venv venv`
* `source venv/bin/activate`

## Prerequisites

* Use `requirements.txt` in the pure-python directory:
  * `pip install -r requirements.txt`
* Note: if there are still certificates problems with pip not seeing them set the certificates path in the environment variable.
  * `export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt' 

or (if there is something wrong with using `requirements.txt`):

* With venv activated:
  * `pip install torch`
  * `pip install transformers numpy`
  * `pip install transformers[torch]`

## Usage

* Using the "offline" verions:
  * Unzip the models in the models/t5-small directory:
    * `sudo apt install p7zip-full`
    * `cd pure-python/models/t5-small`
    * `7z x t5-small.7z.001`
  * Run the "no-internet" scripts:
    * `python3.8 inference-t5_no-internet.py`
    * ...

* Use the "online" versions (downloads from huggingface site):
  * `cd pure-python`
  * `python3.8 inference-t5.py`
  * `python3.8 train-t5_valid-loss.py`
  * `python3.8 inference-t5-finetuned.py`
  * ...
