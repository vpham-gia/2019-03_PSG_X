echo "Starting project initialization. This sould only be run once per machine!"
echo "Creating a local python environment in .venv and activating it"

sudo apt-get update
sudo apt-get install build-essential checkinstall wget
sudo apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev
cd /usr/src
sudo wget https://www.python.org/ftp/python/3.5.6/Python-3.5.6.tgz
sudo tar xzf Python-3.5.6.tgz
cd Python-3.5.6
sudo ./configure --enable-optimizations
sudo make install

# sudo python3.5 -m pip install --upgrade pip
# pip install --user pipenv
# echo "PATH=$HOME/.local/bin:$PATH" >> ~/.bashrc
# source ~/.bashrc
# pipenv install
# ----------------------
# # pip install pyenv
#
# pyenv local 3.5.6
# python -m venv .venv
# source .venv/bin/activate
# pip install --upgrade pip
# pip install pipenv
# pipenv install -r requirements.txt
