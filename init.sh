echo "Starting project initialization for Ubuntu."

echo "Updating apt-get and installing packages .."
sudo apt-get update
sudo apt-get install build-essential checkinstall wget
sudo apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev
cd /usr/src

echo "Installing python 3.5.6 .."
sudo wget https://www.python.org/ftp/python/3.5.6/Python-3.5.6.tgz
sudo tar xzf Python-3.5.6.tgz
cd Python-3.5.6
sudo ./configure --enable-optimizations
sudo make install

echo "Done - environment is ready"

# echo "Installing pipenv .."
# sudo python3.5 -m pip install --upgrade pip
# pip install --user pipenv
# echo "PATH=$HOME/.local/bin:$PATH" >> ~/.bashrc
# source ~/.bashrc
# pipenv install
