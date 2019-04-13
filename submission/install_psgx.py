import os

os.system('python3.5 -m pip install --upgrade pip')
os.system('sudo python3.5 -m pip install --upgrade pip')

os.system('pip3 install --user pipenv')
os.system('echo "PATH=$HOME/.local/bin:$PATH" >> ~/.bashrc')

os.system('$HOME/.local/bin/pipenv install')
os.system('$HOME/.local/bin/pipenv shell')
