if ! command -v poetry &> /dev/null
then
    echo "poetry could not be found, installing..."
    curl -sSL https://install.python-poetry.org | python3 -
    echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bash_profile
    exit
fi