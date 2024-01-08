if ! command -v poetry &> /dev/null
then
    echo "poetry could not be found, installing..."
    curl -sSL https://install.python-poetry.org | python3 -
    echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bash_profile
    # tell git to use SSH to get code
    git config --global url.ssh://git@github.com/.insteadOf https://github.com/
    exit
fi