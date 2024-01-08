
# credentials
IDEAS_GITHUB_TOKEN_FILE=.ideas-github-token
repo=$(shell basename $(CURDIR))

.PHONY: test coverage-report jupyter 

jupyter:
	@echo "Installing kernel  $(repo) in jupyter"
	-yes | jupyter kernelspec uninstall $(repo)
	poetry run python -m ipykernel install --user --name $(repo)


install-poetry:
	@bash install-poetry.sh

test: 
	poetry run coverage run -m pytest -sx --failed-first

test-pip:
	@echo "Testing code installed on base env using pip..."
	pytest -s 


serve: install-poetry
	@echo "Serving docs locally..."
	poetry run mkdocs serve

setup.py: pyproject.toml
	poetry run poetry2setup > setup.py