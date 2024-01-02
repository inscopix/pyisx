
# credentials
IDEAS_GITHUB_TOKEN_FILE=.ideas-github-token

.PHONY: test coverage-report jupyter verify-github-token

jupyter:
	@echo "Installing kernel <py_isx> in jupyter"
	-yes | jupyter kernelspec uninstall py_isx
	poetry run python -m ipykernel install --user --name py_isx


verify-github-token:
	@echo "Verifying GitHub token"
ifneq ($(shell test -f ${IDEAS_GITHUB_TOKEN_FILE} && echo yes),yes)
	$(error The GitHub token file ${IDEAS_GITHUB_TOKEN_FILE} does not exist)
endif

install-poetry:
	@bash install-poetry.sh

test: verify-github-token
	poetry run coverage run -m pytest -sx --failed-first

test-pip:
	@echo "Testing code installed on base env using pip..."
	pytest -s 

coverage-report: .coverage
	poetry run coverage html --omit="*/test*"
	open htmlcov/index.html

serve: install-poetry
	@echo "Serving docs locally..."
	poetry run mkdocs serve

