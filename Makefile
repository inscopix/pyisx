
# credentials
IDEAS_GITHUB_TOKEN_FILE=.ideas-github-token
repo=$(shell basename $(CURDIR))

.PHONY: test coverage-report jupyter verify-github-token

jupyter:
	@echo "Installing kernel  $(repo) in jupyter"
	-yes | jupyter kernelspec uninstall $(repo)
	poetry run python -m ipykernel install --user --name $(repo)


verify-github-token:
	@echo "Verifying GitHub token"
ifneq ($(shell test -f ${IDEAS_GITHUB_TOKEN_FILE} && echo yes),yes)
	$(error The GitHub token file ${IDEAS_GITHUB_TOKEN_FILE} does not exist)
endif

test: verify-github-token
	poetry run coverage run -m pytest -sx --failed-first

test-pip:
	@echo "Testing code installed on base env using pip..."
	pytest -s 

coverage-report: .coverage
	poetry run coverage html --omit="*/test*"
	open htmlcov/index.html

