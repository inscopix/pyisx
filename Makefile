


.PHONY: test coverage-report jupyter

jupyter:
	@echo "Installing kernel <py_isx> in jupyter"
	-yes | jupyter kernelspec uninstall py_isx
	poetry run python -m ipykernel install --user --name py_isx




test:
	poetry run coverage run -m pytest -sx --failed-first
	-rm coverage.svg
	poetry run coverage-badge -o coverage.svg

coverage-report: .coverage
	poetry run coverage html --omit="*/test*"
	open htmlcov/index.html

