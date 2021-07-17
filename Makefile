all:
	poetry run python pd_dataframe.py

install:
	poetry install

clean:
	rm -rf poetry.lock *.pyc

