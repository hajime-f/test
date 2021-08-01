all:
	poetry run python sqlite3_test.py

install:
	poetry install

clean:
	rm -rf poetry.lock *.pyc

