.PHONY: thesis.pdf all clean

all: thesis.pdf

thesis.pdf: thesis.tex
	latexmk -pdf -lualatex -use-make thesis.tex

clean:
	latexmk -C thesis.tex

watch:
	while true; do make thesis.pdf; inotifywait -qre close_write .; done