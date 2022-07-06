.PHONY: all python_requirements genomics_tools work

# ---------- GLOBALS ---------- #
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
SHELL := /bin/bash


all : python_requirements genomics_tools

python_requirements: requirements.txt
	virtualenv env \
	&& source ./env/bin/activate \
	&& pip install -r requirements.txt

genomics_tools: samtools bedtools

# Installing samtools
samtools :
	curl -L https://github.com/samtools/samtools/releases/download/1.11/samtools-1.11.tar.bz2 \
		| tar -xj \
		&& cd samtools-1.11 \
		&& ./configure --prefix=$(PROJECT_DIR) --without-curses \
		&& make \
		&& make install \
		&& cd $(PROJECT_DIR) \
		&& ln -s $(PROJECT_DIR)/samtools-1.11/samtools $(PROJECT_DIR)/libs/samtools

# Installing bedtools
bedtools :
	curl -L https://github.com/arq5x/bedtools2/releases/download/v2.29.1/bedtools-2.29.1.tar.gz \
		| tar zxv \
		&& cd bedtools2 \
		&& make \
		&& cd $(PROJECT_DIR) \
		&& ln -s $(PROJECT_DIR)/bedtools2/bin/bedtools $(PROJECT_DIR)/libs/bedtools