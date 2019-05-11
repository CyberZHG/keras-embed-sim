#!/usr/bin/env bash
pycodestyle --max-line-length=120 keras_embed_sim tests && \
    nosetests --with-coverage --cover-html --cover-html-dir=htmlcov --cover-package=keras_embed_sim tests
