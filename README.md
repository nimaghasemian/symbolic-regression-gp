# Symbolic Regression using Genetic Programming

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python implementation of genetic programming to discover mathematical expressions that fit observed data.

## 📌 Overview

This project implements:
- Genetic programming algorithm for symbolic regression
- Tree-based chromosome representation
- Genetic operations (crossover, mutation, selection)
- Fitness evaluation against target functions

Example application: Discovering `f(x) = -x³` from input-output data.

## 🚀 Features

- **Flexible function sets**: Supports arithmetic operators and transcendental functions
- **Multiple initialization methods**: Full and grow methods for tree generation
- **Selection mechanisms**: Tournament and roulette wheel selection
- **Visualization**: Matplotlib integration for comparing predicted vs actual functions
