# Project Architecture

## Overview

This project implements a modern ML pipeline for electromagnetic multipole analysis with clean separation of concerns, high performance, and comprehensive testing.

## Directory Structure

```
src/                           # Core implementation
├── core/                     # Foundational components
│   ├── config.py            # Unified configuration system
│   ├── dependencies.py      # Dependency management
│   ├── data_generator.py    # Unified data generation
│   ├── library_manager.py   # High-performance library loading
│   └── mpfield.py          # Multipole field computation
├── pipeline/               # Scientific pipeline
│   └── decomposition.py   # Optimized decomposition engine
├── models/                # ML model implementations
│   ├── base.py           # Abstract base model
│   ├── mlp.py           # MLP implementation
│   ├── baseline.py      # Ridge/linear baselines
│   └── registry.py      # Model factory system
├── api/                 # Inference and serving
│   ├── preprocessing.py # Data preprocessing pipeline
│   └── inference.py    # Model inference API
└── cli/               # Command-line interfaces
    └── ...           # Modern CLI implementations

tests/                # Comprehensive test suite
├── unit/           # Unit tests for components
├── integration/   # Integration tests
└── fixtures/     # Test data and utilities

data/              # Data management
├── raw/          # Raw input data
├── processed/   # Processed data
└── ml/         # ML datasets and features

models/           # Model artifacts and tracking
├── artifacts/   # Trained model checkpoints
└── tracking/   # Experiment metadata

Chersie/         # Library data (preserved)
├── Fields0.5/      # Slow library data
└── FieldsFast0.5/  # Fast library data

docs/           # Documentation
```

## Core Components

### 1. Configuration System (`src/core/config.py`)
- **Unified hierarchical configuration** with environment variable support
- **Type-safe configuration classes** using dataclasses
- **Specialized configs** for different components (ML, Pipeline, etc.)

### 2. Library Manager (`src/core/library_manager.py`)  
- **High-performance batch loading** (replaces 510+ individual file reads)
- **Memory-mapped caching** for fast repeated access
- **Thread-safe operation** with automatic loading

### 3. Data Generator (`src/core/data_generator.py`)
- **Unified Latin-square generation** (replaces inconsistent implementations)
- **Supports both fixed (pipeline) and random (ML) modes**
- **Consistent field computation** with validation

### 4. Model Registry (`src/models/registry.py`)
- **Factory pattern** for model creation
- **Pluggable architectures** with consistent interfaces
- **Automatic model discovery** and registration

### 5. Inference API (`src/api/inference.py`)
- **Production-ready serving** with batch processing
- **Preprocessing pipeline** integration
- **Memory-efficient operation** with validation

## Key Improvements

### Performance Optimizations
- **510x reduction** in library file reads (batch loading vs per-mode)
- **Memory-mapped data access** for large libraries
- **Batched inference processing** with configurable batch sizes
- **Efficient preprocessing pipelines** with caching

### Code Quality  
- **Comprehensive type hints** throughout codebase
- **100+ unit tests** covering all major components
- **Consistent error handling** with informative messages
- **Modular design** with clear separation of concerns

### Usability
- **Simple factory functions** for common use cases
- **Environment variable configuration** for deployment
- **Comprehensive logging** with configurable levels
- **Clear migration path** from legacy code

## Design Principles

### 1. Configuration-Driven Design
All components accept configuration objects with sensible defaults and environment variable overrides.

### 2. Dependency Injection  
Components accept their dependencies as constructor arguments, enabling easy testing and flexibility.

### 3. Fail-Fast Validation
Input validation occurs early with clear error messages to catch issues immediately.

### 4. Performance by Default
Efficient implementations are the default, with options for further optimization.

### 5. Backwards Compatibility
Legacy interfaces are preserved during transition while new APIs provide enhanced functionality.
