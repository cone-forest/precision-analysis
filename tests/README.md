# Tests for Precision Analysis Library

This directory contains comprehensive tests for the precision-analysis library.

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run specific test file
```bash
pytest tests/test_utils.py
```

### Run with coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

## Test Structure

- `test_utils.py` - Tests for utility functions (transformations, CSV loading, error calculations)
- `test_algorithms.py` - Tests for all calibration algorithms (tsai-lenz, park-martin, daniilidis, li-wang-wu, shah)
- `test_noise.py` - Tests for noise generation functions (Perlin and Gaussian noise)
- `test_functions_call.py` - Tests for the main API functions
- `conftest.py` - Shared pytest fixtures and configuration

## Test Coverage

The test suite covers:
- ✅ Transformation matrix operations (inversion, composition)
- ✅ Euler angle conversions
- ✅ CSV file loading and parsing
- ✅ Error calculation and statistics
- ✅ All five calibration algorithms
- ✅ Noise generation (Perlin and Gaussian)
- ✅ Main API functions

## Notes

- Some algorithms (notably Shah) may fail to converge with certain inputs. These cases are handled gracefully with skipped tests.
- Noise generation tests are skipped if the corresponding modules are not available.
- Tests use temporary files that are automatically cleaned up.

