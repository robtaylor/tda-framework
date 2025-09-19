# Contributing to TDA Framework

Thank you for your interest in contributing to the TDA Framework! This document provides guidelines and information for contributors.

## Getting Started

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/robtaylor/tda-framework.git
   cd tda-framework
   ```

2. **Install PDM** (if you don't have it):
   ```bash
   pip install pdm
   ```

3. **Install development dependencies**:
   ```bash
   pdm install -G dev
   ```

4. **Run tests to ensure everything works**:
   ```bash
   pdm run test
   ```

### Development Workflow

1. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards below

3. **Run the development tools**:
   ```bash
   pdm run lint        # Check code style
   pdm run typecheck   # Check types
   pdm run test        # Run tests
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and create a pull request**:
   ```bash
   git push origin feature/your-feature-name
   ```

## Coding Standards

### Code Style
- Follow PEP 8 style guidelines
- Use `ruff` for linting: `pdm run lint`
- Use `black` for code formatting: `pdm run format`
- Use type hints for all function parameters and return values

### Documentation
- Add docstrings to all public functions, classes, and modules
- Use Google-style docstrings
- Include examples in docstrings where helpful

### Testing
- Write tests for all new functionality
- Maintain or improve test coverage
- Use descriptive test names that explain what is being tested
- Follow the existing test structure in `tests/`

### Commit Messages
Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions/changes
- `refactor:` for code refactoring
- `perf:` for performance improvements

## Types of Contributions

### New Components
When adding new analog computation components:

1. **Inherit from appropriate base class**:
   - `Component` for basic components
   - `IntegratingComponent` for integrators
   - `NonlinearComponent` for nonlinear functions

2. **Implement required methods**:
   - `update()` - core computation logic
   - Add appropriate ports in `__init__()`

3. **Add comprehensive tests**:
   - Test component creation
   - Test simulation behavior
   - Test edge cases

4. **Example**:
   ```python
   class NewComponent(Component):
       def __init__(self, name: str, parameter: float):
           super().__init__(name)
           self.parameter = parameter
           self.add_port("input", is_input=True)
           self.add_port("output", is_input=False)

       def update(self) -> None:
           input_value = self.get_input_value("input")
           output_value = input_value * self.parameter
           self.set_output_value("output", output_value)
   ```

### Hardware Models
When adding new hardware models:

1. **Inherit from `HardwareModel`**
2. **Implement required methods**:
   - `get_propagation_delay()`
   - `get_power_consumption()`
   - `get_accuracy()`

3. **Add to `HardwareLibrary`**
4. **Include realistic specifications** from datasheets

### Examples
When adding example implementations:

1. **Create in `src/tda_framework/examples/`**
2. **Include comprehensive docstrings**
3. **Show practical use cases**
4. **Add performance analysis**
5. **Include in main documentation**

### Analysis Tools
When adding analysis capabilities:

1. **Place in `src/tda_framework/analysis/`**
2. **Follow existing patterns**
3. **Include visualization if applicable**
4. **Add comprehensive tests**

## Code Review Process

1. **All contributions** must go through pull request review
2. **At least one maintainer** must approve changes
3. **CI must pass** (tests, linting, type checking)
4. **Documentation** must be updated for new features

## Performance Considerations

- **Efficiency**: TDA Framework handles large simulations, so performance matters
- **Memory usage**: Be mindful of memory consumption in long simulations
- **Event scheduling**: Minimize event overhead where possible
- **Profiling**: Use performance analysis tools when optimizing

## Debugging Tips

### Common Issues
1. **Component not updating**: Ensure proper signal connections
2. **Timing issues**: Check event scheduling and propagation delays
3. **Memory leaks**: Clear signal histories for long simulations

### Debugging Tools
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check simulation statistics
stats = simulator.get_statistics()
print(f"Events executed: {stats['events_executed']}")

# Trace signals
tracer = SignalTracer()
tracer.trace_signal(your_signal)
```

## Documentation

### API Documentation
- All public APIs must have comprehensive docstrings
- Include parameter descriptions and types
- Provide usage examples
- Document exceptions that may be raised

### README Updates
- Update feature lists for new capabilities
- Add examples for significant new functionality
- Keep installation and usage instructions current

### Changelog
- Document all changes in semantic versioning format
- Include migration notes for breaking changes

## Release Process

1. **Version bumping** follows semantic versioning (semver)
2. **Update changelog** with all changes since last release
3. **Tag release** with appropriate version
4. **Build and publish** to PyPI (maintainers only)

## Questions?

- **Open an issue** for bugs or feature requests
- **Discussion section** for general questions
- **Email maintainers** for sensitive issues

## License

By contributing to TDA Framework, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to TDA Framework! 🚀