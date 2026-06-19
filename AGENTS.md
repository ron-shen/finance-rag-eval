## Virtual Environment
- Use conda rag-eval-env

## TDD and test scope

Use TDD for core functions and meaningful business logic changes.

Required workflow for core logic changes:

1. Spawn a test_writer agent first to write or update a focused test.
2. Run the test and confirm it fails for the expected reason.
3. Spawn a coder agent to implement the smallest code change to make the test pass.
4. Rerun the test.
5. Refactor only after the test passes.
6. Rerun tests after refactoring.

Only write tests for meaningful behavior, such as:

- Pure functions
- Business rules
- Data transformations
- Validation logic
- Error handling
- Edge cases
- State transitions
- Algorithms

Do not add tests for:

- Trivial getters/setters
- Simple wrappers
- Thin UI glue with no logic
- Styling-only changes
- Config-only changes
- Re-export files
- Pass-through code

If no core behavior changes, do not add tests just to satisfy coverage. Explain why no test was needed.