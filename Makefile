# Development targets

TEST_FLAGS = --workspace --lib --all-targets --no-default-features --features

test-cpu:
	cargo check $(TEST_FLAGS) "cpu,debug" --examples
	cargo test $(TEST_FLAGS) "cpu,debug"

test-wgpu:
	cargo check $(TEST_FLAGS) "wgpu,debug" --examples
	cargo test $(TEST_FLAGS) "wgpu,debug"

test-cuda:
	cargo check $(TEST_FLAGS) "cuda,debug" --examples
	cargo test $(TEST_FLAGS) "cuda,debug"

example-cpu:
	cargo run --example basic_operations --no-default-features --features "cpu,debug"

example-wgpu:
	cargo run --example basic_operations --no-default-features --features "wgpu,debug"

example-cuda:
	cargo run --example basic_operations --no-default-features --features "cuda,debug"

test: test-cpu test-wgpu test-cuda

check-examples:
	cargo check --examples --no-default-features --features "cpu,debug"
	cargo check --examples --no-default-features --features "wgpu,debug"
	cargo check --examples --no-default-features --features "cuda,debug"

# Code quality
fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

clippy:
	cargo clippy --all-features -- -D warnings

# Documentation targets  
docs:
	cargo doc --all-features --no-deps --open

docs-book:
	cd docs && mdbook build

docs-serve:
	cd docs && mdbook serve

docs-test:
	cd docs && mdbook test

# Publishing targets
publish-check: fmt test-all docs-test
	cargo publish --dry-run

publish:
	cargo publish

# Combined targets
check-all: fmt clippy test-all docs-book docs-test
	@echo "All checks passed!"

clean:
	cargo clean
	rm -rf docs/book

connect-githooks:
	git config core.hooksPath .githooks

.PHONY: test test-wgpu test-cuda test-cpu fmt clippy docs docs-book docs-serve docs-test publish-check publish check-all clean connect-githooks
