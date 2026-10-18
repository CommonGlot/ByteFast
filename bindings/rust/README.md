# bytefast (Rust)

Rust bindings for [ByteFast](https://github.com/CommonGlot/ByteFast): load a
`.safetensors` model and predict labels. The crate compiles the C++ library
with the [`cc`](https://crates.io/crates/cc) crate, so it needs a C++17
compiler.

```toml
[dependencies]
bytefast = "0.2"
```

```rust
let model = bytefast::Model::load("model.safetensors")?;
for p in model.predict("The striker scored twice", 3, 0.0)? {
    println!("{} {:.4}", p.label, p.probability);
}
```

A `Model` is `Send + Sync`, so one loaded model can serve many threads.
Models are trained with the `bytefast` command line or the Python package.
