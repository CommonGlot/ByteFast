// Compiles the ByteFast C++ library (everything in src/ but the CLI).
fn main() {
    let mut build = cc::Build::new();
    build.cpp(true).std("c++17").define("NDEBUG", None).opt_level(3);
    let mut files: Vec<_> = std::fs::read_dir("src")
        .expect("src/ with the C++ sources")
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension().map_or(false, |e| e == "cc")
                && path.file_name().map_or(false, |n| n != "main.cc")
        })
        .collect();
    files.sort();
    for file in &files {
        println!("cargo:rerun-if-changed={}", file.display());
    }
    println!("cargo:rerun-if-changed=src");
    if !build.get_compiler().is_like_msvc() {
        build.flag("-pthread");
    }
    build.files(&files).compile("bytefast");
}
