// cargo run --example predict -- model.safetensors < lines.txt
use std::io::BufRead;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: predict MODEL < lines.txt")?;
    let model = bytefast::Model::load(path)?;
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        let top = model.predict(&line, 1, 0.0)?;
        match top.first() {
            Some(p) => println!("{} {:.5}", p.label, p.probability),
            None => println!(),
        }
    }
    Ok(())
}
