// Needs a model: BYTEFAST_TEST_MODEL=path, BYTEFAST_TEST_TEXT and
// BYTEFAST_TEST_LABEL (the expected top label). CI trains one with the CLI.
use bytefast::Model;

#[test]
fn load_predict_and_labels() {
    let Ok(path) = std::env::var("BYTEFAST_TEST_MODEL") else {
        eprintln!("BYTEFAST_TEST_MODEL is not set; skipped");
        return;
    };
    let text = std::env::var("BYTEFAST_TEST_TEXT").unwrap();
    let expected = std::env::var("BYTEFAST_TEST_LABEL").unwrap();
    let model = Model::load(&path).unwrap();
    assert!(model.dimension() > 0);
    assert!(model.labels().contains(&expected.as_str()));
    let top = model.predict(&text, 3, 0.0).unwrap();
    assert_eq!(top[0].label, expected);
    assert!(top.windows(2).all(|w| w[0].probability >= w[1].probability));
    // A model can be shared between threads.
    std::thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| assert_eq!(model.predict(&text, 1, 0.0).unwrap()[0].label, expected));
        }
    });
    assert!(Model::load("/nonexistent.safetensors").is_err());
}
