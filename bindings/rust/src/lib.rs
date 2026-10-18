//! Rust bindings for [ByteFast](https://github.com/CommonGlot/ByteFast), a
//! text classifier with byte, character and word features.
//!
//! ```no_run
//! let model = bytefast::Model::load("model.safetensors")?;
//! for p in model.predict("The striker scored twice", 3, 0.0)? {
//!     println!("{} {:.4}", p.label, p.probability);
//! }
//! # Ok::<(), bytefast::Error>(())
//! ```
//!
//! Models are trained with the `bytefast` command line or the Python
//! package. A loaded model is read-only and can be shared between threads.

use std::ffi::{c_char, CStr, CString};
use std::fmt;
use std::path::Path;
use std::ptr::NonNull;

mod ffi {
    use std::ffi::c_char;

    #[repr(C)]
    pub struct Model {
        _private: [u8; 0],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct Prediction {
        pub label: i32,
        pub probability: f32,
    }

    extern "C" {
        pub fn bytefast_load(path: *const c_char, error: *mut *mut c_char) -> *mut Model;
        pub fn bytefast_free(model: *mut Model);
        pub fn bytefast_free_error(error: *mut c_char);
        pub fn bytefast_predict(
            model: *const Model,
            text: *const c_char,
            length: usize,
            k: i32,
            threshold: f32,
            out: *mut Prediction,
            error: *mut *mut c_char,
        ) -> i32;
        pub fn bytefast_num_labels(model: *const Model) -> i32;
        pub fn bytefast_label(model: *const Model, label: i32) -> *const c_char;
        pub fn bytefast_dimension(model: *const Model) -> i32;
    }
}

/// An error from loading a model or predicting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

fn take_error(error: *mut c_char) -> Error {
    if error.is_null() {
        return Error("unknown error".into());
    }
    let message = unsafe { CStr::from_ptr(error) }.to_string_lossy().into_owned();
    unsafe { ffi::bytefast_free_error(error) };
    Error(message)
}

/// One predicted label and its probability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction<'a> {
    pub label: &'a str,
    pub probability: f32,
}

/// A loaded classifier.
pub struct Model {
    ptr: NonNull<ffi::Model>,
}

// The C++ model is read-only after loading, and bytefast_predict keeps its
// state on the stack, so a model can be used from several threads.
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    /// Loads a `.safetensors` model file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Model, Error> {
        let path = path.as_ref();
        let path = path
            .to_str()
            .ok_or_else(|| Error(format!("path is not UTF-8: {}", path.display())))?;
        let path = CString::new(path).map_err(|_| Error("path contains a NUL byte".into()))?;
        let mut error = std::ptr::null_mut();
        let ptr = unsafe { ffi::bytefast_load(path.as_ptr(), &mut error) };
        NonNull::new(ptr).map(|ptr| Model { ptr }).ok_or_else(|| take_error(error))
    }

    /// The `k` most likely labels of one line of text with probability at
    /// least `threshold`, most likely first. A newline ends the line.
    pub fn predict(&self, text: &str, k: usize, threshold: f32) -> Result<Vec<Prediction<'_>>, Error> {
        let k = k.clamp(1, i32::MAX as usize);
        let mut out = vec![ffi::Prediction::default(); k];
        let mut error = std::ptr::null_mut();
        let n = unsafe {
            ffi::bytefast_predict(
                self.ptr.as_ptr(),
                text.as_ptr() as *const c_char,
                text.len(),
                k as i32,
                threshold,
                out.as_mut_ptr(),
                &mut error,
            )
        };
        if n < 0 {
            return Err(take_error(error));
        }
        Ok(out[..n as usize]
            .iter()
            .map(|p| Prediction {
                label: self.label(p.label as usize).unwrap_or(""),
                probability: p.probability,
            })
            .collect())
    }

    /// The number of labels.
    pub fn num_labels(&self) -> usize {
        unsafe { ffi::bytefast_num_labels(self.ptr.as_ptr()) as usize }
    }

    /// The name of label `index`.
    pub fn label(&self, index: usize) -> Option<&str> {
        let index = i32::try_from(index).ok()?;
        let name = unsafe { ffi::bytefast_label(self.ptr.as_ptr(), index) };
        if name.is_null() {
            return None;
        }
        unsafe { CStr::from_ptr(name) }.to_str().ok()
    }

    /// All label names, in index order.
    pub fn labels(&self) -> Vec<&str> {
        (0..self.num_labels()).filter_map(|i| self.label(i)).collect()
    }

    /// The size of the feature vectors.
    pub fn dimension(&self) -> usize {
        unsafe { ffi::bytefast_dimension(self.ptr.as_ptr()) as usize }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { ffi::bytefast_free(self.ptr.as_ptr()) }
    }
}
