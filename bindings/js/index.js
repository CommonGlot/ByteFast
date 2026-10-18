// Copyright (c) 2026 CommonGlot contributors. MIT License.
//
// JavaScript bindings for ByteFast prediction, on the WebAssembly build of
// the C API (src/bytefast_c.h). Works in Node and in browsers:
//
//   import {loadModel} from "bytefast";
//   const model = await loadModel("model.safetensors");  // path, URL or bytes
//   model.predict("The striker scored twice", 3);         // [{label, probability}]
//   model.free();

import createByteFast from "./bytefast_wasm.js";

let modulePromise = null;
const getModule = () => (modulePromise ??= createByteFast());
let fileCount = 0;

const isNode = typeof process !== "undefined" && !!process.versions?.node;

async function readSource(source) {
  if (source instanceof Uint8Array) {
    return source;
  }
  if (source instanceof ArrayBuffer) {
    return new Uint8Array(source);
  }
  const text = String(source);
  if (isNode && !/^https?:\/\//.test(text)) {
    const {readFile} = await import("node:fs/promises");
    return new Uint8Array(await readFile(source instanceof URL ? source : text));
  }
  const response = await fetch(text);
  if (!response.ok) {
    throw new Error(`cannot fetch ${text}: ${response.status}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

function takeError(m, errorPtr) {
  const message = m.getValue(errorPtr, "i32");
  const text = message ? m.UTF8ToString(message) : "unknown error";
  if (message) {
    m._bytefast_free_error(message);
  }
  return new Error(text);
}

export class Model {
  constructor(m, ptr) {
    this._m = m;
    this._ptr = ptr;
    this.labels = [];
    for (let i = 0; i < m._bytefast_num_labels(ptr); i++) {
      this.labels.push(m.UTF8ToString(m._bytefast_label(ptr, i)));
    }
    this.dimension = m._bytefast_dimension(ptr);
  }

  /** Loads a .safetensors model from a path (Node), a URL or its bytes. */
  static async load(source) {
    const m = await getModule();
    const bytes = await readSource(source);
    const path = `/model-${fileCount++}.safetensors`;
    m.FS.writeFile(path, bytes);
    const errorPtr = m._malloc(4);
    const pathPtr = m._malloc(m.lengthBytesUTF8(path) + 1);
    try {
      m.setValue(errorPtr, 0, "i32");
      m.stringToUTF8(path, pathPtr, m.lengthBytesUTF8(path) + 1);
      const ptr = m._bytefast_load(pathPtr, errorPtr);
      if (!ptr) {
        throw takeError(m, errorPtr);
      }
      return new Model(m, ptr);
    } finally {
      m._free(pathPtr);
      m._free(errorPtr);
      m.FS.unlink(path);
    }
  }

  /**
   * The k most likely labels of one line of text with probability at least
   * threshold, most likely first: [{label, probability}, ...].
   */
  predict(text, k = 1, threshold = 0.0) {
    if (!this._ptr) {
      throw new Error("the model was freed");
    }
    const m = this._m;
    k = Math.max(1, k | 0);
    const bytes = new TextEncoder().encode(String(text));
    const textPtr = m._malloc(bytes.length + 1);
    const outPtr = m._malloc(8 * k);
    const errorPtr = m._malloc(4);
    try {
      m.HEAPU8.set(bytes, textPtr);
      m.setValue(errorPtr, 0, "i32");
      const n = m._bytefast_predict(
          this._ptr, textPtr, bytes.length, k, threshold, outPtr, errorPtr);
      if (n < 0) {
        throw takeError(m, errorPtr);
      }
      const out = [];
      for (let i = 0; i < n; i++) {
        out.push({
          label: this.labels[m.getValue(outPtr + 8 * i, "i32")],
          probability: m.getValue(outPtr + 8 * i + 4, "float"),
        });
      }
      return out;
    } finally {
      m._free(textPtr);
      m._free(outPtr);
      m._free(errorPtr);
    }
  }

  /** Releases the model's memory. */
  free() {
    if (this._ptr) {
      this._m._bytefast_free(this._ptr);
      this._ptr = 0;
    }
  }
}

export const loadModel = (source) => Model.load(source);
